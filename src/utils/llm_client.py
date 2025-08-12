"""LLM client for content generation with OpenAI and Anthropic support."""

import asyncio
import hashlib
import json
from typing import Dict, Any, Optional, List
from datetime import datetime

import openai
import anthropic
import tiktoken
from loguru import logger

from src.config.settings import LLMConfig


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic APIs."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        self.encoding = None
        
        # Initialize available clients
        self._initialize_clients()
    
    def _initialize_clients(self) -> None:
        """Initialize LLM API clients."""
        try:
            if self.config.openai_api_key:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=self.config.openai_api_key
                )
                # Initialize tokenizer for OpenAI
                self.encoding = tiktoken.encoding_for_model("gpt-4")
                logger.info("OpenAI client initialized")
            
            if self.config.anthropic_api_key:
                self.anthropic_client = anthropic.AsyncAnthropic(
                    api_key=self.config.anthropic_api_key
                )
                logger.info("Anthropic client initialized")
            
            if not self.openai_client and not self.anthropic_client:
                raise ValueError("At least one LLM API key must be provided")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {e}")
            raise
    
    async def generate_content(
        self,
        prompt: str,
        content_type: str = "post",
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """Generate content using the configured LLM."""
        try:
            # Prepare the full prompt
            full_prompt = self._prepare_prompt(prompt, content_type, context)
            
            # Use configured parameters or defaults
            max_tokens = max_tokens or self.config.max_tokens
            temperature = temperature or self.config.temperature
            
            # Choose model based on preference and availability
            if self.config.preferred_model.startswith("gpt") and self.openai_client:
                result = await self._generate_openai(full_prompt, max_tokens, temperature)
            elif self.config.preferred_model.startswith("claude") and self.anthropic_client:
                result = await self._generate_anthropic(full_prompt, max_tokens, temperature)
            elif self.openai_client:
                result = await self._generate_openai(full_prompt, max_tokens, temperature)
            elif self.anthropic_client:
                result = await self._generate_anthropic(full_prompt, max_tokens, temperature)
            else:
                raise ValueError("No LLM client available")
            
            # Add metadata
            result.update({
                "content_type": content_type,
                "prompt_hash": hashlib.sha256(full_prompt.encode()).hexdigest()[:16],
                "generated_at": datetime.utcnow().isoformat(),
                "model_used": result.get("model", "unknown")
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise
    
    def _prepare_prompt(
        self, 
        base_prompt: str, 
        content_type: str, 
        context: Optional[str] = None
    ) -> str:
        """Prepare the full prompt with context and instructions."""
        
        # Domain-specific context
        domain_context = (
            "You are creating content for a Twitter bot focused on New Orleans culture, "
            "music scene, and GenZ trends. The content should be authentic, engaging, "
            "and reflect the vibrant culture of New Orleans."
        )
        
        # Content type specific instructions
        if content_type == "post":
            type_instructions = (
                "Create an original tweet (under 280 characters) that's conversational "
                "and engaging. Avoid excessive hashtags or promotional language. "
                "Focus on local culture, music events, food, or GenZ topics."
            )
        elif content_type == "comment":
            type_instructions = (
                "Create a thoughtful reply that adds value to the conversation. "
                "Be supportive, authentic, and brief. Match the tone of the original post."
            )
        elif content_type == "repost_comment":
            type_instructions = (
                "Create a brief comment to accompany a repost. Add insight, context, "
                "or your perspective while keeping it concise and engaging."
            )
        else:
            type_instructions = "Create engaging social media content."
        
        # Combine all parts
        full_prompt = f"""
{domain_context}

{type_instructions}

{base_prompt}
"""
        
        if context:
            full_prompt += f"\n\nContext: {context}"
        
        full_prompt += (
            "\n\nImportant: Keep the tone casual, authentic, and aligned with GenZ "
            "communication style. Avoid corporate or overly promotional language."
        )
        
        return full_prompt.strip()
    
    async def _generate_openai(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Generate content using OpenAI API."""
        try:
            response = await self.openai_client.chat.completions.create(
                model=self.config.preferred_model if self.config.preferred_model.startswith("gpt") else "gpt-4",
                messages=[
                    {"role": "system", "content": "You are a creative social media content generator."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            return {
                "content": content,
                "model": response.model,
                "usage": response.usage.dict() if response.usage else {},
                "finish_reason": response.choices[0].finish_reason,
                "provider": "openai"
            }
            
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
    
    async def _generate_anthropic(
        self, 
        prompt: str, 
        max_tokens: int, 
        temperature: float
    ) -> Dict[str, Any]:
        """Generate content using Anthropic API."""
        try:
            response = await self.anthropic_client.messages.create(
                model=self.config.preferred_model if self.config.preferred_model.startswith("claude") else "claude-3-sonnet-20240229",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            content = response.content[0].text.strip()
            
            return {
                "content": content,
                "model": response.model,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "finish_reason": response.stop_reason,
                "provider": "anthropic"
            }
            
        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text (OpenAI tokenizer as approximation)."""
        if self.encoding:
            return len(self.encoding.encode(text))
        # Rough approximation if no tokenizer available
        return len(text.split()) * 1.3
    
    def validate_content(self, content: str, content_type: str) -> Dict[str, Any]:
        """Validate generated content for social media use."""
        validation = {
            "valid": True,
            "issues": [],
            "character_count": len(content),
            "word_count": len(content.split())
        }
        
        # Twitter character limit
        if content_type in ["post", "comment"] and len(content) > 280:
            validation["valid"] = False
            validation["issues"].append(f"Content too long: {len(content)} characters (max 280)")
        
        # Check for potentially problematic content
        problematic_patterns = [
            r'http[s]?://[^\s]+',  # URLs without context
            r'@[a-zA-Z0-9_]+',     # Mentions (could be problematic)
            r'#\w+\s*#\w+\s*#\w+', # Too many hashtags
        ]
        
        import re
        for pattern in problematic_patterns:
            if re.search(pattern, content):
                validation["issues"].append(f"Contains pattern: {pattern}")
        
        # Check for spam indicators
        spam_indicators = ["click here", "follow me", "dm me", "check out", "buy now"]
        for indicator in spam_indicators:
            if indicator.lower() in content.lower():
                validation["issues"].append(f"Contains spam indicator: {indicator}")
        
        # Update validity based on issues
        if validation["issues"]:
            validation["valid"] = len(validation["issues"]) <= 1  # Allow minor issues
        
        return validation
    
    async def generate_variations(
        self, 
        base_content: str, 
        count: int = 3,
        content_type: str = "post"
    ) -> List[Dict[str, Any]]:
        """Generate variations of content."""
        variations = []
        
        for i in range(count):
            try:
                prompt = f"Create a variation of this {content_type}: {base_content}"
                result = await self.generate_content(
                    prompt, 
                    content_type,
                    temperature=0.8 + (i * 0.1)  # Increase creativity for variations
                )
                variations.append(result)
                
                # Small delay between generations
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to generate variation {i+1}: {e}")
                continue
        
        return variations
    
    async def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze sentiment of content using LLM."""
        try:
            prompt = f"""
Analyze the sentiment of this social media content and provide a score from -1 (very negative) to 1 (very positive):

Content: "{content}"

Respond with just a JSON object containing:
- sentiment_score: number between -1 and 1
- sentiment_label: "positive", "negative", or "neutral"
- confidence: number between 0 and 1
- reasoning: brief explanation
"""
            
            result = await self.generate_content(
                prompt, 
                content_type="analysis",
                max_tokens=150,
                temperature=0.1  # Low temperature for consistent analysis
            )
            
            # Parse JSON response
            try:
                analysis = json.loads(result["content"])
                return analysis
            except json.JSONDecodeError:
                # Fallback simple analysis
                return {
                    "sentiment_score": 0.0,
                    "sentiment_label": "neutral",
                    "confidence": 0.5,
                    "reasoning": "Could not parse detailed analysis"
                }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "reasoning": f"Analysis failed: {e}"
            }
    
    def get_client_status(self) -> Dict[str, Any]:
        """Get status of LLM clients."""
        return {
            "openai_available": self.openai_client is not None,
            "anthropic_available": self.anthropic_client is not None,
            "preferred_model": self.config.preferred_model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }