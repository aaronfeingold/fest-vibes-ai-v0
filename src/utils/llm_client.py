"""LLM client for content generation with OpenAI and Anthropic support."""

import asyncio
import hashlib
import json
import random
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

import openai
import anthropic
import tiktoken
from loguru import logger
from openai import RateLimitError, APIError
from sentence_transformers import SentenceTransformer

from src.config.settings import LLMConfig
from src.prompts.base_prompts import BasePrompts


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic APIs."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.openai_client = None
        self.anthropic_client = None
        self.encoding = None
        self.sentence_transformer = None

        # Advanced rate limiting tracking
        self.last_request_time = 0
        self.request_timestamps = []
        self.token_timestamps = []
        self.token_usage_window = []

        # Rate limits for different tiers (conservative estimates for free tier)
        self.max_requests_per_minute = 3
        self.max_tokens_per_minute = 200000  # Very conservative

        # Exponential backoff settings
        self.retry_count = 0
        self.max_retries = 3
        self.base_delay = 2  # Base delay in seconds
        self.max_delay = 60  # Maximum delay in seconds

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

            # Initialize sentence transformer for embeddings
            try:
                self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Sentence transformer initialized for embedding generation")
            except Exception as e:
                logger.warning(f"Failed to initialize sentence transformer: {e}")
                self.sentence_transformer = None

        except Exception as e:
            logger.error(f"Failed to initialize LLM clients: {e}")
            raise

    def _clean_old_timestamps(self) -> None:
        """Remove timestamps older than 1 minute."""
        cutoff = time.time() - 60
        self.request_timestamps = [ts for ts in self.request_timestamps if ts > cutoff]

        # Clean token usage window (keep last minute)
        cutoff_datetime = datetime.now() - timedelta(minutes=1)
        self.token_usage_window = [
            entry
            for entry in self.token_usage_window
            if entry["timestamp"] > cutoff_datetime
        ]

    def _get_current_token_usage(self) -> int:
        """Get current token usage in the last minute."""
        self._clean_old_timestamps()
        return sum(entry["tokens"] for entry in self.token_usage_window)

    def _can_make_request(self, estimated_tokens: int = 0) -> tuple[bool, float]:
        """Check if we can make a request and return wait time if not."""
        self._clean_old_timestamps()
        current_time = time.time()

        # Check request rate limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            oldest_request = min(self.request_timestamps)
            wait_time = 60 - (current_time - oldest_request) + random.uniform(1, 3)
            return False, wait_time

        # Check token rate limit
        current_tokens = self._get_current_token_usage()
        if current_tokens + estimated_tokens > self.max_tokens_per_minute:
            wait_time = 60 + random.uniform(1, 5)  # Wait for window to reset
            return False, wait_time

        # Check minimum delay between requests (20 seconds)
        if self.last_request_time > 0:
            time_since_last = current_time - self.last_request_time
            if time_since_last < 20:
                wait_time = 20 - time_since_last + random.uniform(1, 3)
                return False, wait_time

        return True, 0

    async def _wait_for_rate_limit(self, estimated_tokens: int = 0) -> None:
        """Enforce sophisticated rate limiting with token awareness."""
        can_proceed, wait_time = self._can_make_request(estimated_tokens)

        if not can_proceed:
            logger.info(
                f"Rate limiting: waiting {wait_time:.1f}s "
                f"(tokens used: {self._get_current_token_usage()}/{self.max_tokens_per_minute})"
            )
            await asyncio.sleep(wait_time)
            # Recursive check after waiting
            await self._wait_for_rate_limit(estimated_tokens)

        # Record the request
        current_time = time.time()
        self.request_timestamps.append(current_time)
        self.last_request_time = current_time

    def _record_token_usage(self, tokens_used: int) -> None:
        """Record token usage for rate limiting."""
        self.token_usage_window.append(
            {"timestamp": datetime.now(), "tokens": tokens_used}
        )

    async def _exponential_backoff_retry(self, func, *args, **kwargs):
        """Execute function with exponential backoff on rate limit errors."""
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except RateLimitError as e:
                if attempt == self.max_retries:
                    logger.error(f"Max retries exceeded for rate limit: {e}")
                    raise

                # Calculate exponential backoff with jitter
                delay = min(
                    self.base_delay * (2**attempt) + random.uniform(0, 1),
                    self.max_delay,
                )
                logger.warning(
                    f"Rate limit hit, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries + 1})"
                )
                await asyncio.sleep(delay)
            except APIError as e:
                logger.error(f"API error: {e}")
                raise
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
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

            # Estimate tokens for rate limiting
            estimated_tokens = self.count_tokens(full_prompt) + max_tokens

            # Apply sophisticated rate limiting before making request
            await self._wait_for_rate_limit(estimated_tokens)

            # Choose model and execute with exponential backoff
            if self.config.preferred_model.startswith("gpt") and self.openai_client:
                result = await self._exponential_backoff_retry(
                    self._generate_openai, full_prompt, max_tokens, temperature
                )
            elif self.config.preferred_model.startswith("claude") and self.anthropic_client:
                result = await self._exponential_backoff_retry(
                    self._generate_anthropic, full_prompt, max_tokens, temperature
                )
            elif self.openai_client:
                result = await self._exponential_backoff_retry(
                    self._generate_openai, full_prompt, max_tokens, temperature
                )
            elif self.anthropic_client:
                result = await self._exponential_backoff_retry(
                    self._generate_anthropic, full_prompt, max_tokens, temperature
                )
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
        self, base_prompt: str, content_type: str, context: Optional[str] = None
    ) -> str:
        """Prepare the full prompt with context and instructions using centralized prompt system."""
        return BasePrompts.prepare_llm_prompt(base_prompt, content_type, context)

    # Collaboration-friendly methods for non-coders to modify domain context
    def get_domain_context(self) -> str:
        """Get the current domain context (for non-coder collaboration)."""
        return BasePrompts.get_domain_context()

    def update_domain_context(self, new_context: str) -> None:
        """Update domain context (for non-coder collaboration)."""
        BasePrompts.update_domain_context(new_context)
        logger.info("Domain context updated by non-technical collaborator")

    def get_content_type_instruction(self, content_type: str) -> str:
        """Get instruction for specific content type (for non-coder collaboration)."""
        return BasePrompts.get_content_type_instruction(content_type)

    def update_content_type_instruction(
        self, content_type: str, instruction: str
    ) -> None:
        """Update instruction for specific content type (for non-coder collaboration)."""
        BasePrompts.update_content_type_instruction(content_type, instruction)
        logger.info(
            f"Content type '{content_type}' instruction updated by non-technical collaborator"
        )

    def list_available_content_types(self) -> List[str]:
        """List all available content types (for non-coder collaboration)."""
        return list(BasePrompts.CONTENT_TYPE_INSTRUCTIONS.keys())

    async def _generate_openai(
        self, prompt: str, max_tokens: int, temperature: float
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

            # Record token usage for rate limiting
            if response.usage:
                total_tokens = response.usage.total_tokens
                self._record_token_usage(total_tokens)

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
        self, prompt: str, max_tokens: int, temperature: float
    ) -> Dict[str, Any]:
        """Generate content using Anthropic API."""
        try:
            model_name = (
                self.config.preferred_model
                if self.config.preferred_model.startswith("claude")
                else "claude-3-sonnet-20240229"
            )
            response = await self.anthropic_client.messages.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
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
        return int(len(text.split()) * 1.3)

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
            r"http[s]?://[^\s]+",  # URLs without context
            r"@[a-zA-Z0-9_]+",  # Mentions (could be problematic)
            r"#\w+\s*#\w+\s*#\w+",  # Too many hashtags
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
        self, base_content: str, count: int = 3, content_type: str = "post"
    ) -> List[Dict[str, Any]]:
        """Generate variations of content."""
        variations = []

        for i in range(count):
            try:
                prompt = f"Create a variation of this {content_type}: {base_content}"
                result = await self.generate_content(
                    prompt,
                    content_type,
                    temperature=0.8 + (i * 0.1),  # Increase creativity for variations
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
                temperature=0.1,  # Low temperature for consistent analysis
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

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate 384-dimensional embedding vector using sentence-transformers all-MiniLM-L6-v2 model."""
        try:
            if not self.sentence_transformer:
                logger.warning(
                    "Sentence transformer not available for embedding generation"
                )
                return None

            # Generate embedding using sentence-transformers (CPU-based, no rate limiting needed)
            # This is a synchronous operation, but we run it in executor to avoid blocking
            loop = asyncio.get_event_loop()

            # Run the synchronous embedding generation in thread pool
            embedding = await loop.run_in_executor(
                None, self.sentence_transformer.encode, text
            )

            # Convert numpy array to list of floats
            embedding_list = embedding.tolist()

            logger.debug(
                f"Generated embedding for text: {text[:50]}... "
                f"(dimensions: {len(embedding_list)}, model: all-MiniLM-L6-v2)"
            )

            # Verify we have the expected 384 dimensions
            if len(embedding_list) != 384:
                logger.warning(
                    f"Unexpected embedding dimensions: {len(embedding_list)} (expected 384)"
                )

            return embedding_list

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def get_client_status(self) -> Dict[str, Any]:
        """Get status of LLM clients."""
        return {
            "openai_available": self.openai_client is not None,
            "anthropic_available": self.anthropic_client is not None,
            "sentence_transformer_available": self.sentence_transformer is not None,
            "preferred_model": self.config.preferred_model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
        }
