"""Content Agent for generating original posts and contextual comments."""

import asyncio
import random
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import (
    ContentCache, ContentType, PerformanceMetrics,
    ContentGenerationRequest, EngagementHistory, ActionType
)
from src.utils.llm_client import LLMClient
from src.utils.rate_limiter import agent_queue_size


class ContentAgent(BaseAgent):
    """Agent responsible for content generation and management."""
    
    def __init__(self, *args, **kwargs):
        super().__init__("ContentAgent", *args, **kwargs)
        
        # Initialize LLM client
        self.llm_client = LLMClient(self.config.llm)
        
        # Content queues
        self.post_queue = asyncio.Queue()
        self.comment_queue = asyncio.Queue()
        
        # Content templates and inspiration
        self.content_themes = {
            "music": [
                "What's your favorite New Orleans music venue?",
                "That jazz sound hits different in the French Quarter",
                "Bounce music keeping the city alive",
                "Local musicians deserve more recognition",
                "Nothing like a second line on Sunday",
                "The brass bands here are unmatched"
            ],
            "culture": [
                "New Orleans energy is unmatched",
                "This city teaches you how to live",
                "Where else can you get beignets at 3am?",
                "The architecture tells stories",
                "Every neighborhood has its own vibe",
                "Mardi Gras isn't just a season, it's a lifestyle"
            ],
            "food": [
                "Po-boy debates are serious business here",
                "Café du Monde or local spot?",
                "Crawfish season hits different",
                "Creole vs Cajun - the eternal question",
                "Sunday brunch is a religion here",
                "Food truck finds in the Quarter"
            ],
            "genz": [
                "This city gets Gen Z creativity",
                "TikTok can't capture the real NOLA vibe",
                "Young artists carrying on traditions",
                "Finding community in unexpected places",
                "Authenticity over everything",
                "Creating art from struggle"
            ],
            "events": [
                "Festival season is approaching",
                "Free music everywhere you turn",
                "Support local venues",
                "The underground scene is thriving",
                "Art markets and music collide",
                "Community over everything"
            ]
        }
        
        # Contextual response patterns
        self.response_patterns = {
            "supportive": [
                "This is so important",
                "Needed to hear this today",
                "Absolutely love this perspective",
                "You're speaking truth",
                "This resonates deeply"
            ],
            "enthusiastic": [
                "This is everything!",
                "So here for this energy",
                "Love seeing this representation",
                "This makes my heart full",
                "Perfectly said"
            ],
            "curious": [
                "Tell me more about this",
                "What's your experience been?",
                "I'm intrigued by this take",
                "Help me understand better",
                "What led you to this insight?"
            ],
            "relatable": [
                "Felt this in my soul",
                "Why is this so accurate",
                "Calling me out with this truth",
                "Living this reality daily",
                "This hits different"
            ]
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Execute content agent workflow."""
        results = {
            "posts_generated": 0,
            "comments_generated": 0,
            "posts_published": 0,
            "cache_maintained": False,
            "errors": []
        }
        
        try:
            # 1. Maintain content cache
            cache_results = await self._maintain_content_cache()
            results["posts_generated"] = cache_results["posts_generated"]
            results["comments_generated"] = cache_results["comments_generated"]
            results["cache_maintained"] = True
            
            # 2. Publish scheduled posts
            publish_results = await self._publish_scheduled_content()
            results["posts_published"] = publish_results["published"]
            
            # 3. Generate contextual comments for timeline
            await self._generate_contextual_comments()
            
            # 4. Update content performance metrics
            await self._update_content_performance()
            
        except Exception as e:
            logger.error(f"Content agent execution failed: {e}")
            results["errors"].append(str(e))
            raise
        
        return results
    
    async def _maintain_content_cache(self) -> Dict[str, Any]:
        """Maintain content cache at optimal levels."""
        posts_generated = 0
        comments_generated = 0
        
        try:
            # Check current cache levels
            unused_posts = await self.mongodb.get_unused_content(ContentType.POST.value, limit=100)
            unused_comments = await self.mongodb.get_unused_content(ContentType.COMMENT.value, limit=100)
            
            # Generate posts if cache is low
            if len(unused_posts) < self.config.agents.min_content_cache_size:
                needed = self.config.agents.max_content_cache_size - len(unused_posts)
                posts_generated = await self._generate_original_posts(needed)
            
            # Generate comments if cache is low
            if len(unused_comments) < self.config.agents.min_content_cache_size:
                needed = self.config.agents.max_content_cache_size - len(unused_comments)
                comments_generated = await self._generate_template_comments(needed)
            
            logger.info(f"Content cache maintained: {posts_generated} posts, {comments_generated} comments generated")
            
        except Exception as e:
            logger.error(f"Failed to maintain content cache: {e}")
            raise
        
        return {
            "posts_generated": posts_generated,
            "comments_generated": comments_generated
        }
    
    async def _generate_original_posts(self, count: int) -> int:
        """Generate original posts for the cache."""
        generated = 0
        
        try:
            for _ in range(count):
                if not self.safety_limits.can_perform_action("posts"):
                    logger.warning("Daily post generation limit reached")
                    break
                
                # Select random theme and inspiration
                theme = random.choice(list(self.content_themes.keys()))
                inspiration = random.choice(self.content_themes[theme])
                
                # Create generation request
                request = ContentGenerationRequest(
                    content_type=ContentType.POST,
                    context=f"Theme: {theme}. Inspiration: {inspiration}",
                    tone="casual",
                    max_length=280
                )
                
                # Generate content
                content_result = await self._generate_content_with_validation(request)
                
                if content_result["valid"]:
                    # Create cache entry
                    content_cache = ContentCache(
                        content_type=ContentType.POST,
                        content=content_result["content"],
                        prompt_used=content_result.get("prompt", ""),
                        tags=[theme],
                        word_count=content_result.get("word_count", 0),
                        character_count=content_result.get("character_count", 0),
                        sentiment_score=content_result.get("sentiment_score", 0.0)
                    )
                    
                    await self.mongodb.create_content(content_cache)
                    generated += 1
                    
                    logger.debug(f"Generated post: {content_result['content'][:50]}...")
                
                # Add delay to avoid rate limits
                await asyncio.sleep(random.uniform(2, 5))
        
        except Exception as e:
            logger.error(f"Failed to generate original posts: {e}")
            raise
        
        return generated
    
    async def _generate_template_comments(self, count: int) -> int:
        """Generate template comments for various contexts."""
        generated = 0
        
        try:
            for _ in range(count):
                # Select random response pattern
                pattern_type = random.choice(list(self.response_patterns.keys()))
                base_response = random.choice(self.response_patterns[pattern_type])
                
                # Generate a more natural variation
                request = ContentGenerationRequest(
                    content_type=ContentType.COMMENT,
                    context=f"Create a {pattern_type} comment variation of: {base_response}",
                    tone="conversational",
                    max_length=100
                )
                
                content_result = await self._generate_content_with_validation(request)
                
                if content_result["valid"]:
                    content_cache = ContentCache(
                        content_type=ContentType.COMMENT,
                        content=content_result["content"],
                        prompt_used=content_result.get("prompt", ""),
                        tags=[pattern_type],
                        word_count=content_result.get("word_count", 0),
                        character_count=content_result.get("character_count", 0),
                        sentiment_score=content_result.get("sentiment_score", 0.0)
                    )
                    
                    await self.mongodb.create_content(content_cache)
                    generated += 1
                
                await asyncio.sleep(random.uniform(1, 3))
        
        except Exception as e:
            logger.error(f"Failed to generate template comments: {e}")
            raise
        
        return generated
    
    async def _generate_content_with_validation(self, request: ContentGenerationRequest) -> Dict[str, Any]:
        """Generate content with validation and retry logic."""
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                # Create prompt based on request
                prompt = self.config.get_content_generation_prompt(request.content_type.value)
                if request.context:
                    prompt += f"\n\nContext: {request.context}"
                
                # Generate content
                llm_result = await self.llm_client.generate_content(
                    prompt,
                    content_type=request.content_type.value,
                    context=request.context,
                    max_tokens=min(request.max_length * 2, 500),  # Give LLM room to work
                    temperature=0.7 + (attempt * 0.1)  # Increase creativity on retries
                )
                
                content = llm_result["content"]
                
                # Validate content
                validation = self.llm_client.validate_content(content, request.content_type.value)
                
                if validation["valid"]:
                    # Analyze sentiment
                    sentiment = await self.llm_client.analyze_sentiment(content)
                    
                    return {
                        "content": content,
                        "valid": True,
                        "prompt": prompt,
                        "word_count": validation["word_count"],
                        "character_count": validation["character_count"],
                        "sentiment_score": sentiment.get("sentiment_score", 0.0),
                        "model_used": llm_result.get("model", "unknown"),
                        "attempt": attempt + 1
                    }
                else:
                    logger.warning(f"Content validation failed (attempt {attempt + 1}): {validation['issues']}")
                    if attempt == max_attempts - 1:
                        return {
                            "content": content,
                            "valid": False,
                            "issues": validation["issues"],
                            "attempt": attempt + 1
                        }
            
            except Exception as e:
                logger.error(f"Content generation attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise
                
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return {"content": "", "valid": False, "attempt": max_attempts}
    
    async def _publish_scheduled_content(self) -> Dict[str, Any]:
        """Publish scheduled content."""
        published = 0
        
        try:
            if not self.safety_limits.can_perform_action("posts"):
                logger.info("Daily post limit reached, skipping content publishing")
                return {"published": 0}
            
            # Get unused posts from cache
            available_posts = await self.mongodb.get_unused_content(ContentType.POST.value, limit=5)
            
            if not available_posts:
                logger.warning("No posts available for publishing")
                return {"published": 0}
            
            # Select best post based on sentiment and theme diversity
            selected_post = self._select_best_post(available_posts)
            
            if selected_post:
                success = await self._publish_post(selected_post)
                if success:
                    published = 1
        
        except Exception as e:
            logger.error(f"Failed to publish scheduled content: {e}")
            raise
        
        return {"published": published}
    
    def _select_best_post(self, posts: List[ContentCache]) -> Optional[ContentCache]:
        """Select the best post from available options."""
        if not posts:
            return None
        
        # Score posts based on various factors
        scored_posts = []
        
        for post in posts:
            score = 0.0
            
            # Prefer positive sentiment
            if post.sentiment_score and post.sentiment_score > 0:
                score += post.sentiment_score * 0.3
            
            # Prefer optimal length (not too short, not too long)
            char_count = post.character_count or len(post.content)
            if 50 <= char_count <= 200:
                score += 0.3
            elif 200 < char_count <= 280:
                score += 0.2
            
            # Prefer content with diverse themes
            # (This would require tracking recently posted themes)
            score += 0.2  # Base score for variety
            
            # Add some randomness to avoid patterns
            score += random.uniform(0, 0.2)
            
            scored_posts.append((post, score))
        
        # Sort by score and return best
        scored_posts.sort(key=lambda x: x[1], reverse=True)
        return scored_posts[0][0]
    
    async def _publish_post(self, post: ContentCache) -> bool:
        """Publish a post to Twitter."""
        try:
            # Make tweet API call
            result = await self.safe_twitter_call(
                "create_tweet",
                "bot_post",
                text=post.content
            )
            
            # Mark content as used
            await self.mongodb.mark_content_used(
                str(post.id),
                {"published_at": datetime.utcnow().isoformat()}
            )
            
            # Record engagement history
            engagement = EngagementHistory(
                target_tweet_id=str(result.get("id", "")) if result else "",
                target_user_id="",  # Self-post
                action_type=ActionType.COMMENT,  # Using COMMENT for posts
                success=True,
                response_data={"result": str(result) if result else "dry_run"},
                content_used=post.content
            )
            
            await self.mongodb.create_engagement(engagement)
            
            logger.info(f"Published post: {post.content[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish post: {e}")
            
            # Record failed engagement
            engagement = EngagementHistory(
                target_tweet_id="",
                target_user_id="",
                action_type=ActionType.COMMENT,
                success=False,
                error_message=str(e),
                content_used=post.content
            )
            
            await self.mongodb.create_engagement(engagement)
            return False
    
    async def _generate_contextual_comments(self) -> None:
        """Generate contextual comments for timeline interactions."""
        try:
            # This would be called by the engagement agent when needed
            # For now, we ensure we have enough template comments
            unused_comments = await self.mongodb.get_unused_content(ContentType.COMMENT.value, limit=20)
            
            if len(unused_comments) < 10:
                logger.info("Generating additional contextual comments")
                await self._generate_template_comments(10)
        
        except Exception as e:
            logger.error(f"Failed to generate contextual comments: {e}")
    
    async def generate_contextual_comment(self, tweet_context: Dict[str, Any]) -> Optional[str]:
        """Generate a contextual comment for a specific tweet."""
        try:
            tweet_text = tweet_context.get("text", "")
            author_info = tweet_context.get("author", {})
            
            # Create contextual prompt
            prompt = f"""
Create a thoughtful, authentic reply to this tweet. Be supportive and engaging 
while staying true to New Orleans culture and GenZ communication style.

Tweet: "{tweet_text}"
Author info: {author_info.get('description', 'No bio available')}

Generate a natural, conversational response that adds value to the conversation.
"""
            
            request = ContentGenerationRequest(
                content_type=ContentType.COMMENT,
                context=prompt,
                tone="conversational",
                max_length=200
            )
            
            result = await self._generate_content_with_validation(request)
            
            if result["valid"]:
                return result["content"]
            else:
                # Fallback to template comment
                template_comments = await self.mongodb.get_unused_content(
                    ContentType.COMMENT.value, 
                    limit=5
                )
                if template_comments:
                    return random.choice(template_comments).content
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate contextual comment: {e}")
            return None
    
    async def _update_content_performance(self) -> None:
        """Update performance metrics for published content."""
        try:
            # Get recent engagements to update content performance
            recent_engagements = await self.mongodb.get_recent_engagements(
                action_type=ActionType.COMMENT.value,  # Posts recorded as comments
                hours=24
            )
            
            for engagement in recent_engagements:
                if engagement.content_used and engagement.success:
                    # This would ideally fetch actual Twitter metrics
                    # For now, we simulate based on engagement success
                    performance_update = {
                        "engagement_rate": random.uniform(0.02, 0.05),  # 2-5%
                        "last_updated": datetime.utcnow().isoformat()
                    }
                    
                    # Find and update the content cache entry
                    # This is a simplified version - in production, we'd need better tracking
        
        except Exception as e:
            logger.error(f"Failed to update content performance: {e}")
    
    async def _should_run(self) -> bool:
        """Check if content agent should run."""
        if not await super()._should_run():
            return False
        
        # Check if we have capacity for content generation
        remaining_posts = self.safety_limits.get_remaining_actions("posts")
        if remaining_posts["daily"] <= 0:
            logger.debug("Daily post limit reached")
            return False
        
        return True
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current content agent status."""
        base_status = self.get_health_status()
        
        # Update queue metrics
        total_queue_size = self.post_queue.qsize() + self.comment_queue.qsize()
        agent_queue_size.labels(agent=self.name).set(total_queue_size)
        
        content_status = {
            **base_status,
            "queue_sizes": {
                "posts": self.post_queue.qsize(),
                "comments": self.comment_queue.qsize()
            },
            "llm_client": self.llm_client.get_client_status(),
            "content_themes": len(self.content_themes),
            "response_patterns": len(self.response_patterns),
            "safety_limits": {
                "posts": self.safety_limits.get_remaining_actions("posts")
            }
        }
        
        return content_status