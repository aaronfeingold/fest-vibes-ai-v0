"""Engagement Agent for strategic likes, reposts, and comments."""

import asyncio
import random
import re
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import (
    EngagementHistory,
    ActionType,
    EngagementDecision,
    ContentType,
)
from src.prompts.agent_prompts import EngagementPrompts


class EngagementAgent(BaseAgent):
    """Agent responsible for strategic social media engagement."""

    def __init__(self, *args, **kwargs):
        super().__init__("EngagementAgent", *args, **kwargs)

        # Action queues with priority
        self.like_queue = asyncio.PriorityQueue()
        self.repost_queue = asyncio.PriorityQueue()
        self.comment_queue = asyncio.PriorityQueue()

        # Engagement scoring factors
        self.engagement_weights = {
            "author_relevance": 0.25,  # How relevant is the author to our domain
            "content_relevance": 0.25,  # How relevant is the content
            "engagement_potential": 0.20,  # How likely to generate engagement
            "timing": 0.15,  # How fresh/timely is the content
            "authenticity": 0.15,  # How authentic/non-spammy it appears
        }

        # Keywords that indicate relevance to our domain
        self.relevance_keywords = {
            "high_value": [
                "new orleans",
                "nola",
                "jazz",
                "music",
                "festival",
                "bounce",
                "brass band",
                "french quarter",
                "mardi gras",
                "local artist",
                "underground",
                "scene",
            ],
            "medium_value": [
                "louisiana",
                "south",
                "culture",
                "art",
                "food",
                "community",
                "creative",
                "authentic",
                "real",
                "support local",
            ],
            "negative": [
                "follow me",
                "dm me",
                "check out",
                "buy now",
                "click here",
                "spam",
                "bot",
                "fake",
                "scam",
            ],
        }

        # Patterns that indicate good engagement opportunities
        self.engagement_patterns = {
            "questions": r"\?",
            "calls_for_support": r"(support|help|boost|share)",
            "local_events": r"(tonight|today|this weekend|upcoming)",
            "emotional_content": r"(love|amazing|incredible|beautiful|proud)",
            "community_building": r"(together|community|family|crew|team)",
        }

        # Content that should be avoided
        self.avoid_patterns = {
            "political": r"(politics|election|vote|democrat|republican)",
            "controversial": r"(hate|angry|fight|drama|beef)",
            "spam_indicators": r"(follow back|f4f|l4l|sub4sub|dm for)",
            "overly_promotional": r"(buy|sale|discount|limited time|act now)",
        }

    async def execute(self) -> Dict[str, Any]:
        """Execute engagement agent workflow."""
        results = {
            "tweets_analyzed": 0,
            "likes_executed": 0,
            "reposts_executed": 0,
            "comments_executed": 0,
            "engagement_opportunities": 0,
            "errors": [],
        }

        try:
            # 1. Monitor timeline for engagement opportunities
            timeline_results = await self._analyze_timeline()
            results["tweets_analyzed"] = timeline_results["analyzed"]
            results["engagement_opportunities"] = timeline_results["opportunities"]

            # 2. Execute queued likes
            like_results = await self._execute_likes()
            results["likes_executed"] = like_results["executed"]

            # 3. Execute queued reposts
            repost_results = await self._execute_reposts()
            results["reposts_executed"] = repost_results["executed"]

            # 4. Execute queued comments
            comment_results = await self._execute_comments()
            results["comments_executed"] = comment_results["executed"]

            # 5. Monitor engagement health
            await self._monitor_engagement_health()

        except Exception as e:
            logger.error(f"Engagement agent execution failed: {e}")
            results["errors"].append(str(e))
            raise

        return results

    async def _analyze_timeline(self) -> Dict[str, Any]:
        """Analyze timeline tweets for engagement opportunities."""
        analyzed = 0
        opportunities = 0

        try:
            # Get timeline tweets from users we follow
            timeline_tweets = await self.get_timeline_tweets(count=50)

            for tweet in timeline_tweets:
                analyzed += 1

                # Score tweet for different engagement types
                like_decision = await self._analyze_like_opportunity(tweet)
                repost_decision = await self._analyze_repost_opportunity(tweet)
                comment_decision = await self._analyze_comment_opportunity(tweet)

                # Queue high-scoring opportunities
                if like_decision.decision:
                    priority = int(
                        (1.0 - like_decision.score) * 100
                    )  # Lower score = higher priority
                    await self.like_queue.put((priority, tweet, like_decision))
                    opportunities += 1

                if repost_decision.decision:
                    priority = int((1.0 - repost_decision.score) * 100)
                    await self.repost_queue.put((priority, tweet, repost_decision))
                    opportunities += 1

                if comment_decision.decision:
                    priority = int((1.0 - comment_decision.score) * 100)
                    await self.comment_queue.put((priority, tweet, comment_decision))
                    opportunities += 1

                # Record analysis metrics
                await self._record_metric(
                    "engagement_analysis",
                    max(
                        like_decision.score,
                        repost_decision.score,
                        comment_decision.score,
                    ),
                    {
                        "tweet_id": tweet["id"],
                        "like_score": like_decision.score,
                        "repost_score": repost_decision.score,
                        "comment_score": comment_decision.score,
                    },
                )

        except Exception as e:
            logger.error(f"Timeline analysis failed: {e}")
            raise

        logger.info(
            f"Analyzed {analyzed} tweets, found {opportunities} engagement opportunities"
        )
        return {"analyzed": analyzed, "opportunities": opportunities}

    async def _analyze_like_opportunity(
        self, tweet: Dict[str, Any]
    ) -> EngagementDecision:
        """Analyze tweet for like opportunity."""
        factors = {}

        # Author relevance
        author_score = await self._score_author_relevance(tweet["author_id"])
        factors["author_relevance"] = author_score

        # Content relevance
        content_score = self._score_content_relevance(tweet["text"])
        factors["content_relevance"] = content_score

        # Engagement potential (lower barrier for likes)
        engagement_score = self._score_engagement_potential(tweet, action_type="like")
        factors["engagement_potential"] = engagement_score

        # Timing score
        timing_score = self._score_timing(tweet.get("created_at"))
        factors["timing"] = timing_score

        # Authenticity score
        authenticity_score = self._score_authenticity(tweet["text"])
        factors["authenticity"] = authenticity_score

        # Calculate weighted overall score
        overall_score = sum(
            score * self.engagement_weights[factor] for factor, score in factors.items()
        )

        # Decision threshold for likes (lower than reposts/comments)
        decision = overall_score >= self.config.agents.like_threshold

        return EngagementDecision(
            tweet_id=tweet["id"],
            action_type=ActionType.LIKE,
            score=overall_score,
            decision=decision,
            factors=factors,
            confidence=min(1.0, overall_score + 0.1),
        )

    async def _analyze_repost_opportunity(
        self, tweet: Dict[str, Any]
    ) -> EngagementDecision:
        """Analyze tweet for repost opportunity."""
        factors = {}

        # Author relevance (higher weight for reposts)
        author_score = await self._score_author_relevance(tweet["author_id"])
        factors["author_relevance"] = author_score

        # Content relevance (higher weight for reposts)
        content_score = self._score_content_relevance(tweet["text"])
        factors["content_relevance"] = content_score

        # Engagement potential
        engagement_score = self._score_engagement_potential(tweet, action_type="repost")
        factors["engagement_potential"] = engagement_score

        # Timing score
        timing_score = self._score_timing(tweet.get("created_at"))
        factors["timing"] = timing_score

        # Authenticity score (higher weight for reposts)
        authenticity_score = self._score_authenticity(tweet["text"])
        factors["authenticity"] = authenticity_score

        # Adjust weights for repost (more stringent)
        repost_weights = {
            "author_relevance": 0.30,
            "content_relevance": 0.30,
            "engagement_potential": 0.20,
            "timing": 0.10,
            "authenticity": 0.10,
        }

        overall_score = sum(
            score * repost_weights[factor] for factor, score in factors.items()
        )

        decision = overall_score >= self.config.agents.repost_threshold

        return EngagementDecision(
            tweet_id=tweet["id"],
            action_type=ActionType.REPOST,
            score=overall_score,
            decision=decision,
            factors=factors,
            confidence=min(1.0, overall_score + 0.05),
        )

    async def _analyze_comment_opportunity(
        self, tweet: Dict[str, Any]
    ) -> EngagementDecision:
        """Analyze tweet for comment opportunity."""
        factors = {}

        # Author relevance (highest weight for comments)
        author_score = await self._score_author_relevance(tweet["author_id"])
        factors["author_relevance"] = author_score

        # Content relevance (must be high for comments)
        content_score = self._score_content_relevance(tweet["text"])
        factors["content_relevance"] = content_score

        # Engagement potential (questions, community posts, etc.)
        engagement_score = self._score_engagement_potential(
            tweet, action_type="comment"
        )
        factors["engagement_potential"] = engagement_score

        # Timing score (comments work better on fresh content)
        timing_score = self._score_timing(tweet.get("created_at"))
        factors["timing"] = timing_score

        # Authenticity score (critical for comments)
        authenticity_score = self._score_authenticity(tweet["text"])
        factors["authenticity"] = authenticity_score

        # Adjust weights for comments (most stringent)
        comment_weights = {
            "author_relevance": 0.35,
            "content_relevance": 0.25,
            "engagement_potential": 0.25,
            "timing": 0.10,
            "authenticity": 0.05,
        }

        overall_score = sum(
            score * comment_weights[factor] for factor, score in factors.items()
        )

        decision = overall_score >= self.config.agents.comment_threshold

        return EngagementDecision(
            tweet_id=tweet["id"],
            action_type=ActionType.COMMENT,
            score=overall_score,
            decision=decision,
            factors=factors,
            confidence=overall_score,
        )

    async def _score_author_relevance(self, author_id: str) -> float:
        """Score author relevance to our domain."""
        try:
            # Get author info from our database
            user = await self.mongodb.get_user(author_id)

            if user:
                # Use existing engagement score and relevance score
                base_score = (user.engagement_score + user.relevance_score) / 2

                # Bonus for users we follow
                if user.follow_status == "following":
                    base_score += 0.2

                # Bonus for users with recent interactions
                if (
                    user.last_interaction
                    and user.last_interaction > datetime.utcnow() - timedelta(days=7)
                ):
                    base_score += 0.1

                return min(1.0, base_score)

            # If no user data, return neutral score
            return 0.3

        except Exception as e:
            logger.error(f"Failed to score author relevance: {e}")
            return 0.2

    def _score_content_relevance(self, text: str) -> float:
        """Score content relevance to our domain."""
        text_lower = text.lower()
        score = 0.0

        # High value keywords
        for keyword in self.relevance_keywords["high_value"]:
            if keyword in text_lower:
                score += 0.15

        # Medium value keywords
        for keyword in self.relevance_keywords["medium_value"]:
            if keyword in text_lower:
                score += 0.08

        # Penalty for negative indicators
        for keyword in self.relevance_keywords["negative"]:
            if keyword in text_lower:
                score -= 0.3

        # Check for domain-specific patterns
        if re.search(r"(music|artist|show|gig|performance)", text_lower):
            score += 0.1

        if re.search(r"(new orleans|nola|louisiana)", text_lower):
            score += 0.15

        return max(0.0, min(1.0, score))

    def _score_engagement_potential(
        self, tweet: Dict[str, Any], action_type: str
    ) -> float:
        """Score tweet's potential for generating engagement."""
        text = tweet["text"].lower()
        score = 0.0

        # Pattern matching for engagement opportunities
        for pattern_type, pattern in self.engagement_patterns.items():
            if re.search(pattern, text):
                if action_type == "like":
                    score += 0.2
                elif action_type == "repost" and pattern_type in [
                    "local_events",
                    "community_building",
                ]:
                    score += 0.3
                elif action_type == "comment" and pattern_type in [
                    "questions",
                    "calls_for_support",
                ]:
                    score += 0.4

        # Check for patterns to avoid
        for pattern_type, pattern in self.avoid_patterns.items():
            if re.search(pattern, text):
                score -= 0.4

        # Engagement metrics bonus (if available)
        public_metrics = tweet.get("public_metrics", {})
        if public_metrics:
            likes = public_metrics.get("like_count", 0)
            retweets = public_metrics.get("retweet_count", 0)

            # Sweet spot: some engagement but not viral
            if 5 <= likes <= 100:
                score += 0.2
            if 2 <= retweets <= 20:
                score += 0.15

        return max(0.0, min(1.0, score))

    def _score_timing(self, created_at: Optional[datetime]) -> float:
        """Score tweet freshness for engagement timing."""
        if not created_at:
            return 0.5

        try:
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))

            age_hours = (
                datetime.utcnow().replace(tzinfo=created_at.tzinfo) - created_at
            ).total_seconds() / 3600

            # Optimal engagement window
            if age_hours <= 2:
                return 1.0  # Very fresh
            elif age_hours <= 6:
                return 0.8  # Fresh
            elif age_hours <= 24:
                return 0.6  # Recent
            elif age_hours <= 72:
                return 0.3  # Older
            else:
                return 0.1  # Too old

        except Exception as e:
            logger.error(f"Failed to score timing: {e}")
            return 0.5

    def _score_authenticity(self, text: str) -> float:
        """Score content authenticity (non-spammy)."""
        score = 1.0
        text_lower = text.lower()

        # Check for spam indicators
        spam_patterns = [
            r"follow.*back",
            r"dm.*me",
            r"check.*out.*my",
            r"click.*link",
            r"buy.*now",
            r"limited.*time",
        ]

        for pattern in spam_patterns:
            if re.search(pattern, text_lower):
                score -= 0.4

        # Too many hashtags
        hashtag_count = len(re.findall(r"#\w+", text))
        if hashtag_count > 5:
            score -= 0.3
        elif hashtag_count > 3:
            score -= 0.1

        # Too many mentions
        mention_count = len(re.findall(r"@\w+", text))
        if mention_count > 3:
            score -= 0.2

        # All caps (spammy)
        if text.isupper() and len(text) > 20:
            score -= 0.3

        return max(0.0, score)

    async def _execute_likes(self) -> Dict[str, Any]:
        """Execute queued like actions."""
        executed = 0

        while not self.like_queue.empty() and self.safety_limits.can_perform_action(
            "likes"
        ):
            try:
                priority, tweet, decision = await self.like_queue.get()

                success = await self._like_tweet(tweet, decision)
                if success:
                    executed += 1

                # Human-like timing between likes
                await asyncio.sleep(random.uniform(15, 45))

            except Exception as e:
                logger.error(f"Failed to execute like: {e}")
                continue

        return {"executed": executed}

    async def _execute_reposts(self) -> Dict[str, Any]:
        """Execute queued repost actions."""
        executed = 0

        while not self.repost_queue.empty() and self.safety_limits.can_perform_action(
            "reposts"
        ):
            try:
                priority, tweet, decision = await self.repost_queue.get()

                success = await self._repost_tweet(tweet, decision)
                if success:
                    executed += 1

                # Longer delay between reposts
                await asyncio.sleep(random.uniform(60, 180))

            except Exception as e:
                logger.error(f"Failed to execute repost: {e}")
                continue

        return {"executed": executed}

    async def _execute_comments(self) -> Dict[str, Any]:
        """Execute queued comment actions."""
        executed = 0

        while not self.comment_queue.empty() and self.safety_limits.can_perform_action(
            "comments"
        ):
            try:
                priority, tweet, decision = await self.comment_queue.get()

                success = await self._comment_on_tweet(tweet, decision)
                if success:
                    executed += 1

                # Longest delay between comments
                await asyncio.sleep(random.uniform(120, 300))

            except Exception as e:
                logger.error(f"Failed to execute comment: {e}")
                continue

        return {"executed": executed}

    async def _like_tweet(
        self, tweet: Dict[str, Any], decision: EngagementDecision
    ) -> bool:
        """Execute like action on a tweet."""
        try:
            result = await self.safe_twitter_call("like", "bot_like", tweet["id"])

            # Record engagement
            engagement = EngagementHistory(
                target_tweet_id=tweet["id"],
                target_user_id=tweet["author_id"],
                action_type=ActionType.LIKE,
                success=True,
                response_data={"result": str(result) if result else "dry_run"},
                decision_score=decision.score,
                decision_factors=decision.factors,
            )

            await self.mongodb.create_engagement(engagement)

            logger.info(f"Liked tweet {tweet['id'][:10]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to like tweet: {e}")
            return False

    async def _repost_tweet(
        self, tweet: Dict[str, Any], decision: EngagementDecision
    ) -> bool:
        """Execute repost action on a tweet."""
        try:
            # For high-value content, add a comment to the repost
            repost_comment = None
            if decision.score > 0.8:
                # Generate repost comment using centralized prompt system
                repost_prompt = EngagementPrompts.get_repost_comment_prompt(tweet, decision.score)
                # Note: In full implementation, this would use LLM client to generate content
                # For now, we'll skip the complex generation
                repost_comment = None

            if repost_comment:
                result = await self.safe_twitter_call(
                    "create_tweet",
                    "bot_repost",
                    text=repost_comment,
                    quote_tweet_id=tweet["id"],
                )
            else:
                result = await self.safe_twitter_call(
                    "retweet", "bot_repost", tweet["id"]
                )

            # Record engagement
            engagement = EngagementHistory(
                target_tweet_id=tweet["id"],
                target_user_id=tweet["author_id"],
                action_type=ActionType.REPOST,
                success=True,
                response_data={"result": str(result) if result else "dry_run"},
                decision_score=decision.score,
                decision_factors=decision.factors,
                content_used=repost_comment,
            )

            await self.mongodb.create_engagement(engagement)

            logger.info(f"Reposted tweet {tweet['id'][:10]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to repost tweet: {e}")
            return False

    async def _comment_on_tweet(
        self, tweet: Dict[str, Any], decision: EngagementDecision
    ) -> bool:
        """Execute comment action on a tweet."""
        try:
            # Import content agent to generate contextual comment
            # In practice, this should be injected as a dependency
            # This is a simplified approach - in practice, we'd use proper DI
            content_agent = None  # Would be injected

            # Generate contextual comment
            comment_text = await self._generate_contextual_comment(tweet)

            if not comment_text:
                logger.warning("Failed to generate comment text")
                return False

            result = await self.safe_twitter_call(
                "create_tweet",
                "bot_comment",
                text=comment_text,
                in_reply_to_tweet_id=tweet["id"],
            )

            # Record engagement
            engagement = EngagementHistory(
                target_tweet_id=tweet["id"],
                target_user_id=tweet["author_id"],
                action_type=ActionType.COMMENT,
                success=True,
                response_data={"result": str(result) if result else "dry_run"},
                decision_score=decision.score,
                decision_factors=decision.factors,
                content_used=comment_text,
            )

            await self.mongodb.create_engagement(engagement)

            logger.info(f"Commented on tweet {tweet['id'][:10]}...")
            return True

        except Exception as e:
            logger.error(f"Failed to comment on tweet: {e}")
            return False

    async def _generate_contextual_comment(
        self, tweet: Dict[str, Any]
    ) -> Optional[str]:
        """Generate contextual comment for a tweet."""
        try:
            # Get a template comment from cache
            template_comments = await self.mongodb.get_unused_content(
                ContentType.COMMENT.value, limit=10
            )

            if template_comments:
                # Select best template based on context
                selected_comment = random.choice(template_comments)

                # Mark as used
                await self.mongodb.mark_content_used(str(selected_comment.id))

                return selected_comment.content

            # Use centralized fallback responses
            return EngagementPrompts.get_fallback_response()

        except Exception as e:
            logger.error(f"Failed to generate contextual comment: {e}")
            return None

    async def _monitor_engagement_health(self) -> None:
        """Monitor overall engagement health and patterns."""
        try:
            # Check recent engagement success rates
            recent_engagements = await self.mongodb.get_recent_engagements(hours=24)

            if recent_engagements:
                total = len(recent_engagements)
                successful = sum(1 for e in recent_engagements if e.success)
                success_rate = successful / total if total > 0 else 0

                # Record health metric
                await self._record_metric("engagement_health", success_rate)

                if success_rate < 0.8:
                    logger.warning(f"Engagement success rate low: {success_rate:.2%}")

        except Exception as e:
            logger.error(f"Failed to monitor engagement health: {e}")

    async def _should_run(self) -> bool:
        """Check if engagement agent should run."""
        if not await super()._should_run():
            return False

        # Check if we have capacity for any engagement actions
        remaining_likes = self.safety_limits.get_remaining_actions("likes")
        remaining_reposts = self.safety_limits.get_remaining_actions("reposts")
        remaining_comments = self.safety_limits.get_remaining_actions("comments")

        if (
            remaining_likes["hourly"] <= 0
            and remaining_reposts["hourly"] <= 0
            and remaining_comments["hourly"] <= 0
        ):
            logger.debug("All hourly engagement limits reached")
            return False

        return True

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current engagement agent status."""
        base_status = self.get_health_status()

        # Update queue metrics
        total_queue_size = (
            self.like_queue.qsize()
            + self.repost_queue.qsize()
            + self.comment_queue.qsize()
        )
        agent_queue_size.labels(agent=self.name).set(total_queue_size)

        engagement_status = {
            **base_status,
            "queue_sizes": {
                "likes": self.like_queue.qsize(),
                "reposts": self.repost_queue.qsize(),
                "comments": self.comment_queue.qsize(),
            },
            "safety_limits": {
                "likes": self.safety_limits.get_remaining_actions("likes"),
                "reposts": self.safety_limits.get_remaining_actions("reposts"),
                "comments": self.safety_limits.get_remaining_actions("comments"),
            },
            "thresholds": {
                "like": self.config.agents.like_threshold,
                "repost": self.config.agents.repost_threshold,
                "comment": self.config.agents.comment_threshold,
            },
        }

        return engagement_status
