"""Follow Agent for user discovery and relationship management."""

import asyncio
import random
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta

from loguru import logger

from src.agents.base_agent import BaseAgent
from src.models.data_models import (
    User, FollowStatus, EngagementHistory, ActionType,
    FollowDecision, UserAnalysis
)
from src.utils.rate_limiter import agent_queue_size


class FollowAgent(BaseAgent):
    """Agent responsible for user discovery and follow/unfollow decisions."""
    
    def __init__(self, *args, **kwargs):
        super().__init__("FollowAgent", *args, **kwargs)
        self.discovery_queue = asyncio.Queue()
        self.follow_queue = asyncio.Queue()
        self.unfollow_queue = asyncio.Queue()
        
        # Discovery patterns for New Orleans/Music domain
        self.search_patterns = [
            "New Orleans music",
            "NOLA jazz",
            "Louisiana bounce",
            "French Quarter",
            "Mardi Gras",
            "New Orleans culture",
            "NOLA food",
            "Bywater",
            "Treme",
            "jazz musician",
            "blues New Orleans",
            "krewe"
        ]
        
        # Keywords that indicate relevance
        self.relevance_keywords = {
            "location": ["new orleans", "nola", "louisiana", "la", "french quarter", 
                        "bywater", "treme", "uptown", "downtown", "marigny"],
            "music": ["jazz", "blues", "bounce", "brass band", "music", "musician", 
                     "singer", "dj", "producer", "concert", "festival"],
            "culture": ["krewe", "mardi gras", "carnival", "second line", "culture", 
                       "art", "food", "cuisine", "creole", "cajun"],
            "genz": ["gen z", "genz", "young", "college", "student", "tiktok", 
                    "instagram", "viral", "trend", "meme"]
        }
    
    async def execute(self) -> Dict[str, Any]:
        """Execute follow agent workflow."""
        results = {
            "discovered_users": 0,
            "follow_decisions": 0,
            "follows_executed": 0,
            "unfollows_executed": 0,
            "errors": []
        }
        
        try:
            # 1. Discover new users
            discovery_results = await self._discover_users()
            results["discovered_users"] = discovery_results["count"]
            
            # 2. Analyze existing users for follow decisions
            analysis_results = await self._analyze_users_for_follows()
            results["follow_decisions"] = analysis_results["decisions_made"]
            
            # 3. Execute queued follows
            follow_results = await self._execute_follows()
            results["follows_executed"] = follow_results["executed"]
            
            # 4. Manage unfollows (inactive/irrelevant users)
            unfollow_results = await self._manage_unfollows()
            results["unfollows_executed"] = unfollow_results["executed"]
            
            # 5. Update user relationship data
            await self._update_user_relationships()
            
        except Exception as e:
            logger.error(f"Follow agent execution failed: {e}")
            results["errors"].append(str(e))
            raise
        
        return results
    
    async def _discover_users(self) -> Dict[str, Any]:
        """Discover new users in the target domain."""
        discovered_count = 0
        
        try:
            # Select random search patterns to avoid repetition
            patterns = random.sample(
                self.search_patterns, 
                min(3, len(self.search_patterns))
            )
            
            for pattern in patterns:
                if not self.safety_limits.can_perform_action("follows"):
                    logger.warning("Daily follow limit reached, stopping discovery")
                    break
                
                try:
                    # Search for users
                    users = await self.search_users(pattern, max_results=20)
                    
                    for user_data in users:
                        # Check if user already exists
                        existing_user = await self.mongodb.get_user(user_data["id"])
                        
                        if not existing_user:
                            # Create new user record
                            user = User(
                                twitter_user_id=user_data["id"],
                                username=user_data["username"],
                                display_name=user_data.get("name", ""),
                                bio=user_data.get("description", ""),
                                location=user_data.get("location", ""),
                                follower_count=user_data.get("public_metrics", {}).get("followers_count", 0),
                                following_count=user_data.get("public_metrics", {}).get("following_count", 0),
                                tweet_count=user_data.get("public_metrics", {}).get("tweet_count", 0),
                                last_activity=user_data.get("created_at"),
                                tags=[pattern.lower().replace(" ", "_")]
                            )
                            
                            # Calculate initial engagement score
                            user.engagement_score = await self._calculate_engagement_score(user)
                            user.relevance_score = await self._calculate_relevance_score(user)
                            
                            await self.mongodb.create_user(user)
                            discovered_count += 1
                            
                            logger.debug(f"Discovered user @{user.username} (score: {user.engagement_score:.2f})")
                        
                        else:
                            # Update existing user data
                            update_data = {
                                "follower_count": user_data.get("public_metrics", {}).get("followers_count", 0),
                                "following_count": user_data.get("public_metrics", {}).get("following_count", 0),
                                "tweet_count": user_data.get("public_metrics", {}).get("tweet_count", 0),
                            }
                            await self.mongodb.update_user(user_data["id"], update_data)
                    
                    # Add delay between searches
                    await asyncio.sleep(random.uniform(2, 5))
                    
                except Exception as e:
                    logger.error(f"Error discovering users for pattern '{pattern}': {e}")
                    continue
            
        except Exception as e:
            logger.error(f"User discovery failed: {e}")
            raise
        
        logger.info(f"Discovered {discovered_count} new users")
        return {"count": discovered_count}
    
    async def _analyze_users_for_follows(self) -> Dict[str, Any]:
        """Analyze users and make follow decisions."""
        decisions_made = 0
        
        try:
            # Get users that we're not following but have good engagement scores
            candidate_users = await self.mongodb.get_users_for_discovery(limit=50)
            
            for user in candidate_users:
                if not self.safety_limits.can_perform_action("follows"):
                    logger.warning("Daily follow limit reached")
                    break
                
                # Analyze user for follow potential
                analysis = await self._analyze_user(user)
                decision = FollowDecision(
                    user_id=user.twitter_user_id,
                    score=analysis.overall_score,
                    decision=analysis.overall_score >= self.config.agents.follow_threshold,
                    factors=analysis.factors,
                    confidence=min(1.0, analysis.overall_score + 0.1)
                )
                
                if decision.decision:
                    await self.follow_queue.put({
                        "user": user,
                        "decision": decision,
                        "priority": int(analysis.overall_score * 10)
                    })
                    
                    logger.debug(f"Queued @{user.username} for follow (score: {analysis.overall_score:.2f})")
                
                decisions_made += 1
                
                # Record decision metrics
                await self._record_metric(
                    "follow_decision",
                    analysis.overall_score,
                    {
                        "user_id": user.twitter_user_id,
                        "decision": decision.decision,
                        "factors": analysis.factors
                    }
                )
        
        except Exception as e:
            logger.error(f"User analysis failed: {e}")
            raise
        
        return {"decisions_made": decisions_made}
    
    async def _execute_follows(self) -> Dict[str, Any]:
        """Execute queued follow actions."""
        executed = 0
        
        while not self.follow_queue.empty() and self.safety_limits.can_perform_action("follows"):
            try:
                follow_action = await self.follow_queue.get()
                user = follow_action["user"]
                decision = follow_action["decision"]
                
                # Execute follow
                success = await self._follow_user(user, decision)
                
                if success:
                    executed += 1
                    # Update user status
                    await self.mongodb.update_user(
                        user.twitter_user_id,
                        {"follow_status": FollowStatus.FOLLOWING.value}
                    )
                
                # Add human-like delay
                await asyncio.sleep(random.uniform(30, 120))
                
            except Exception as e:
                logger.error(f"Failed to execute follow: {e}")
                continue
        
        return {"executed": executed}
    
    async def _manage_unfollows(self) -> Dict[str, Any]:
        """Manage unfollowing inactive or irrelevant users."""
        executed = 0
        
        try:
            # Get users we're following
            following_users = await self.mongodb.get_users_by_follow_status(
                FollowStatus.FOLLOWING.value, 
                limit=100
            )
            
            for user in following_users:
                # Check if user should be unfollowed
                should_unfollow = await self._should_unfollow_user(user)
                
                if should_unfollow:
                    success = await self._unfollow_user(user)
                    if success:
                        executed += 1
                        await self.mongodb.update_user(
                            user.twitter_user_id,
                            {"follow_status": FollowStatus.NOT_FOLLOWING.value}
                        )
                    
                    # Add delay between unfollows
                    await asyncio.sleep(random.uniform(10, 30))
        
        except Exception as e:
            logger.error(f"Unfollow management failed: {e}")
            raise
        
        return {"executed": executed}
    
    async def _analyze_user(self, user: User) -> UserAnalysis:
        """Perform comprehensive user analysis."""
        factors = {}
        
        # Location scoring
        location_score = self._score_location(user.location or "", user.bio or "")
        factors["location"] = location_score
        
        # Activity scoring
        activity_score = self._score_activity(user)
        factors["activity"] = activity_score
        
        # Engagement scoring
        engagement_score = self._score_engagement(user)
        factors["engagement"] = engagement_score
        
        # Relevance scoring
        relevance_score = self._score_relevance(user.bio or "", user.location or "")
        factors["relevance"] = relevance_score
        
        # Calculate overall score with weights
        overall_score = (
            location_score * self.config.agents.location_weight +
            activity_score * self.config.agents.activity_weight +
            engagement_score * self.config.agents.engagement_weight +
            relevance_score * self.config.agents.relevance_weight
        )
        
        # Generate recommendation
        if overall_score >= 0.8:
            recommendation = "highly_recommended"
        elif overall_score >= 0.6:
            recommendation = "recommended"
        elif overall_score >= 0.4:
            recommendation = "consider"
        else:
            recommendation = "skip"
        
        return UserAnalysis(
            user_id=user.twitter_user_id,
            location_score=location_score,
            activity_score=activity_score,
            engagement_score=engagement_score,
            relevance_score=relevance_score,
            overall_score=overall_score,
            recommendation=recommendation,
            factors=factors
        )
    
    def _score_location(self, location: str, bio: str) -> float:
        """Score user based on location relevance."""
        text = f"{location} {bio}".lower()
        
        score = 0.0
        for keyword in self.relevance_keywords["location"]:
            if keyword in text:
                score += 0.2
        
        return min(1.0, score)
    
    def _score_activity(self, user: User) -> float:
        """Score user based on activity level."""
        # Recent activity
        activity_score = 0.0
        
        if user.last_activity:
            days_since_activity = (datetime.utcnow() - user.last_activity).days
            if days_since_activity <= 1:
                activity_score += 0.4
            elif days_since_activity <= 7:
                activity_score += 0.3
            elif days_since_activity <= 30:
                activity_score += 0.2
        
        # Tweet frequency (estimated)
        if user.tweet_count > 1000:
            activity_score += 0.3
        elif user.tweet_count > 100:
            activity_score += 0.2
        elif user.tweet_count > 10:
            activity_score += 0.1
        
        # Follower/following ratio
        if user.follower_count > 0 and user.following_count > 0:
            ratio = user.follower_count / user.following_count
            if 0.1 <= ratio <= 10:  # Healthy ratio
                activity_score += 0.3
            elif ratio > 10:  # Popular account
                activity_score += 0.2
        
        return min(1.0, activity_score)
    
    def _score_engagement(self, user: User) -> float:
        """Score user based on engagement potential."""
        score = 0.0
        
        # Follower count (but not too high to avoid spam)
        if 50 <= user.follower_count <= 10000:
            score += 0.4
        elif 10 <= user.follower_count < 50:
            score += 0.3
        elif user.follower_count > 10000:
            score += 0.2
        
        # Following count (active users follow others)
        if 20 <= user.following_count <= 2000:
            score += 0.3
        elif user.following_count > 2000:
            score += 0.1
        
        # Use existing engagement score if available
        if hasattr(user, 'engagement_score') and user.engagement_score:
            score += user.engagement_score * 0.3
        
        return min(1.0, score)
    
    def _score_relevance(self, bio: str, location: str) -> float:
        """Score user based on content relevance."""
        text = f"{bio} {location}".lower()
        score = 0.0
        
        # Music-related keywords
        for keyword in self.relevance_keywords["music"]:
            if keyword in text:
                score += 0.15
        
        # Culture keywords
        for keyword in self.relevance_keywords["culture"]:
            if keyword in text:
                score += 0.1
        
        # GenZ keywords
        for keyword in self.relevance_keywords["genz"]:
            if keyword in text:
                score += 0.05
        
        return min(1.0, score)
    
    async def _calculate_engagement_score(self, user: User) -> float:
        """Calculate initial engagement score for a user."""
        return self._score_engagement(user)
    
    async def _calculate_relevance_score(self, user: User) -> float:
        """Calculate relevance score for a user."""
        return self._score_relevance(user.bio or "", user.location or "")
    
    async def _follow_user(self, user: User, decision: FollowDecision) -> bool:
        """Execute follow action for a user."""
        try:
            # Make follow API call
            result = await self.safe_twitter_call(
                "follow_user",
                "bot_follow",
                user.twitter_user_id
            )
            
            # Record engagement history
            engagement = EngagementHistory(
                target_tweet_id="",  # Not applicable for follows
                target_user_id=user.twitter_user_id,
                action_type=ActionType.FOLLOW,
                success=True,
                response_data={"result": str(result) if result else "dry_run"},
                decision_score=decision.score,
                decision_factors=decision.factors
            )
            
            await self.mongodb.create_engagement(engagement)
            
            logger.info(f"Successfully followed @{user.username}")
            return True
            
        except Exception as e:
            # Record failed engagement
            engagement = EngagementHistory(
                target_tweet_id="",
                target_user_id=user.twitter_user_id,
                action_type=ActionType.FOLLOW,
                success=False,
                response_data={},
                error_message=str(e),
                decision_score=decision.score
            )
            
            await self.mongodb.create_engagement(engagement)
            
            logger.error(f"Failed to follow @{user.username}: {e}")
            return False
    
    async def _should_unfollow_user(self, user: User) -> bool:
        """Determine if a user should be unfollowed."""
        # Unfollow if inactive for too long
        if user.last_activity:
            days_inactive = (datetime.utcnow() - user.last_activity).days
            if days_inactive > 30:
                return True
        
        # Unfollow if engagement score is too low
        if user.engagement_score < 0.3:
            return True
        
        # Unfollow if no mutual engagement over time
        if user.interaction_count == 0 and user.discovered_date:
            days_since_follow = (datetime.utcnow() - user.discovered_date).days
            if days_since_follow > 7:
                return True
        
        return False
    
    async def _unfollow_user(self, user: User) -> bool:
        """Execute unfollow action for a user."""
        try:
            result = await self.safe_twitter_call(
                "unfollow_user",
                "bot_follow",
                user.twitter_user_id
            )
            
            # Record engagement history
            engagement = EngagementHistory(
                target_tweet_id="",
                target_user_id=user.twitter_user_id,
                action_type=ActionType.UNFOLLOW,
                success=True,
                response_data={"result": str(result) if result else "dry_run"}
            )
            
            await self.mongodb.create_engagement(engagement)
            
            logger.info(f"Successfully unfollowed @{user.username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unfollow @{user.username}: {e}")
            return False
    
    async def _update_user_relationships(self) -> None:
        """Update user relationship data based on interactions."""
        try:
            # Get recent engagements
            recent_engagements = await self.mongodb.get_recent_engagements(hours=24)
            
            # Update interaction counts
            for engagement in recent_engagements:
                if engagement.success and engagement.target_user_id:
                    await self.mongodb.update_user(
                        engagement.target_user_id,
                        {
                            "last_interaction": engagement.timestamp,
                            "$inc": {"interaction_count": 1}
                        }
                    )
        
        except Exception as e:
            logger.error(f"Failed to update user relationships: {e}")
    
    async def _should_run(self) -> bool:
        """Check if follow agent should run."""
        # Check basic conditions from parent
        if not await super()._should_run():
            return False
        
        # Check if we have follow capacity
        remaining = self.safety_limits.get_remaining_actions("follows")
        if remaining["daily"] <= 0:
            logger.debug("Daily follow limit reached")
            return False
        
        return True
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current follow agent status."""
        base_status = self.get_health_status()
        
        # Update queue sizes for metrics
        agent_queue_size.labels(agent=self.name).set(
            self.discovery_queue.qsize() + 
            self.follow_queue.qsize() + 
            self.unfollow_queue.qsize()
        )
        
        follow_status = {
            **base_status,
            "queue_sizes": {
                "discovery": self.discovery_queue.qsize(),
                "follow": self.follow_queue.qsize(),
                "unfollow": self.unfollow_queue.qsize()
            },
            "safety_limits": {
                "follows": self.safety_limits.get_remaining_actions("follows")
            },
            "search_patterns": len(self.search_patterns)
        }
        
        return follow_status