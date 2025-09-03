"""Base agent class with common functionality."""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

import tweepy
from loguru import logger
from prometheus_client import Counter, Histogram, Gauge

from src.config.settings import BotConfig
from src.database.mongodb_manager import MongoDBManager
from src.database.postgres_manager import PostgreSQLManager
from src.utils.rate_limiter import RateLimiter, ExponentialBackoff, SafetyLimits
from src.models.data_models import BotMetrics


# Prometheus metrics
agent_actions_total = Counter('agent_actions_total', 'Total agent actions', ['agent', 'action_type', 'status'])
agent_action_duration = Histogram('agent_action_duration_seconds', 'Agent action duration', ['agent', 'action_type'])
agent_errors_total = Counter('agent_errors_total', 'Total agent errors', ['agent', 'error_type'])
agent_queue_size = Gauge('agent_queue_size', 'Current agent queue size', ['agent'])


class BaseAgent(ABC):
    """Abstract base class for all bot agents."""

    def __init__(
        self,
        name: str,
        config: BotConfig,
        mongodb: MongoDBManager,
        postgres: PostgreSQLManager,
        rate_limiter: RateLimiter
    ):
        self.name = name
        self.config = config
        self.mongodb = mongodb
        self.postgres = postgres
        self.rate_limiter = rate_limiter
        self.safety_limits = SafetyLimits()

        # Twitter API client
        self.twitter_client = self._create_twitter_client()

        # State tracking
        self.running = False
        self.last_run = None
        self.error_count = 0
        self.success_count = 0

        # Backoff handling
        self.backoff = ExponentialBackoff(
            initial_delay=self.config.rate_limit.initial_backoff,
            max_delay=self.config.rate_limit.max_backoff,
            multiplier=self.config.rate_limit.backoff_multiplier
        )

    def _create_twitter_client(self) -> tweepy.Client:
        """Create Twitter API client."""
        try:
            client = tweepy.Client(
                bearer_token=self.config.twitter.bearer_token,
                consumer_key=self.config.twitter.api_key,
                consumer_secret=self.config.twitter.api_secret,
                access_token=self.config.twitter.access_token,
                access_token_secret=self.config.twitter.access_token_secret,
                wait_on_rate_limit=False  # We handle rate limiting ourselves
            )

            # Verify credentials
            if not self.config.development.dry_run:
                me = client.get_me()
                logger.info(f"Twitter client authenticated as @{me.data.username}")

            return client

        except Exception as e:
            logger.error(f"Failed to create Twitter client: {e}")
            raise

    @abstractmethod
    async def execute(self) -> Dict[str, Any]:
        """Execute agent's main logic."""
        pass

    @abstractmethod
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        pass

    async def run_once(self) -> Dict[str, Any]:
        """Run agent once with error handling and metrics."""
        start_time = time.time()
        result = {"success": False, "error": None, "data": {}}

        try:
            logger.info(f"Starting {self.name} agent execution")

            # Check if we should run
            if not await self._should_run():
                result = {
                    "success": True,
                    "skipped": True,
                    "reason": "Rate limited or safety constraints"
                }
                return result

            # Execute agent logic
            with agent_action_duration.labels(agent=self.name, action_type='execute').time():
                execution_result = await self.execute()

            result = {
                "success": True,
                "data": execution_result,
                "execution_time": time.time() - start_time
            }

            self.success_count += 1
            self.last_run = datetime.utcnow()
            self.backoff.reset()

            # Record success metric
            await self._record_metric("execution_success", 1.0, {"execution_time": result["execution_time"]})

            agent_actions_total.labels(
                agent=self.name, action_type="execute", status="success"
            ).inc()

            logger.info(f"{self.name} agent completed successfully")

        except Exception as e:
            self.error_count += 1
            result["error"] = str(e)

            logger.error(f"{self.name} agent failed: {e}")

            # Handle different types of errors
            await self._handle_error(e)

            # Record error metric
            await self._record_metric("execution_error", 1.0, {"error_type": type(e).__name__})

            agent_actions_total.labels(
                agent=self.name, action_type="execute", status="error"
            ).inc()

            agent_errors_total.labels(
                agent=self.name, error_type=type(e).__name__
            ).inc()

            # Apply backoff on error
            await self.backoff.wait()

        return result

    async def _should_run(self) -> bool:
        """Check if agent should run based on various constraints."""
        # Check if agent is in backoff
        if not self.backoff.should_retry and self.error_count > 0:
            logger.debug(f"{self.name} agent in backoff")
            return False

        # Check safety limits for agent-specific actions
        # This is overridden by specific agents
        return True

    async def _handle_error(self, error: Exception) -> None:
        """Handle different types of errors appropriately."""
        error_type = type(error).__name__

        if isinstance(error, tweepy.TooManyRequests):
            logger.warning(f"Twitter rate limit hit for {self.name}")
            # Apply backoff to relevant rate limit keys
            self.rate_limiter.apply_backoff("twitter_api", 900)  # 15 minutes

        elif isinstance(error, tweepy.Forbidden):
            logger.error(f"Twitter API forbidden error for {self.name}: {error}")
            # Longer backoff for permission errors
            self.rate_limiter.apply_backoff("twitter_api", 3600)  # 1 hour

        elif isinstance(error, tweepy.BadRequest):
            logger.error(f"Twitter API bad request for {self.name}: {error}")
            # No backoff for bad requests, but log for review

        elif isinstance(error, Exception):
            logger.error(f"Unexpected error in {self.name}: {error}")

    async def _record_metric(
        self, metric_type: str, value: float, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a metric to the database."""
        try:
            metric = BotMetrics(
                metric_type=metric_type,
                agent=self.name,
                value=value,
                timestamp=datetime.utcnow(),
                metadata=metadata or {}
            )

            await self.mongodb.create_metric(metric)

        except Exception as e:
            logger.error(f"Failed to record metric {metric_type}: {e}")

    async def safe_twitter_call(
        self,
        api_method: str,
        rate_limit_key: str,
        *args,
        **kwargs
    ) -> Any:
        """Make a safe Twitter API call with rate limiting and error handling."""
        # Check rate limits
        if not await self.rate_limiter.acquire(rate_limit_key):
            raise Exception(f"Rate limit exceeded for {rate_limit_key}")

        # Check safety limits
        action_type = api_method.split('_')[-1]  # Extract action from method name
        if not self.safety_limits.can_perform_action(action_type):
            raise Exception(f"Safety limit exceeded for {action_type}")

        try:
            # Get the actual method from the client
            method = getattr(self.twitter_client, api_method)

            # Make the call
            if self.config.development.dry_run:
                logger.info(f"DRY RUN: Would call {api_method} with args={args}, kwargs={kwargs}")
                return {"dry_run": True, "method": api_method}
            else:
                result = method(*args, **kwargs)

                # Record successful action
                self.safety_limits.record_action(action_type)

                # Log API call if enabled
                if self.config.development.log_api_requests:
                    logger.debug(f"API call {api_method} successful")

                return result

        except tweepy.TooManyRequests as e:
            logger.warning(f"Rate limit hit for {api_method}")
            self.rate_limiter.apply_backoff(rate_limit_key, 900)
            raise

        except Exception as e:
            logger.error(f"Twitter API error in {api_method}: {e}")
            raise

    async def get_timeline_tweets(self, user_id: Optional[str] = None, count: int = 20) -> List[Dict[str, Any]]:
        """Get timeline tweets safely."""
        try:
            if user_id:
                response = await self.safe_twitter_call(
                    "get_users_tweets",
                    "user_lookup",
                    user_id,
                    max_results=count,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations']
                )
            else:
                response = await self.safe_twitter_call(
                    "get_home_timeline",
                    "search_tweets",
                    max_results=count,
                    tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations']
                )

            if response and hasattr(response, 'data'):
                return [self._tweet_to_dict(tweet) for tweet in response.data]
            return []

        except Exception as e:
            logger.error(f"Failed to get timeline tweets: {e}")
            return []

    def _tweet_to_dict(self, tweet) -> Dict[str, Any]:
        """Convert tweet object to dictionary."""
        return {
            "id": tweet.id,
            "text": tweet.text,
            "author_id": tweet.author_id,
            "created_at": tweet.created_at,
            "public_metrics": tweet.public_metrics if hasattr(tweet, 'public_metrics') else {},
            "context_annotations": tweet.context_annotations if hasattr(tweet, 'context_annotations') else []
        }

    async def search_users(
        self,
        query: str,
        max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """Search for users by finding their tweets and extracting user info."""
        try:
            # Search for recent tweets matching the query
            response = await self.safe_twitter_call(
                "search_recent_tweets",
                "search_tweets",
                query=query,
                max_results=min(max_results, 100),  # Twitter API limit
                tweet_fields=['author_id', 'created_at', 'public_metrics'],
                expansions=['author_id'],
                user_fields=['created_at', 'description', 'location', 'public_metrics', 'verified']
            )

            if response and hasattr(response, 'includes') and 'users' in response.includes:
                # Extract unique users from the tweet results
                users = response.includes['users']
                return [self._user_to_dict(user) for user in users]
            return []

        except Exception as e:
            logger.error(f"Failed to search users: {e}")
            return []

    def _user_to_dict(self, user) -> Dict[str, Any]:
        """Convert user object to dictionary."""
        return {
            "id": user.id,
            "username": user.username,
            "name": user.name,
            "description": user.description if hasattr(user, 'description') else "",
            "location": user.location if hasattr(user, 'location') else "",
            "created_at": user.created_at if hasattr(user, 'created_at') else None,
            "public_metrics": user.public_metrics if hasattr(user, 'public_metrics') else {},
            "verified": user.verified if hasattr(user, 'verified') else False
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status."""
        return {
            "name": self.name,
            "running": self.running,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.success_count + self.error_count),
            "backoff_active": not self.backoff.should_retry,
            "rate_limit_status": self.rate_limiter.get_status()
        }

    async def cleanup(self) -> None:
        """Clean up agent resources."""
        self.running = False
        logger.info(f"{self.name} agent cleaned up")
