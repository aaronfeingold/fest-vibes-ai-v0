"""Rate limiting utilities for Twitter API and bot safety."""

import asyncio
import time
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from loguru import logger


@dataclass
class RateLimit:
    """Rate limit configuration."""
    max_calls: int
    window_seconds: int
    name: str


class RateLimiter:
    """Advanced rate limiter with multiple strategies."""
    
    def __init__(self):
        self._calls: Dict[str, deque] = defaultdict(deque)
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._backoff_until: Dict[str, float] = {}
        
        # Twitter API rate limits
        self.limits = {
            "tweet_create": RateLimit(300, 900, "Tweet Creation"),  # 15 min window
            "tweet_like": RateLimit(300, 900, "Tweet Likes"),
            "tweet_retweet": RateLimit(300, 900, "Tweet Retweets"),
            "follow_create": RateLimit(400, 86400, "Follow Users"),  # 24 hour window
            "user_lookup": RateLimit(300, 900, "User Lookup"),
            "search_tweets": RateLimit(180, 900, "Search Tweets"),
            
            # Bot-specific conservative limits
            "bot_follow": RateLimit(50, 86400, "Bot Follows"),
            "bot_post": RateLimit(10, 86400, "Bot Posts"),
            "bot_like": RateLimit(30, 3600, "Bot Likes"),  # per hour
            "bot_repost": RateLimit(15, 3600, "Bot Reposts"),
            "bot_comment": RateLimit(10, 3600, "Bot Comments"),
        }
    
    async def check_rate_limit(self, key: str) -> bool:
        """Check if we're within rate limits for a given key."""
        if key not in self.limits:
            logger.warning(f"Unknown rate limit key: {key}")
            return True
        
        async with self._locks[key]:
            now = time.time()
            limit = self.limits[key]
            
            # Check if we're in backoff
            if key in self._backoff_until and now < self._backoff_until[key]:
                remaining = self._backoff_until[key] - now
                logger.warning(f"Rate limit {key} in backoff for {remaining:.1f}s more")
                return False
            
            # Clean old calls
            calls = self._calls[key]
            cutoff = now - limit.window_seconds
            while calls and calls[0] < cutoff:
                calls.popleft()
            
            # Check if we can make another call
            if len(calls) >= limit.max_calls:
                oldest_call = calls[0]
                wait_time = oldest_call + limit.window_seconds - now
                logger.warning(
                    f"Rate limit {limit.name} exceeded: {len(calls)}/{limit.max_calls} "
                    f"calls in {limit.window_seconds}s window. Wait {wait_time:.1f}s"
                )
                return False
            
            return True
    
    async def acquire(self, key: str) -> bool:
        """Acquire rate limit permission (blocks until available)."""
        if key not in self.limits:
            return True
        
        while not await self.check_rate_limit(key):
            # Calculate wait time
            limit = self.limits[key]
            calls = self._calls[key]
            
            if calls:
                oldest_call = calls[0]
                wait_time = max(1, oldest_call + limit.window_seconds - time.time())
            else:
                wait_time = 1
            
            logger.info(f"Rate limit {limit.name}: waiting {wait_time:.1f}s")
            await asyncio.sleep(min(wait_time, 60))  # Cap wait at 1 minute
        
        # Record the call
        async with self._locks[key]:
            self._calls[key].append(time.time())
        
        return True
    
    async def try_acquire(self, key: str) -> bool:
        """Try to acquire rate limit permission (non-blocking)."""
        if await self.check_rate_limit(key):
            async with self._locks[key]:
                self._calls[key].append(time.time())
            return True
        return False
    
    def apply_backoff(self, key: str, seconds: float) -> None:
        """Apply exponential backoff to a rate limit key."""
        until = time.time() + seconds
        self._backoff_until[key] = until
        logger.warning(f"Applied {seconds}s backoff to {key} until {datetime.fromtimestamp(until)}")
    
    def get_remaining_calls(self, key: str) -> int:
        """Get remaining calls in current window."""
        if key not in self.limits:
            return float('inf')
        
        limit = self.limits[key]
        calls = self._calls[key]
        
        # Clean old calls
        now = time.time()
        cutoff = now - limit.window_seconds
        while calls and calls[0] < cutoff:
            calls.popleft()
        
        return max(0, limit.max_calls - len(calls))
    
    def get_reset_time(self, key: str) -> Optional[datetime]:
        """Get when rate limit resets."""
        if key not in self.limits or not self._calls[key]:
            return None
        
        limit = self.limits[key]
        oldest_call = self._calls[key][0]
        reset_time = oldest_call + limit.window_seconds
        
        return datetime.fromtimestamp(reset_time)
    
    def get_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all rate limits."""
        status = {}
        
        for key, limit in self.limits.items():
            remaining = self.get_remaining_calls(key)
            reset_time = self.get_reset_time(key)
            in_backoff = (
                key in self._backoff_until and 
                time.time() < self._backoff_until[key]
            )
            
            status[key] = {
                "name": limit.name,
                "max_calls": limit.max_calls,
                "window_seconds": limit.window_seconds,
                "remaining": remaining,
                "reset_time": reset_time.isoformat() if reset_time else None,
                "in_backoff": in_backoff,
                "backoff_until": (
                    datetime.fromtimestamp(self._backoff_until[key]).isoformat()
                    if in_backoff else None
                )
            }
        
        return status


class ExponentialBackoff:
    """Exponential backoff for error handling."""
    
    def __init__(self, initial_delay: float = 1.0, max_delay: float = 300.0, multiplier: float = 2.0):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.current_delay = initial_delay
        self.attempt = 0
    
    async def wait(self) -> None:
        """Wait for the current delay period."""
        if self.attempt > 0:
            delay = min(self.current_delay, self.max_delay)
            logger.info(f"Backoff delay: {delay:.1f}s (attempt {self.attempt})")
            await asyncio.sleep(delay)
            self.current_delay *= self.multiplier
        
        self.attempt += 1
    
    def reset(self) -> None:
        """Reset backoff to initial state."""
        self.current_delay = self.initial_delay
        self.attempt = 0
    
    @property
    def should_retry(self) -> bool:
        """Check if we should continue retrying."""
        return self.current_delay <= self.max_delay


class ActionQueue:
    """Queue for managing bot actions with timing control."""
    
    def __init__(self, min_interval: int = 30, max_interval: int = 300):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.last_action_time = 0
        self.running = False
    
    async def add_action(self, action_type: str, action_data: Dict[str, Any], priority: int = 0) -> None:
        """Add action to queue."""
        action = {
            "type": action_type,
            "data": action_data,
            "priority": priority,
            "created_at": time.time()
        }
        await self.queue.put(action)
    
    async def get_next_action(self) -> Optional[Dict[str, Any]]:
        """Get next action from queue with timing control."""
        if self.queue.empty():
            return None
        
        action = await self.queue.get()
        
        # Ensure minimum interval between actions
        now = time.time()
        time_since_last = now - self.last_action_time
        
        if time_since_last < self.min_interval:
            wait_time = self.min_interval - time_since_last
            # Add some randomness to avoid detection
            import random
            wait_time += random.uniform(0, self.min_interval * 0.5)
            
            logger.debug(f"Waiting {wait_time:.1f}s before next action")
            await asyncio.sleep(wait_time)
        
        self.last_action_time = time.time()
        return action
    
    def queue_size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()


class SafetyLimits:
    """Additional safety limits to prevent spam detection."""
    
    def __init__(self):
        self.daily_actions: Dict[str, int] = defaultdict(int)
        self.hourly_actions: Dict[str, int] = defaultdict(int)
        self.last_reset_day = datetime.utcnow().date()
        self.last_reset_hour = datetime.utcnow().hour
        
        # Safety thresholds
        self.daily_limits = {
            "follows": 50,
            "posts": 10,
            "likes": 200,
            "reposts": 50,
            "comments": 30
        }
        
        self.hourly_limits = {
            "likes": 30,
            "reposts": 15,
            "comments": 10,
            "follows": 10
        }
    
    def _check_reset(self) -> None:
        """Check if counters need to be reset."""
        now = datetime.utcnow()
        
        # Reset daily counters
        if now.date() != self.last_reset_day:
            self.daily_actions.clear()
            self.last_reset_day = now.date()
            logger.info("Reset daily action counters")
        
        # Reset hourly counters
        if now.hour != self.last_reset_hour:
            self.hourly_actions.clear()
            self.last_reset_hour = now.hour
            logger.info("Reset hourly action counters")
    
    def can_perform_action(self, action_type: str) -> bool:
        """Check if action is within safety limits."""
        self._check_reset()
        
        # Check daily limits
        if action_type in self.daily_limits:
            if self.daily_actions[action_type] >= self.daily_limits[action_type]:
                logger.warning(
                    f"Daily limit reached for {action_type}: "
                    f"{self.daily_actions[action_type]}/{self.daily_limits[action_type]}"
                )
                return False
        
        # Check hourly limits
        if action_type in self.hourly_limits:
            if self.hourly_actions[action_type] >= self.hourly_limits[action_type]:
                logger.warning(
                    f"Hourly limit reached for {action_type}: "
                    f"{self.hourly_actions[action_type]}/{self.hourly_limits[action_type]}"
                )
                return False
        
        return True
    
    def record_action(self, action_type: str) -> None:
        """Record that an action was performed."""
        self._check_reset()
        self.daily_actions[action_type] += 1
        self.hourly_actions[action_type] += 1
        
        logger.debug(
            f"Recorded {action_type}: "
            f"daily={self.daily_actions[action_type]}/{self.daily_limits.get(action_type, '∞')}, "
            f"hourly={self.hourly_actions[action_type]}/{self.hourly_limits.get(action_type, '∞')}"
        )
    
    def get_remaining_actions(self, action_type: str) -> Dict[str, int]:
        """Get remaining actions for the day/hour."""
        self._check_reset()
        
        daily_remaining = (
            self.daily_limits.get(action_type, float('inf')) - 
            self.daily_actions[action_type]
        )
        
        hourly_remaining = (
            self.hourly_limits.get(action_type, float('inf')) - 
            self.hourly_actions[action_type]
        )
        
        return {
            "daily": max(0, daily_remaining),
            "hourly": max(0, hourly_remaining)
        }