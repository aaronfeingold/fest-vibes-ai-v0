"""Tests for utility modules."""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.utils.rate_limiter import RateLimiter, ExponentialBackoff, ActionQueue, SafetyLimits
from src.utils.llm_client import LLMClient
from src.utils.monitoring import HealthChecker, AlertManager, AlertSeverity
from src.config.settings import LLMConfig


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter for testing."""
        return RateLimiter()
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_within_limit(self, rate_limiter):
        """Test rate limit check when within limits."""
        # Should be within limit initially
        result = await rate_limiter.check_rate_limit("tweet_create")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_rate_limit_exceeded(self, rate_limiter):
        """Test rate limit check when limit is exceeded."""
        # Simulate exceeding the limit
        limit = rate_limiter.limits["tweet_create"]
        calls = rate_limiter._calls["tweet_create"]
        
        # Fill up the calls to the limit
        current_time = time.time()
        for _ in range(limit.max_calls):
            calls.append(current_time)
        
        result = await rate_limiter.check_rate_limit("tweet_create")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_acquire_blocking(self, rate_limiter):
        """Test blocking acquire method."""
        # Should acquire successfully when within limit
        result = await rate_limiter.acquire("bot_post")
        assert result is True
        
        # Check that the call was recorded
        calls = rate_limiter._calls["bot_post"]
        assert len(calls) == 1
    
    @pytest.mark.asyncio
    async def test_try_acquire_non_blocking(self, rate_limiter):
        """Test non-blocking try_acquire method."""
        # Should succeed when within limit
        result = await rate_limiter.try_acquire("bot_like")
        assert result is True
        
        # Should fail when limit is exceeded
        limit = rate_limiter.limits["bot_like"]
        calls = rate_limiter._calls["bot_like"]
        
        # Fill up to limit
        current_time = time.time()
        for _ in range(limit.max_calls - 1):  # -1 because we already have one call
            calls.append(current_time)
        
        result = await rate_limiter.try_acquire("bot_like")
        assert result is False
    
    def test_apply_backoff(self, rate_limiter):
        """Test backoff application."""
        rate_limiter.apply_backoff("test_key", 60)
        
        assert "test_key" in rate_limiter._backoff_until
        assert rate_limiter._backoff_until["test_key"] > time.time()
    
    def test_get_remaining_calls(self, rate_limiter):
        """Test getting remaining calls."""
        # Initially should have full limit
        remaining = rate_limiter.get_remaining_calls("bot_post")
        limit = rate_limiter.limits["bot_post"]
        assert remaining == limit.max_calls
        
        # After one call, should have one less
        rate_limiter._calls["bot_post"].append(time.time())
        remaining = rate_limiter.get_remaining_calls("bot_post")
        assert remaining == limit.max_calls - 1
    
    def test_get_status(self, rate_limiter):
        """Test status reporting."""
        status = rate_limiter.get_status()
        
        assert isinstance(status, dict)
        assert "tweet_create" in status
        assert "bot_post" in status
        
        for key, info in status.items():
            assert "name" in info
            assert "max_calls" in info
            assert "remaining" in info


class TestExponentialBackoff:
    """Test cases for ExponentialBackoff."""
    
    def test_initial_state(self):
        """Test initial backoff state."""
        backoff = ExponentialBackoff(initial_delay=1.0, max_delay=60.0, multiplier=2.0)
        
        assert backoff.current_delay == 1.0
        assert backoff.attempt == 0
        assert backoff.should_retry is True
    
    @pytest.mark.asyncio
    async def test_wait_and_increment(self):
        """Test wait behavior and delay increment."""
        backoff = ExponentialBackoff(initial_delay=0.01, max_delay=1.0, multiplier=2.0)
        
        # First wait should be fast (no delay on first attempt)
        start_time = time.time()
        await backoff.wait()
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be very quick
        assert backoff.attempt == 1
        
        # Second wait should include delay
        start_time = time.time()
        await backoff.wait()
        elapsed = time.time() - start_time
        assert elapsed >= 0.01  # Should wait at least initial delay
        assert backoff.attempt == 2
        assert backoff.current_delay == 0.02  # Should have multiplied
    
    def test_reset(self):
        """Test backoff reset."""
        backoff = ExponentialBackoff(initial_delay=1.0)
        
        # Simulate some attempts
        backoff.attempt = 5
        backoff.current_delay = 32.0
        
        backoff.reset()
        
        assert backoff.attempt == 0
        assert backoff.current_delay == 1.0
    
    def test_should_retry_limit(self):
        """Test retry limit."""
        backoff = ExponentialBackoff(initial_delay=1.0, max_delay=10.0)
        
        # Should retry initially
        assert backoff.should_retry is True
        
        # Simulate many failures
        backoff.current_delay = 20.0  # Beyond max_delay
        assert backoff.should_retry is False


class TestActionQueue:
    """Test cases for ActionQueue."""
    
    @pytest.fixture
    def action_queue(self):
        """Create action queue for testing."""
        return ActionQueue(min_interval=0.01, max_interval=0.1)  # Fast intervals for testing
    
    @pytest.mark.asyncio
    async def test_add_and_get_action(self, action_queue):
        """Test adding and getting actions."""
        # Add an action
        await action_queue.add_action("test_action", {"data": "test"}, priority=1)
        
        assert action_queue.queue_size() == 1
        
        # Get the action
        action = await action_queue.get_next_action()
        
        assert action is not None
        assert action["type"] == "test_action"
        assert action["data"]["data"] == "test"
        assert action["priority"] == 1
        assert "created_at" in action
    
    @pytest.mark.asyncio
    async def test_timing_control(self, action_queue):
        """Test timing control between actions."""
        # Add two actions
        await action_queue.add_action("action1", {"data": "test1"})
        await action_queue.add_action("action2", {"data": "test2"})
        
        # Get first action (should be immediate)
        start_time = time.time()
        action1 = await action_queue.get_next_action()
        elapsed1 = time.time() - start_time
        assert elapsed1 < 0.1
        
        # Get second action (should include delay)
        start_time = time.time()
        action2 = await action_queue.get_next_action()
        elapsed2 = time.time() - start_time
        assert elapsed2 >= 0.01  # Should wait at least min_interval
    
    @pytest.mark.asyncio
    async def test_empty_queue(self, action_queue):
        """Test behavior with empty queue."""
        action = await action_queue.get_next_action()
        assert action is None


class TestSafetyLimits:
    """Test cases for SafetyLimits."""
    
    @pytest.fixture
    def safety_limits(self):
        """Create safety limits for testing."""
        return SafetyLimits()
    
    def test_initial_state(self, safety_limits):
        """Test initial safety limits state."""
        assert safety_limits.can_perform_action("follows") is True
        assert safety_limits.can_perform_action("posts") is True
        assert safety_limits.can_perform_action("likes") is True
    
    def test_record_and_check_actions(self, safety_limits):
        """Test recording actions and checking limits."""
        # Record some follows
        for _ in range(10):
            safety_limits.record_action("follows")
        
        # Should still be within daily limit
        assert safety_limits.can_perform_action("follows") is True
        
        # Record more to exceed hourly limit
        for _ in range(5):  # Total 15, exceeds hourly limit of 10
            safety_limits.record_action("follows")
        
        # Should now be blocked by hourly limit
        assert safety_limits.can_perform_action("follows") is False
    
    def test_get_remaining_actions(self, safety_limits):
        """Test getting remaining action counts."""
        # Record some actions
        for _ in range(5):
            safety_limits.record_action("posts")
        
        remaining = safety_limits.get_remaining_actions("posts")
        
        assert remaining["daily"] == 5  # 10 daily limit - 5 used = 5 remaining
        assert remaining["hourly"] == float('inf')  # No hourly limit for posts


class TestLLMClient:
    """Test cases for LLMClient."""
    
    @pytest.fixture
    def llm_config(self):
        """Mock LLM configuration."""
        config = Mock(spec=LLMConfig)
        config.openai_api_key = "test_openai_key"
        config.anthropic_api_key = None
        config.preferred_model = "gpt-4"
        config.max_tokens = 1000
        config.temperature = 0.7
        return config
    
    @pytest.fixture
    def llm_client(self, llm_config):
        """Create LLM client for testing."""
        with patch('openai.AsyncOpenAI'), patch('tiktoken.encoding_for_model'):
            client = LLMClient(llm_config)
            client.openai_client = AsyncMock()
            client.encoding = Mock()
            client.encoding.encode.return_value = [1, 2, 3, 4, 5]  # 5 tokens
            return client
    
    @pytest.mark.asyncio
    async def test_generate_content_openai(self, llm_client):
        """Test content generation with OpenAI."""
        # Mock OpenAI response
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Generated content"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.model = "gpt-4"
        mock_response.usage = Mock()
        mock_response.usage.dict.return_value = {"total_tokens": 50}
        
        llm_client.openai_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        result = await llm_client.generate_content(
            "Test prompt",
            content_type="post"
        )
        
        assert result["content"] == "Generated content"
        assert result["model"] == "gpt-4"
        assert result["provider"] == "openai"
        assert "generated_at" in result
    
    def test_validate_content(self, llm_client):
        """Test content validation."""
        # Test valid content
        validation = llm_client.validate_content("Good content", "post")
        assert validation["valid"] is True
        assert validation["character_count"] == len("Good content")
        
        # Test content that's too long
        long_content = "x" * 300
        validation = llm_client.validate_content(long_content, "post")
        assert validation["valid"] is False
        assert "too long" in validation["issues"][0].lower()
        
        # Test content with spam indicators
        spam_content = "Click here to buy now!"
        validation = llm_client.validate_content(spam_content, "post")
        assert len(validation["issues"]) > 0
    
    def test_count_tokens(self, llm_client):
        """Test token counting."""
        text = "This is a test"
        token_count = llm_client.count_tokens(text)
        assert token_count == 5  # Mocked encoding returns 5 tokens
    
    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, llm_client):
        """Test sentiment analysis."""
        # Mock LLM response with JSON
        mock_response = {
            "content": '{"sentiment_score": 0.8, "sentiment_label": "positive", "confidence": 0.9, "reasoning": "Positive language"}',
            "model": "gpt-4"
        }
        
        llm_client.generate_content = AsyncMock(return_value=mock_response)
        
        result = await llm_client.analyze_sentiment("This is amazing!")
        
        assert result["sentiment_score"] == 0.8
        assert result["sentiment_label"] == "positive"
        assert result["confidence"] == 0.9


class TestHealthChecker:
    """Test cases for HealthChecker."""
    
    @pytest.fixture
    def health_checker(self):
        """Create health checker for testing."""
        return HealthChecker()
    
    def test_register_health_check(self, health_checker):
        """Test registering health checks."""
        def test_check():
            return {"healthy": True, "details": {}}
        
        health_checker.register_health_check("test_check", test_check)
        
        assert "test_check" in health_checker.health_checks
        assert health_checker.health_checks["test_check"] == test_check
    
    @pytest.mark.asyncio
    async def test_perform_health_checks(self, health_checker):
        """Test performing health checks."""
        # Register test checks
        def healthy_check():
            return {"healthy": True, "details": {"status": "ok"}}
        
        def unhealthy_check():
            return {"healthy": False, "details": {"status": "error"}}
        
        async def async_check():
            return {"healthy": True, "details": {"status": "async_ok"}}
        
        health_checker.register_health_check("healthy", healthy_check)
        health_checker.register_health_check("unhealthy", unhealthy_check)
        health_checker.register_health_check("async", async_check)
        
        status = await health_checker.perform_health_checks()
        
        assert status["overall"] == "unhealthy"  # One check failed
        assert len(status["components"]) == 3
        assert status["components"]["healthy"]["status"] == "healthy"
        assert status["components"]["unhealthy"]["status"] == "unhealthy"
        assert status["components"]["async"]["status"] == "healthy"


class TestAlertManager:
    """Test cases for AlertManager."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create alert manager for testing."""
        return AlertManager()
    
    @pytest.mark.asyncio
    async def test_create_alert(self, alert_manager):
        """Test alert creation."""
        alert = await alert_manager.create_alert(
            alert_id="test_alert",
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            description="This is a test alert",
            metadata={"test": True}
        )
        
        assert alert is not None
        assert alert.id == "test_alert"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.title == "Test Alert"
        assert not alert.resolved
        
        # Check that alert is stored
        assert "test_alert" in alert_manager.active_alerts
        assert len(alert_manager.alert_history) == 1
    
    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test alert resolution."""
        # Create an alert first
        await alert_manager.create_alert(
            alert_id="test_alert",
            severity=AlertSeverity.ERROR,
            title="Test Alert",
            description="Test"
        )
        
        # Resolve it
        success = await alert_manager.resolve_alert("test_alert")
        
        assert success is True
        assert "test_alert" not in alert_manager.active_alerts
        
        # Check in history that it's marked as resolved
        resolved_alert = next(
            (alert for alert in alert_manager.alert_history if alert.id == "test_alert"),
            None
        )
        assert resolved_alert is not None
        assert resolved_alert.resolved is True
        assert resolved_alert.resolution_timestamp is not None
    
    def test_rate_limiting(self, alert_manager):
        """Test alert rate limiting."""
        alert_id = "rate_limited_alert"
        
        # First alert should not be rate limited
        assert not alert_manager._is_rate_limited(alert_id)
        
        # Mark as sent
        alert_manager.alert_rate_limits[alert_id] = datetime.utcnow()
        
        # Should now be rate limited
        assert alert_manager._is_rate_limited(alert_id, cooldown_minutes=30)
        
        # Should not be rate limited if cooldown has passed
        alert_manager.alert_rate_limits[alert_id] = datetime.utcnow() - timedelta(minutes=31)
        assert not alert_manager._is_rate_limited(alert_id, cooldown_minutes=30)
    
    def test_get_alert_stats(self, alert_manager):
        """Test alert statistics."""
        # Create some test alerts in history
        alert_manager.alert_history = [
            Mock(severity=AlertSeverity.WARNING, timestamp=datetime.utcnow()),
            Mock(severity=AlertSeverity.ERROR, timestamp=datetime.utcnow()),
            Mock(severity=AlertSeverity.WARNING, timestamp=datetime.utcnow() - timedelta(days=2))
        ]
        
        # Create active alert
        alert_manager.active_alerts["active1"] = Mock()
        
        stats = alert_manager.get_alert_stats()
        
        assert stats["total_alerts"] == 3
        assert stats["active_alerts"] == 1
        assert stats["severity_breakdown"]["warning"] == 2
        assert stats["severity_breakdown"]["error"] == 1
        assert stats["alert_rate"] == 2  # 2 alerts in last 24 hours


if __name__ == "__main__":
    pytest.main([__file__])