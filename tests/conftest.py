"""Pytest configuration and fixtures."""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock

# Set test environment
os.environ["DEBUG"] = "true"
os.environ["DRY_RUN"] = "true"
os.environ["MOCK_TWITTER_API"] = "true"


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_twitter_response():
    """Mock Twitter API response."""
    return {
        "data": [
            {
                "id": "1234567890",
                "text": "Test tweet content",
                "author_id": "9876543210",
                "created_at": "2023-01-01T00:00:00.000Z",
                "public_metrics": {
                    "like_count": 10,
                    "retweet_count": 2,
                    "reply_count": 1,
                    "quote_count": 0,
                },
            }
        ]
    }


@pytest.fixture
def mock_llm_response():
    """Mock LLM API response."""
    return {
        "content": "Generated content from LLM",
        "model": "gpt-4",
        "usage": {"total_tokens": 50},
        "finish_reason": "stop",
        "provider": "openai",
    }


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "id": "123456789",
        "username": "test_user",
        "name": "Test User",
        "description": "Local New Orleans musician who loves jazz",
        "location": "New Orleans, LA",
        "created_at": "2020-01-01T00:00:00.000Z",
        "public_metrics": {
            "followers_count": 500,
            "following_count": 200,
            "tweet_count": 1000,
            "listed_count": 5,
        },
        "verified": False,
    }


@pytest.fixture
def sample_tweet_data():
    """Sample tweet data for testing."""
    return {
        "id": "1234567890123456789",
        "text": "Amazing jazz performance at Preservation Hall last night! #NOLA #jazz",
        "author_id": "123456789",
        "created_at": "2023-01-01T20:00:00.000Z",
        "public_metrics": {
            "like_count": 25,
            "retweet_count": 5,
            "reply_count": 3,
            "quote_count": 1,
        },
        "context_annotations": [
            {"domain": {"name": "Music"}, "entity": {"name": "Jazz"}},
            {"domain": {"name": "Place"}, "entity": {"name": "New Orleans"}},
        ],
    }


@pytest.fixture
def sample_content_data():
    """Sample content data for testing."""
    return [
        {
            "content": "The energy in the French Quarter tonight is unmatched",
            "content_type": "post",
            "tags": ["culture", "nola"],
            "sentiment_score": 0.8,
        },
        {
            "content": "Jazz brunches hit different when you're in the right city 🎺",
            "content_type": "post",
            "tags": ["music", "food"],
            "sentiment_score": 0.7,
        },
        {
            "content": "This is so important for our local music scene!",
            "content_type": "comment",
            "tags": ["supportive"],
            "sentiment_score": 0.6,
        },
    ]


@pytest.fixture
def mock_database_data():
    """Mock database data for testing."""
    return {
        "users": [
            {
                "twitter_user_id": "123",
                "username": "nola_musician",
                "location": "New Orleans",
                "bio": "Jazz trumpet player in the Quarter",
                "engagement_score": 0.8,
                "relevance_score": 0.9,
                "follow_status": "not_following",
            },
            {
                "twitter_user_id": "456",
                "username": "bounce_producer",
                "location": "New Orleans, LA",
                "bio": "Making beats that make the city move",
                "engagement_score": 0.7,
                "relevance_score": 0.8,
                "follow_status": "following",
            },
        ],
        "content": [
            {
                "content_type": "post",
                "content": "Second line Sunday energy ⚡",
                "used": False,
                "sentiment_score": 0.9,
            },
            {
                "content_type": "comment",
                "content": "This hits different in the best way",
                "used": False,
                "sentiment_score": 0.7,
            },
        ],
        "engagements": [
            {
                "target_tweet_id": "789",
                "target_user_id": "123",
                "action_type": "like",
                "success": True,
                "timestamp": "2023-01-01T12:00:00.000Z",
            }
        ],
    }


@pytest.fixture
def mock_embedding():
    """Mock embedding vector for testing."""
    # Create a 1536-dimensional vector (OpenAI embedding size)
    return [0.1] * 1536


@pytest.fixture
def mock_metrics_data():
    """Mock metrics data for testing."""
    return [
        {
            "metric_type": "execution_success",
            "agent": "FollowAgent",
            "value": 1.0,
            "timestamp": "2023-01-01T12:00:00.000Z",
            "metadata": {"execution_time": 2.5},
        },
        {
            "metric_type": "engagement_rate",
            "agent": "EngagementAgent",
            "value": 0.05,
            "timestamp": "2023-01-01T12:05:00.000Z",
            "metadata": {"tweet_id": "123456"},
        },
    ]


# Test markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "external: mark test as requiring external services"
    )
