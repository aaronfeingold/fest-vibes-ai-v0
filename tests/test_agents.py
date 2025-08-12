"""Tests for bot agents."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.config.settings import BotConfig
from src.agents.follow_agent import FollowAgent
from src.agents.content_agent import ContentAgent
from src.agents.engagement_agent import EngagementAgent
from src.models.data_models import User, ContentCache, ContentType, FollowStatus


@pytest.fixture
def mock_config():
    """Mock bot configuration."""
    config = Mock(spec=BotConfig)
    config.agents = Mock()
    config.agents.follow_threshold = 0.7
    config.agents.like_threshold = 0.6
    config.agents.repost_threshold = 0.8
    config.agents.comment_threshold = 0.9
    config.agents.target_location = "New Orleans"
    config.agents.content_domains = ["music", "nola_culture", "genz_trends"]
    config.agents.min_content_cache_size = 10
    config.agents.max_content_cache_size = 50
    
    config.llm = Mock()
    config.llm.openai_api_key = "test_key"
    config.llm.preferred_model = "gpt-4"
    config.llm.max_tokens = 1000
    config.llm.temperature = 0.7
    
    config.development = Mock()
    config.development.dry_run = True
    config.development.mock_twitter_api = True
    
    return config


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB manager."""
    mongodb = AsyncMock()
    mongodb.get_users_for_discovery.return_value = [
        User(
            twitter_user_id="123",
            username="test_user",
            location="New Orleans",
            bio="Local musician",
            engagement_score=0.8,
            relevance_score=0.7
        )
    ]
    mongodb.get_unused_content.return_value = [
        ContentCache(
            content_type=ContentType.POST,
            content="Test content",
            sentiment_score=0.5
        )
    ]
    return mongodb


@pytest.fixture
def mock_postgres():
    """Mock PostgreSQL manager."""
    return AsyncMock()


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter."""
    rate_limiter = Mock()
    rate_limiter.acquire = AsyncMock(return_value=True)
    rate_limiter.try_acquire = AsyncMock(return_value=True)
    rate_limiter.get_status.return_value = {}
    return rate_limiter


class TestFollowAgent:
    """Test cases for FollowAgent."""
    
    @pytest.fixture
    def follow_agent(self, mock_config, mock_mongodb, mock_postgres, mock_rate_limiter):
        """Create follow agent for testing."""
        with patch('src.agents.base_agent.BaseAgent._create_twitter_client'):
            agent = FollowAgent(
                config=mock_config,
                mongodb=mock_mongodb,
                postgres=mock_postgres,
                rate_limiter=mock_rate_limiter
            )
            agent.twitter_client = Mock()
            return agent
    
    @pytest.mark.asyncio
    async def test_analyze_user(self, follow_agent):
        """Test user analysis functionality."""
        user = User(
            twitter_user_id="123",
            username="test_user",
            location="New Orleans, LA",
            bio="Jazz musician in the French Quarter",
            follower_count=500,
            following_count=200,
            tweet_count=1000,
            engagement_score=0.7
        )
        
        analysis = await follow_agent._analyze_user(user)
        
        assert analysis.user_id == "123"
        assert analysis.location_score > 0  # Should score well for New Orleans
        assert analysis.relevance_score > 0  # Should score well for music keywords
        assert analysis.overall_score > 0
        assert analysis.recommendation in ["highly_recommended", "recommended", "consider", "skip"]
    
    @pytest.mark.asyncio
    async def test_score_location(self, follow_agent):
        """Test location scoring."""
        # Test high-scoring location
        score = follow_agent._score_location("New Orleans, LA", "Living in NOLA")
        assert score > 0.3
        
        # Test low-scoring location
        score = follow_agent._score_location("Random City", "No relevant keywords")
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_score_relevance(self, follow_agent):
        """Test relevance scoring."""
        # Test high-relevance content
        score = follow_agent._score_relevance(
            "Jazz musician performing in New Orleans", 
            "French Quarter"
        )
        assert score > 0.2
        
        # Test low-relevance content
        score = follow_agent._score_relevance("Random content", "Nowhere")
        assert score == 0.0
    
    @pytest.mark.asyncio
    async def test_execute(self, follow_agent):
        """Test follow agent execution."""
        with patch.object(follow_agent, '_discover_users', return_value={"count": 5}), \
             patch.object(follow_agent, '_analyze_users_for_follows', return_value={"decisions_made": 3}), \
             patch.object(follow_agent, '_execute_follows', return_value={"executed": 2}), \
             patch.object(follow_agent, '_manage_unfollows', return_value={"executed": 1}), \
             patch.object(follow_agent, '_update_user_relationships', return_value=None):
            
            result = await follow_agent.execute()
            
            assert result["discovered_users"] == 5
            assert result["follow_decisions"] == 3
            assert result["follows_executed"] == 2
            assert result["unfollows_executed"] == 1
    
    def test_get_agent_status(self, follow_agent):
        """Test agent status reporting."""
        status = follow_agent.get_agent_status()
        
        assert "name" in status
        assert "queue_sizes" in status
        assert "safety_limits" in status
        assert status["name"] == "FollowAgent"


class TestContentAgent:
    """Test cases for ContentAgent."""
    
    @pytest.fixture
    def content_agent(self, mock_config, mock_mongodb, mock_postgres, mock_rate_limiter):
        """Create content agent for testing."""
        with patch('src.agents.base_agent.BaseAgent._create_twitter_client'), \
             patch('src.utils.llm_client.LLMClient'):
            agent = ContentAgent(
                config=mock_config,
                mongodb=mock_mongodb,
                postgres=mock_postgres,
                rate_limiter=mock_rate_limiter
            )
            agent.twitter_client = Mock()
            agent.llm_client = Mock()
            return agent
    
    @pytest.mark.asyncio
    async def test_select_best_post(self, content_agent):
        """Test post selection logic."""
        posts = [
            ContentCache(
                content_type=ContentType.POST,
                content="Short post",
                character_count=20,
                sentiment_score=0.3
            ),
            ContentCache(
                content_type=ContentType.POST,
                content="Perfect length post about New Orleans music scene",
                character_count=120,
                sentiment_score=0.8
            ),
            ContentCache(
                content_type=ContentType.POST,
                content="Very long post that goes on and on about various topics without much focus and probably exceeds the optimal length for social media engagement which could hurt performance metrics",
                character_count=280,
                sentiment_score=0.5
            )
        ]
        
        selected = content_agent._select_best_post(posts)
        
        # Should select the post with good length and high sentiment
        assert selected.character_count == 120
        assert selected.sentiment_score == 0.8
    
    @pytest.mark.asyncio
    async def test_generate_contextual_comment(self, content_agent):
        """Test contextual comment generation."""
        tweet_context = {
            "text": "Great jazz performance last night!",
            "author": {"description": "Local music lover"}
        }
        
        # Mock the LLM client
        content_agent.llm_client.generate_content = AsyncMock(return_value={
            "content": "Love seeing local music appreciation!",
            "valid": True
        })
        content_agent.llm_client.validate_content = Mock(return_value={"valid": True})
        
        with patch.object(content_agent, '_generate_content_with_validation') as mock_generate:
            mock_generate.return_value = {
                "content": "Love seeing local music appreciation!",
                "valid": True
            }
            
            comment = await content_agent.generate_contextual_comment(tweet_context)
            
            assert comment is not None
            assert len(comment) > 0
    
    @pytest.mark.asyncio
    async def test_execute(self, content_agent):
        """Test content agent execution."""
        with patch.object(content_agent, '_maintain_content_cache', return_value={"posts_generated": 5, "comments_generated": 10}), \
             patch.object(content_agent, '_publish_scheduled_content', return_value={"published": 1}), \
             patch.object(content_agent, '_generate_contextual_comments', return_value=None), \
             patch.object(content_agent, '_update_content_performance', return_value=None):
            
            result = await content_agent.execute()
            
            assert result["posts_generated"] == 5
            assert result["comments_generated"] == 10
            assert result["posts_published"] == 1


class TestEngagementAgent:
    """Test cases for EngagementAgent."""
    
    @pytest.fixture
    def engagement_agent(self, mock_config, mock_mongodb, mock_postgres, mock_rate_limiter):
        """Create engagement agent for testing."""
        with patch('src.agents.base_agent.BaseAgent._create_twitter_client'):
            agent = EngagementAgent(
                config=mock_config,
                mongodb=mock_mongodb,
                postgres=mock_postgres,
                rate_limiter=mock_rate_limiter
            )
            agent.twitter_client = Mock()
            return agent
    
    def test_score_content_relevance(self, engagement_agent):
        """Test content relevance scoring."""
        # Test high-relevance content
        score = engagement_agent._score_content_relevance(
            "Amazing jazz performance in New Orleans last night! #NOLA #jazz"
        )
        assert score > 0.3
        
        # Test spam content
        score = engagement_agent._score_content_relevance(
            "Follow me back! Check out my profile! Buy now!"
        )
        assert score < 0.1  # Should be heavily penalized
    
    def test_score_engagement_potential(self, engagement_agent):
        """Test engagement potential scoring."""
        # Test question tweet (high engagement potential)
        tweet = {
            "text": "What's your favorite New Orleans music venue?",
            "public_metrics": {"like_count": 10, "retweet_count": 2}
        }
        score = engagement_agent._score_engagement_potential(tweet, "comment")
        assert score > 0.4  # Should score high for questions
        
        # Test promotional content (low engagement potential)
        tweet = {
            "text": "Buy my new album now! Limited time offer!",
            "public_metrics": {"like_count": 1, "retweet_count": 0}
        }
        score = engagement_agent._score_engagement_potential(tweet, "like")
        assert score < 0.2  # Should score low for promotional content
    
    def test_score_timing(self, engagement_agent):
        """Test timing scoring."""
        # Test very fresh content
        fresh_time = datetime.utcnow() - timedelta(minutes=30)
        score = engagement_agent._score_timing(fresh_time)
        assert score >= 0.8
        
        # Test old content
        old_time = datetime.utcnow() - timedelta(days=2)
        score = engagement_agent._score_timing(old_time)
        assert score <= 0.3
    
    def test_score_authenticity(self, engagement_agent):
        """Test authenticity scoring."""
        # Test authentic content
        score = engagement_agent._score_authenticity(
            "Just discovered this amazing local artist performing tonight"
        )
        assert score >= 0.8
        
        # Test spammy content
        score = engagement_agent._score_authenticity(
            "FOLLOW BACK!!! CHECK OUT MY PROFILE!!! DM ME NOW!!! #follow #followback"
        )
        assert score <= 0.3  # Should be heavily penalized
    
    @pytest.mark.asyncio
    async def test_analyze_like_opportunity(self, engagement_agent):
        """Test like opportunity analysis."""
        tweet = {
            "id": "123456",
            "text": "Beautiful jazz performance in the French Quarter tonight!",
            "author_id": "789",
            "created_at": datetime.utcnow() - timedelta(minutes=30),
            "public_metrics": {"like_count": 15, "retweet_count": 3}
        }
        
        with patch.object(engagement_agent, '_score_author_relevance', return_value=0.8):
            decision = await engagement_agent._analyze_like_opportunity(tweet)
            
            assert decision.tweet_id == "123456"
            assert decision.action_type.value == "like"
            assert decision.score > 0
            assert isinstance(decision.decision, bool)
            assert "author_relevance" in decision.factors
            assert "content_relevance" in decision.factors


@pytest.mark.asyncio
async def test_agent_integration():
    """Test integration between agents."""
    # This would test how agents work together
    # For example, how the content agent provides content for the engagement agent
    pass


if __name__ == "__main__":
    pytest.main([__file__])