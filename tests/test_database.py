"""Tests for database managers."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.database.mongodb_manager import MongoDBManager
from src.database.postgres_manager import PostgreSQLManager
from src.config.settings import DatabaseConfig
from src.models.data_models import (
    User, ContentCache, EngagementHistory, ContentType, 
    ActionType, ContentEmbedding, UserInterestVector
)


@pytest.fixture
def db_config():
    """Mock database configuration."""
    config = Mock(spec=DatabaseConfig)
    config.mongodb_uri = "mongodb://test:test@localhost:27017/test"
    config.postgres_uri = "postgresql://test:test@localhost:5432/test"
    config.mongodb_max_pool_size = 5
    config.postgres_max_pool_size = 5
    return config


class TestMongoDBManager:
    """Test cases for MongoDB manager."""
    
    @pytest.fixture
    async def mongodb_manager(self, db_config):
        """Create MongoDB manager for testing."""
        with patch('motor.motor_asyncio.AsyncIOMotorClient') as mock_client:
            mock_db = AsyncMock()
            mock_client.return_value = mock_client
            mock_client.__getitem__.return_value = mock_db
            mock_client.admin.command = AsyncMock()
            
            manager = MongoDBManager(db_config)
            manager.client = mock_client
            manager.database = mock_db
            manager._connected = True
            
            yield manager
    
    @pytest.mark.asyncio
    async def test_create_user(self, mongodb_manager):
        """Test user creation."""
        user = User(
            twitter_user_id="123",
            username="test_user",
            location="New Orleans",
            bio="Test bio",
            engagement_score=0.8
        )
        
        # Mock the insert operation
        mock_result = AsyncMock()
        mock_result.inserted_id = "507f1f77bcf86cd799439011"
        mongodb_manager.database.users.insert_one = AsyncMock(return_value=mock_result)
        
        user_id = await mongodb_manager.create_user(user)
        
        assert user_id == "507f1f77bcf86cd799439011"
        mongodb_manager.database.users.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_user(self, mongodb_manager):
        """Test user retrieval."""
        mock_user_doc = {
            "twitter_user_id": "123",
            "username": "test_user",
            "location": "New Orleans",
            "bio": "Test bio",
            "engagement_score": 0.8,
            "discovered_date": datetime.utcnow(),
            "last_updated": datetime.utcnow()
        }
        
        mongodb_manager.database.users.find_one = AsyncMock(return_value=mock_user_doc)
        
        user = await mongodb_manager.get_user("123")
        
        assert user is not None
        assert user.twitter_user_id == "123"
        assert user.username == "test_user"
        assert user.engagement_score == 0.8
    
    @pytest.mark.asyncio
    async def test_update_user(self, mongodb_manager):
        """Test user update."""
        mock_result = AsyncMock()
        mock_result.modified_count = 1
        mongodb_manager.database.users.update_one = AsyncMock(return_value=mock_result)
        
        success = await mongodb_manager.update_user("123", {"engagement_score": 0.9})
        
        assert success is True
        mongodb_manager.database.users.update_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_content(self, mongodb_manager):
        """Test content creation."""
        content = ContentCache(
            content_type=ContentType.POST,
            content="Test content",
            tags=["music"]
        )
        
        mock_result = AsyncMock()
        mock_result.inserted_id = "507f1f77bcf86cd799439011"
        mongodb_manager.database.content_cache.insert_one = AsyncMock(return_value=mock_result)
        
        content_id = await mongodb_manager.create_content(content)
        
        assert content_id == "507f1f77bcf86cd799439011"
        mongodb_manager.database.content_cache.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_unused_content(self, mongodb_manager):
        """Test unused content retrieval."""
        mock_content_docs = [
            {
                "content_type": "post",
                "content": "Test content 1",
                "generated_at": datetime.utcnow(),
                "used": False,
                "tags": ["music"]
            },
            {
                "content_type": "post",
                "content": "Test content 2",
                "generated_at": datetime.utcnow(),
                "used": False,
                "tags": ["culture"]
            }
        ]
        
        # Mock cursor
        mock_cursor = AsyncMock()
        mock_cursor.__aiter__.return_value = mock_content_docs
        
        mongodb_manager.database.content_cache.find.return_value = mock_cursor
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = mock_cursor
        
        content_list = await mongodb_manager.get_unused_content("post", limit=10)
        
        assert len(content_list) == 2
        assert all(content.content_type.value == "post" for content in content_list)
    
    @pytest.mark.asyncio
    async def test_create_engagement(self, mongodb_manager):
        """Test engagement history creation."""
        engagement = EngagementHistory(
            target_tweet_id="123",
            target_user_id="456",
            action_type=ActionType.LIKE,
            success=True,
            response_data={"status": "success"}
        )
        
        mock_result = AsyncMock()
        mock_result.inserted_id = "507f1f77bcf86cd799439011"
        mongodb_manager.database.engagement_history.insert_one = AsyncMock(return_value=mock_result)
        
        engagement_id = await mongodb_manager.create_engagement(engagement)
        
        assert engagement_id == "507f1f77bcf86cd799439011"
        mongodb_manager.database.engagement_history.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_engagement_count(self, mongodb_manager):
        """Test engagement count retrieval."""
        mongodb_manager.database.engagement_history.count_documents = AsyncMock(return_value=5)
        
        count = await mongodb_manager.get_engagement_count("like", hours=1)
        
        assert count == 5
        mongodb_manager.database.engagement_history.count_documents.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, mongodb_manager):
        """Test old data cleanup."""
        # Mock delete results
        mock_eng_result = AsyncMock()
        mock_eng_result.deleted_count = 10
        mock_metrics_result = AsyncMock()
        mock_metrics_result.deleted_count = 5
        mock_content_result = AsyncMock()
        mock_content_result.deleted_count = 3
        
        mongodb_manager.database.engagement_history.delete_many = AsyncMock(return_value=mock_eng_result)
        mongodb_manager.database.bot_metrics.delete_many = AsyncMock(return_value=mock_metrics_result)
        mongodb_manager.database.content_cache.delete_many = AsyncMock(return_value=mock_content_result)
        
        stats = await mongodb_manager.cleanup_old_data(days_to_keep=30)
        
        assert stats["engagements_deleted"] == 10
        assert stats["metrics_deleted"] == 5
        assert stats["content_deleted"] == 3


class TestPostgreSQLManager:
    """Test cases for PostgreSQL manager."""
    
    @pytest.fixture
    async def postgres_manager(self, db_config):
        """Create PostgreSQL manager for testing."""
        with patch('asyncpg.create_pool') as mock_pool:
            mock_connection = AsyncMock()
            mock_pool.return_value = mock_pool
            mock_pool.acquire.return_value.__aenter__.return_value = mock_connection
            mock_pool.acquire.return_value.__aexit__.return_value = None
            
            manager = PostgreSQLManager(db_config)
            manager.pool = mock_pool
            manager._connected = True
            
            yield manager
    
    @pytest.mark.asyncio
    async def test_create_content_embedding(self, postgres_manager):
        """Test content embedding creation."""
        embedding = ContentEmbedding(
            content_hash="abc123",
            content="Test content",
            embedding=[0.1, 0.2, 0.3] * 512,  # 1536 dimensions
            content_type="post"
        )
        
        # Mock connection and query result
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        postgres_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        embedding_id = await postgres_manager.create_content_embedding(embedding)
        
        assert embedding_id == 1
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_similar_content(self, postgres_manager):
        """Test similar content search."""
        query_embedding = [0.1, 0.2, 0.3] * 512  # 1536 dimensions
        
        # Mock similar content results
        mock_results = [
            {"id": 1, "content": "Similar content 1", "distance": 0.1},
            {"id": 2, "content": "Similar content 2", "distance": 0.2}
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_results)
        postgres_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        results = await postgres_manager.find_similar_content(
            query_embedding, 
            content_type="post", 
            similarity_threshold=0.8,
            limit=10
        )
        
        assert len(results) == 2
        assert results[0].content_id == 1
        assert results[0].similarity == 0.9  # 1 - 0.1
        assert results[1].content_id == 2
        assert results[1].similarity == 0.8  # 1 - 0.2
    
    @pytest.mark.asyncio
    async def test_create_user_interest_vector(self, postgres_manager):
        """Test user interest vector creation."""
        user_vector = UserInterestVector(
            twitter_user_id="123",
            interest_embedding=[0.1, 0.2, 0.3] * 512,
            interaction_count=5,
            engagement_score=0.8
        )
        
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        postgres_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        success = await postgres_manager.create_user_interest_vector(user_vector)
        
        assert success is True
        mock_conn.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_find_similar_users(self, postgres_manager):
        """Test similar users search."""
        query_embedding = [0.1, 0.2, 0.3] * 512
        
        mock_results = [
            {"twitter_user_id": "123", "distance": 0.1, "engagement_score": 0.8},
            {"twitter_user_id": "456", "distance": 0.2, "engagement_score": 0.7}
        ]
        
        mock_conn = AsyncMock()
        mock_conn.fetch = AsyncMock(return_value=mock_results)
        postgres_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        results = await postgres_manager.find_similar_users(
            query_embedding,
            similarity_threshold=0.8,
            limit=10
        )
        
        assert len(results) == 2
        assert results[0].user_id == "123"
        assert results[0].similarity == 0.9
        assert results[0].engagement_score == 0.8
    
    @pytest.mark.asyncio
    async def test_cache_search_result(self, postgres_manager):
        """Test search result caching."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=1)
        postgres_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        cache_id = await postgres_manager.cache_search_result(
            query_text="test query",
            query_embedding=[0.1, 0.2, 0.3] * 512,
            result_content_ids=[1, 2, 3],
            search_type="similarity",
            expires_hours=24
        )
        
        assert cache_id == 1
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_cached_search_result(self, postgres_manager):
        """Test cached search result retrieval."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=[1, 2, 3])
        postgres_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        result = await postgres_manager.get_cached_search_result(
            query_text="test query",
            search_type="similarity"
        )
        
        assert result == [1, 2, 3]
        mock_conn.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_clean_expired_cache(self, postgres_manager):
        """Test expired cache cleanup."""
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock(return_value=5)
        postgres_manager.pool.acquire.return_value.__aenter__.return_value = mock_conn
        
        deleted_count = await postgres_manager.clean_expired_cache()
        
        assert deleted_count == 5
        mock_conn.fetchval.assert_called_once()
    
    def test_compute_content_hash(self):
        """Test content hash computation."""
        content = "Test content"
        hash_value = PostgreSQLManager.compute_content_hash(content)
        
        assert len(hash_value) == 64  # SHA-256 hash length
        assert isinstance(hash_value, str)
        
        # Same content should produce same hash
        hash_value2 = PostgreSQLManager.compute_content_hash(content)
        assert hash_value == hash_value2
    
    def test_normalize_embedding(self):
        """Test embedding normalization."""
        embedding = [3.0, 4.0, 0.0]  # Vector with magnitude 5
        normalized = PostgreSQLManager.normalize_embedding(embedding)
        
        # Check that the normalized vector has magnitude 1
        import math
        magnitude = math.sqrt(sum(x*x for x in normalized))
        assert abs(magnitude - 1.0) < 1e-10
        
        # Check proportions are preserved
        assert abs(normalized[0] - 0.6) < 1e-10  # 3/5
        assert abs(normalized[1] - 0.8) < 1e-10  # 4/5
        assert abs(normalized[2] - 0.0) < 1e-10  # 0/5


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database operations."""
    
    @pytest.mark.asyncio
    async def test_user_content_flow(self):
        """Test the flow from user discovery to content engagement."""
        # This would test the complete flow:
        # 1. User discovered and stored in MongoDB
        # 2. User interest vector computed and stored in PostgreSQL
        # 3. Content generated and cached in MongoDB
        # 4. Content embedding stored in PostgreSQL
        # 5. Engagement actions recorded in MongoDB
        pass
    
    @pytest.mark.asyncio
    async def test_content_similarity_workflow(self):
        """Test content similarity and recommendation workflow."""
        # This would test:
        # 1. Content stored with embeddings
        # 2. Similar content discovery
        # 3. Cache performance
        pass


if __name__ == "__main__":
    pytest.main([__file__])