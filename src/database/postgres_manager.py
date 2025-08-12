"""PostgreSQL with pgvector connection and operations manager."""

import asyncio
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

import asyncpg
import numpy as np
from loguru import logger

from src.models.data_models import (
    ContentEmbedding, 
    UserInterestVector, 
    TweetEmbedding,
    SemanticSearchResult,
    UserSearchResult
)
from src.config.settings import DatabaseConfig


class PostgreSQLManager:
    """PostgreSQL with pgvector connection and operations manager."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL."""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.postgres_uri,
                min_size=1,
                max_size=self.config.postgres_max_pool_size,
                command_timeout=60,
            )
            
            # Test connection and ensure pgvector extension is enabled
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
                
                # Check if pgvector is available
                result = await conn.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
                )
                if not result:
                    raise RuntimeError("pgvector extension is not installed")
            
            self._connected = True
            logger.info("Successfully connected to PostgreSQL with pgvector")
            
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close PostgreSQL connection pool."""
        if self.pool:
            await self.pool.close()
            self._connected = False
            logger.info("Disconnected from PostgreSQL")
    
    # Content embedding operations
    
    async def create_content_embedding(self, content_embedding: ContentEmbedding) -> int:
        """Create content embedding record."""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO content_embeddings 
                    (content_hash, content, embedding, content_type, created_at)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (content_hash) 
                    DO UPDATE SET 
                        embedding = $3,
                        updated_at = $5
                    RETURNING id
                """
                
                embedding_id = await conn.fetchval(
                    query,
                    content_embedding.content_hash,
                    content_embedding.content,
                    content_embedding.embedding,
                    content_embedding.content_type,
                    content_embedding.created_at
                )
                
                logger.debug(f"Created/updated content embedding {embedding_id}")
                return embedding_id
                
        except Exception as e:
            logger.error(f"Failed to create content embedding: {e}")
            raise
    
    async def get_content_embedding(self, content_hash: str) -> Optional[ContentEmbedding]:
        """Get content embedding by hash."""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT id, content_hash, content, embedding, content_type, created_at, updated_at
                    FROM content_embeddings 
                    WHERE content_hash = $1
                """
                
                row = await conn.fetchrow(query, content_hash)
                if row:
                    return ContentEmbedding(
                        id=row["id"],
                        content_hash=row["content_hash"],
                        content=row["content"],
                        embedding=row["embedding"],
                        content_type=row["content_type"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get content embedding: {e}")
            raise
    
    async def find_similar_content(
        self,
        query_embedding: List[float],
        content_type: Optional[str] = None,
        similarity_threshold: float = 0.8,
        limit: int = 10
    ) -> List[SemanticSearchResult]:
        """Find similar content using vector similarity."""
        try:
            async with self.pool.acquire() as conn:
                if content_type:
                    query = """
                        SELECT id, content, (embedding <=> $1) as distance
                        FROM content_embeddings 
                        WHERE content_type = $2 
                            AND (embedding <=> $1) <= $3
                        ORDER BY embedding <=> $1
                        LIMIT $4
                    """
                    rows = await conn.fetch(
                        query, 
                        query_embedding, 
                        content_type,
                        1 - similarity_threshold,  # Convert similarity to distance
                        limit
                    )
                else:
                    query = """
                        SELECT id, content, (embedding <=> $1) as distance
                        FROM content_embeddings 
                        WHERE (embedding <=> $1) <= $2
                        ORDER BY embedding <=> $1
                        LIMIT $3
                    """
                    rows = await conn.fetch(
                        query, 
                        query_embedding,
                        1 - similarity_threshold,
                        limit
                    )
                
                results = []
                for row in rows:
                    results.append(SemanticSearchResult(
                        content_id=row["id"],
                        content=row["content"],
                        similarity=1 - row["distance"]  # Convert distance back to similarity
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to find similar content: {e}")
            raise
    
    # User interest vector operations
    
    async def create_user_interest_vector(self, user_vector: UserInterestVector) -> bool:
        """Create or update user interest vector."""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO user_interest_vectors 
                    (twitter_user_id, interest_embedding, last_updated, interaction_count, engagement_score)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (twitter_user_id) 
                    DO UPDATE SET 
                        interest_embedding = $2,
                        last_updated = $3,
                        interaction_count = user_interest_vectors.interaction_count + $4,
                        engagement_score = $5
                """
                
                await conn.execute(
                    query,
                    user_vector.twitter_user_id,
                    user_vector.interest_embedding,
                    user_vector.last_updated,
                    user_vector.interaction_count,
                    user_vector.engagement_score
                )
                
                logger.debug(f"Created/updated user interest vector for {user_vector.twitter_user_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create user interest vector: {e}")
            raise
    
    async def get_user_interest_vector(self, twitter_user_id: str) -> Optional[UserInterestVector]:
        """Get user interest vector by Twitter user ID."""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT twitter_user_id, interest_embedding, last_updated, 
                           interaction_count, engagement_score
                    FROM user_interest_vectors 
                    WHERE twitter_user_id = $1
                """
                
                row = await conn.fetchrow(query, twitter_user_id)
                if row:
                    return UserInterestVector(
                        twitter_user_id=row["twitter_user_id"],
                        interest_embedding=row["interest_embedding"],
                        last_updated=row["last_updated"],
                        interaction_count=row["interaction_count"],
                        engagement_score=row["engagement_score"]
                    )
                return None
                
        except Exception as e:
            logger.error(f"Failed to get user interest vector: {e}")
            raise
    
    async def find_similar_users(
        self,
        query_embedding: List[float],
        similarity_threshold: float = 0.8,
        limit: int = 10
    ) -> List[UserSearchResult]:
        """Find similar users based on interest vectors."""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT twitter_user_id, (interest_embedding <=> $1) as distance, engagement_score
                    FROM user_interest_vectors 
                    WHERE (interest_embedding <=> $1) <= $2
                    ORDER BY interest_embedding <=> $1
                    LIMIT $3
                """
                
                rows = await conn.fetch(
                    query,
                    query_embedding,
                    1 - similarity_threshold,
                    limit
                )
                
                results = []
                for row in rows:
                    results.append(UserSearchResult(
                        user_id=row["twitter_user_id"],
                        similarity=1 - row["distance"],
                        engagement_score=row["engagement_score"]
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to find similar users: {e}")
            raise
    
    # Tweet embedding operations
    
    async def create_tweet_embedding(self, tweet_embedding: TweetEmbedding) -> int:
        """Create tweet embedding record."""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO tweet_embeddings 
                    (tweet_id, tweet_text, embedding, author_id, engagement_metrics, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (tweet_id) 
                    DO UPDATE SET 
                        embedding = $3,
                        engagement_metrics = $5
                    RETURNING id
                """
                
                embedding_id = await conn.fetchval(
                    query,
                    tweet_embedding.tweet_id,
                    tweet_embedding.tweet_text,
                    tweet_embedding.embedding,
                    tweet_embedding.author_id,
                    tweet_embedding.engagement_metrics,
                    tweet_embedding.created_at
                )
                
                logger.debug(f"Created/updated tweet embedding {embedding_id}")
                return embedding_id
                
        except Exception as e:
            logger.error(f"Failed to create tweet embedding: {e}")
            raise
    
    async def get_tweet_embeddings_by_author(
        self, 
        author_id: str, 
        limit: int = 100
    ) -> List[TweetEmbedding]:
        """Get tweet embeddings by author ID."""
        try:
            async with self.pool.acquire() as conn:
                query = """
                    SELECT id, tweet_id, tweet_text, embedding, author_id, 
                           engagement_metrics, created_at
                    FROM tweet_embeddings 
                    WHERE author_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """
                
                rows = await conn.fetch(query, author_id, limit)
                
                embeddings = []
                for row in rows:
                    embeddings.append(TweetEmbedding(
                        id=row["id"],
                        tweet_id=row["tweet_id"],
                        tweet_text=row["tweet_text"],
                        embedding=row["embedding"],
                        author_id=row["author_id"],
                        engagement_metrics=row["engagement_metrics"],
                        created_at=row["created_at"]
                    ))
                
                return embeddings
                
        except Exception as e:
            logger.error(f"Failed to get tweet embeddings by author: {e}")
            raise
    
    # Semantic search cache operations
    
    async def cache_search_result(
        self,
        query_text: str,
        query_embedding: List[float],
        result_content_ids: List[int],
        search_type: str,
        expires_hours: int = 24
    ) -> int:
        """Cache semantic search results."""
        try:
            query_hash = hashlib.sha256(
                f"{query_text}_{search_type}".encode()
            ).hexdigest()[:64]
            
            expires_at = datetime.utcnow() + timedelta(hours=expires_hours)
            
            async with self.pool.acquire() as conn:
                query = """
                    INSERT INTO semantic_search_cache 
                    (query_hash, query_text, query_embedding, result_content_ids, 
                     search_type, created_at, expires_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (query_hash) 
                    DO UPDATE SET 
                        result_content_ids = $4,
                        expires_at = $7
                    RETURNING id
                """
                
                cache_id = await conn.fetchval(
                    query,
                    query_hash,
                    query_text,
                    query_embedding,
                    result_content_ids,
                    search_type,
                    datetime.utcnow(),
                    expires_at
                )
                
                return cache_id
                
        except Exception as e:
            logger.error(f"Failed to cache search result: {e}")
            raise
    
    async def get_cached_search_result(
        self, 
        query_text: str, 
        search_type: str
    ) -> Optional[List[int]]:
        """Get cached search results if available and not expired."""
        try:
            query_hash = hashlib.sha256(
                f"{query_text}_{search_type}".encode()
            ).hexdigest()[:64]
            
            async with self.pool.acquire() as conn:
                query = """
                    SELECT result_content_ids
                    FROM semantic_search_cache 
                    WHERE query_hash = $1 
                      AND search_type = $2 
                      AND expires_at > $3
                """
                
                result = await conn.fetchval(
                    query,
                    query_hash,
                    search_type,
                    datetime.utcnow()
                )
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get cached search result: {e}")
            raise
    
    # Utility operations
    
    async def clean_expired_cache(self) -> int:
        """Clean expired cache entries."""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval(
                    "SELECT clean_expired_cache()"
                )
                
                logger.info(f"Cleaned {result} expired cache entries")
                return result
                
        except Exception as e:
            logger.error(f"Failed to clean expired cache: {e}")
            raise
    
    async def get_vector_stats(self) -> Dict[str, Any]:
        """Get vector database statistics."""
        try:
            async with self.pool.acquire() as conn:
                stats = {}
                
                # Content embeddings count
                stats["content_embeddings_count"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM content_embeddings"
                )
                
                # User vectors count
                stats["user_vectors_count"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM user_interest_vectors"
                )
                
                # Tweet embeddings count
                stats["tweet_embeddings_count"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM tweet_embeddings"
                )
                
                # Cache entries count
                stats["cache_entries_count"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM semantic_search_cache"
                )
                
                # Recent activity (last 24 hours)
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                
                stats["recent_content_embeddings"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM content_embeddings WHERE created_at >= $1",
                    recent_cutoff
                )
                
                stats["recent_tweet_embeddings"] = await conn.fetchval(
                    "SELECT COUNT(*) FROM tweet_embeddings WHERE created_at >= $1",
                    recent_cutoff
                )
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get vector stats: {e}")
            raise
    
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA-256 hash for content."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    @staticmethod
    def normalize_embedding(embedding: List[float]) -> List[float]:
        """Normalize embedding vector for better similarity search."""
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm == 0:
            return embedding
        return (embedding_array / norm).tolist()