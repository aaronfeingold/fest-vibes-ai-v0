"""MongoDB connection and operations manager."""

import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import IndexModel, ASCENDING, DESCENDING
from loguru import logger

from src.models.data_models import User, ContentCache, EngagementHistory, BotMetrics
from src.config.settings import DatabaseConfig


class MongoDBManager:
    """MongoDB connection and operations manager."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self._connected = False
    
    async def connect(self) -> None:
        """Establish connection to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(
                self.config.mongodb_uri,
                maxPoolSize=self.config.mongodb_max_pool_size,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
            
            # Get database name from URI or use default
            db_name = "twitter_bot"
            if "/" in self.config.mongodb_uri:
                db_name = self.config.mongodb_uri.split("/")[-1].split("?")[0]
            
            self.database = self.client[db_name]
            
            # Test connection
            await self.client.admin.command("ping")
            await self._create_indexes()
            
            self._connected = True
            logger.info("Successfully connected to MongoDB")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self._connected = False
            logger.info("Disconnected from MongoDB")
    
    async def _create_indexes(self) -> None:
        """Create database indexes for optimal performance."""
        try:
            # Users collection indexes
            users_indexes = [
                IndexModel([("twitter_user_id", ASCENDING)], unique=True),
                IndexModel([("username", ASCENDING)]),
                IndexModel([("location", ASCENDING)]),
                IndexModel([("follow_status", ASCENDING)]),
                IndexModel([("engagement_score", DESCENDING)]),
                IndexModel([("last_activity", DESCENDING)]),
                IndexModel([("discovered_date", DESCENDING)]),
                IndexModel([("tags", ASCENDING)]),
            ]
            await self.database.users.create_indexes(users_indexes)
            
            # Content cache indexes
            content_indexes = [
                IndexModel([("content_type", ASCENDING)]),
                IndexModel([("generated_at", DESCENDING)]),
                IndexModel([("used", ASCENDING)]),
                IndexModel([("tags", ASCENDING)]),
                IndexModel([("performance_metrics.likes", DESCENDING)]),
            ]
            await self.database.content_cache.create_indexes(content_indexes)
            
            # Engagement history indexes
            engagement_indexes = [
                IndexModel([("target_tweet_id", ASCENDING)]),
                IndexModel([("action_type", ASCENDING)]),
                IndexModel([("timestamp", DESCENDING)]),
                IndexModel([("success", ASCENDING)]),
                IndexModel([("target_user_id", ASCENDING)]),
            ]
            await self.database.engagement_history.create_indexes(engagement_indexes)
            
            # Bot metrics indexes
            metrics_indexes = [
                IndexModel([("metric_type", ASCENDING)]),
                IndexModel([("timestamp", DESCENDING)]),
                IndexModel([("agent", ASCENDING)]),
            ]
            await self.database.bot_metrics.create_indexes(metrics_indexes)
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create MongoDB indexes: {e}")
            raise
    
    # User operations
    
    async def create_user(self, user: User) -> str:
        """Create a new user record."""
        try:
            user_dict = user.dict(by_alias=True, exclude_unset=True)
            user_dict.pop("id", None)  # Remove id to let MongoDB generate it
            
            result = await self.database.users.insert_one(user_dict)
            logger.debug(f"Created user record for {user.username}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to create user {user.username}: {e}")
            raise
    
    async def get_user(self, twitter_user_id: str) -> Optional[User]:
        """Get user by Twitter user ID."""
        try:
            user_doc = await self.database.users.find_one(
                {"twitter_user_id": twitter_user_id}
            )
            return User(**user_doc) if user_doc else None
            
        except Exception as e:
            logger.error(f"Failed to get user {twitter_user_id}: {e}")
            raise
    
    async def update_user(self, twitter_user_id: str, update_data: Dict[str, Any]) -> bool:
        """Update user record."""
        try:
            update_data["last_updated"] = datetime.utcnow()
            
            result = await self.database.users.update_one(
                {"twitter_user_id": twitter_user_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to update user {twitter_user_id}: {e}")
            raise
    
    async def get_users_by_follow_status(self, status: str, limit: int = 100) -> List[User]:
        """Get users by follow status."""
        try:
            cursor = self.database.users.find({"follow_status": status}).limit(limit)
            users = []
            async for user_doc in cursor:
                users.append(User(**user_doc))
            return users
            
        except Exception as e:
            logger.error(f"Failed to get users by status {status}: {e}")
            raise
    
    async def get_users_for_discovery(self, limit: int = 100) -> List[User]:
        """Get users suitable for discovery (not following, high engagement)."""
        try:
            cursor = self.database.users.find({
                "follow_status": "not_following",
                "engagement_score": {"$gte": 0.5}
            }).sort("engagement_score", DESCENDING).limit(limit)
            
            users = []
            async for user_doc in cursor:
                users.append(User(**user_doc))
            return users
            
        except Exception as e:
            logger.error(f"Failed to get discovery users: {e}")
            raise
    
    # Content cache operations
    
    async def create_content(self, content: ContentCache) -> str:
        """Create cached content."""
        try:
            content_dict = content.dict(by_alias=True, exclude_unset=True)
            content_dict.pop("id", None)
            
            result = await self.database.content_cache.insert_one(content_dict)
            logger.debug(f"Created content cache entry")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to create content: {e}")
            raise
    
    async def get_unused_content(self, content_type: str, limit: int = 10) -> List[ContentCache]:
        """Get unused content by type."""
        try:
            cursor = self.database.content_cache.find({
                "content_type": content_type,
                "used": False
            }).sort("generated_at", ASCENDING).limit(limit)
            
            content_list = []
            async for content_doc in cursor:
                content_list.append(ContentCache(**content_doc))
            return content_list
            
        except Exception as e:
            logger.error(f"Failed to get unused content: {e}")
            raise
    
    async def mark_content_used(self, content_id: str, performance_data: Optional[Dict] = None) -> bool:
        """Mark content as used and optionally update performance."""
        try:
            update_data = {
                "used": True,
                "used_at": datetime.utcnow()
            }
            
            if performance_data:
                for key, value in performance_data.items():
                    update_data[f"performance_metrics.{key}"] = value
            
            result = await self.database.content_cache.update_one(
                {"_id": content_id},
                {"$set": update_data}
            )
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"Failed to mark content used {content_id}: {e}")
            raise
    
    # Engagement history operations
    
    async def create_engagement(self, engagement: EngagementHistory) -> str:
        """Create engagement history record."""
        try:
            engagement_dict = engagement.dict(by_alias=True, exclude_unset=True)
            engagement_dict.pop("id", None)
            
            result = await self.database.engagement_history.insert_one(engagement_dict)
            logger.debug(f"Created engagement record for {engagement.action_type}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to create engagement: {e}")
            raise
    
    async def get_recent_engagements(
        self, 
        action_type: Optional[str] = None,
        hours: int = 24,
        limit: int = 100
    ) -> List[EngagementHistory]:
        """Get recent engagement history."""
        try:
            query = {"timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours)}}
            if action_type:
                query["action_type"] = action_type
            
            cursor = self.database.engagement_history.find(query).sort(
                "timestamp", DESCENDING
            ).limit(limit)
            
            engagements = []
            async for eng_doc in cursor:
                engagements.append(EngagementHistory(**eng_doc))
            return engagements
            
        except Exception as e:
            logger.error(f"Failed to get recent engagements: {e}")
            raise
    
    async def get_engagement_count(self, action_type: str, hours: int = 1) -> int:
        """Get count of specific engagement type within time period."""
        try:
            count = await self.database.engagement_history.count_documents({
                "action_type": action_type,
                "success": True,
                "timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours)}
            })
            return count
            
        except Exception as e:
            logger.error(f"Failed to get engagement count: {e}")
            raise
    
    # Bot metrics operations
    
    async def create_metric(self, metric: BotMetrics) -> str:
        """Create bot metric record."""
        try:
            metric_dict = metric.dict(by_alias=True, exclude_unset=True)
            metric_dict.pop("id", None)
            
            result = await self.database.bot_metrics.insert_one(metric_dict)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Failed to create metric: {e}")
            raise
    
    async def get_metrics(
        self,
        metric_type: Optional[str] = None,
        agent: Optional[str] = None,
        hours: int = 24,
        limit: int = 1000
    ) -> List[BotMetrics]:
        """Get bot metrics with optional filtering."""
        try:
            query = {"timestamp": {"$gte": datetime.utcnow() - timedelta(hours=hours)}}
            if metric_type:
                query["metric_type"] = metric_type
            if agent:
                query["agent"] = agent
            
            cursor = self.database.bot_metrics.find(query).sort(
                "timestamp", DESCENDING
            ).limit(limit)
            
            metrics = []
            async for metric_doc in cursor:
                metrics.append(BotMetrics(**metric_doc))
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise
    
    # Utility operations
    
    async def cleanup_old_data(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean up old data beyond retention period."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
            
            # Clean old engagement history
            eng_result = await self.database.engagement_history.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            # Clean old metrics
            metrics_result = await self.database.bot_metrics.delete_many({
                "timestamp": {"$lt": cutoff_date}
            })
            
            # Clean used content older than retention period
            content_result = await self.database.content_cache.delete_many({
                "used": True,
                "used_at": {"$lt": cutoff_date}
            })
            
            cleanup_stats = {
                "engagements_deleted": eng_result.deleted_count,
                "metrics_deleted": metrics_result.deleted_count,
                "content_deleted": content_result.deleted_count
            }
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            return cleanup_stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            raise
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Collection counts
            stats["users_count"] = await self.database.users.count_documents({})
            stats["content_count"] = await self.database.content_cache.count_documents({})
            stats["engagement_count"] = await self.database.engagement_history.count_documents({})
            stats["metrics_count"] = await self.database.bot_metrics.count_documents({})
            
            # Recent activity (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            stats["recent_engagements"] = await self.database.engagement_history.count_documents({
                "timestamp": {"$gte": recent_cutoff}
            })
            stats["recent_content"] = await self.database.content_cache.count_documents({
                "generated_at": {"$gte": recent_cutoff}
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise