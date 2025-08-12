// MongoDB initialization script
db = db.getSiblingDB('twitter_bot');

// Create collections with proper indexes
db.createCollection('users');
db.createCollection('content_cache');
db.createCollection('engagement_history');
db.createCollection('bot_metrics');

// Create indexes for users collection
db.users.createIndex({ "twitter_user_id": 1 }, { unique: true });
db.users.createIndex({ "username": 1 });
db.users.createIndex({ "location": 1 });
db.users.createIndex({ "follow_status": 1 });
db.users.createIndex({ "engagement_score": -1 });
db.users.createIndex({ "last_activity": -1 });
db.users.createIndex({ "discovered_date": -1 });
db.users.createIndex({ "tags": 1 });

// Create indexes for content_cache collection
db.content_cache.createIndex({ "content_type": 1 });
db.content_cache.createIndex({ "generated_at": -1 });
db.content_cache.createIndex({ "used": 1 });
db.content_cache.createIndex({ "tags": 1 });
db.content_cache.createIndex({ "performance_metrics.likes": -1 });

// Create indexes for engagement_history collection
db.engagement_history.createIndex({ "target_tweet_id": 1 });
db.engagement_history.createIndex({ "action_type": 1 });
db.engagement_history.createIndex({ "timestamp": -1 });
db.engagement_history.createIndex({ "success": 1 });

// Create indexes for bot_metrics collection
db.bot_metrics.createIndex({ "metric_type": 1 });
db.bot_metrics.createIndex({ "timestamp": -1 });
db.bot_metrics.createIndex({ "agent": 1 });

print('MongoDB initialization completed successfully');