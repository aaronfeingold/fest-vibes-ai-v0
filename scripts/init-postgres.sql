-- PostgreSQL initialization script with pgvector extension

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Content embeddings for semantic analysis
CREATE TABLE IF NOT EXISTS content_embeddings (
    id SERIAL PRIMARY KEY,
    content_hash VARCHAR(64) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- OpenAI embedding dimension
    content_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- User interest vectors
CREATE TABLE IF NOT EXISTS user_interest_vectors (
    twitter_user_id VARCHAR(50) PRIMARY KEY,
    interest_embedding VECTOR(1536) NOT NULL,
    last_updated TIMESTAMP DEFAULT NOW(),
    interaction_count INTEGER DEFAULT 0,
    engagement_score FLOAT DEFAULT 0.0
);

-- Semantic search results cache
CREATE TABLE IF NOT EXISTS semantic_search_cache (
    id SERIAL PRIMARY KEY,
    query_hash VARCHAR(64) NOT NULL,
    query_text TEXT NOT NULL,
    query_embedding VECTOR(1536) NOT NULL,
    result_content_ids INTEGER[] NOT NULL,
    search_type VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '24 hours')
);

-- Tweet embeddings for content analysis
CREATE TABLE IF NOT EXISTS tweet_embeddings (
    id SERIAL PRIMARY KEY,
    tweet_id VARCHAR(50) UNIQUE NOT NULL,
    tweet_text TEXT NOT NULL,
    embedding VECTOR(1536) NOT NULL,
    author_id VARCHAR(50) NOT NULL,
    engagement_metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for optimal performance
CREATE INDEX IF NOT EXISTS idx_content_embeddings_type ON content_embeddings(content_type);
CREATE INDEX IF NOT EXISTS idx_content_embeddings_created ON content_embeddings(created_at);
CREATE INDEX IF NOT EXISTS idx_content_embeddings_hash ON content_embeddings(content_hash);

CREATE INDEX IF NOT EXISTS idx_user_interest_updated ON user_interest_vectors(last_updated);
CREATE INDEX IF NOT EXISTS idx_user_interest_score ON user_interest_vectors(engagement_score);

CREATE INDEX IF NOT EXISTS idx_search_cache_expires ON semantic_search_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_search_cache_type ON semantic_search_cache(search_type);
CREATE INDEX IF NOT EXISTS idx_search_cache_hash ON semantic_search_cache(query_hash);

CREATE INDEX IF NOT EXISTS idx_tweet_embeddings_author ON tweet_embeddings(author_id);
CREATE INDEX IF NOT EXISTS idx_tweet_embeddings_created ON tweet_embeddings(created_at);

-- Create function to clean expired cache
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM semantic_search_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Vector similarity search functions
CREATE OR REPLACE FUNCTION find_similar_content(
    query_embedding VECTOR(1536),
    content_type_filter VARCHAR(50) DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.8,
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE(
    content_id INTEGER,
    content TEXT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ce.id,
        ce.content,
        (ce.embedding <=> query_embedding)::FLOAT as similarity
    FROM content_embeddings ce
    WHERE 
        (content_type_filter IS NULL OR ce.content_type = content_type_filter)
        AND (ce.embedding <=> query_embedding) <= (1 - similarity_threshold)
    ORDER BY ce.embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION find_similar_users(
    query_embedding VECTOR(1536),
    similarity_threshold FLOAT DEFAULT 0.8,
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE(
    user_id VARCHAR(50),
    similarity FLOAT,
    engagement_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        uiv.twitter_user_id,
        (uiv.interest_embedding <=> query_embedding)::FLOAT as similarity,
        uiv.engagement_score
    FROM user_interest_vectors uiv
    WHERE (uiv.interest_embedding <=> query_embedding) <= (1 - similarity_threshold)
    ORDER BY uiv.interest_embedding <=> query_embedding
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON DATABASE twitter_bot IS 'Autonomous Twitter Bot Database with Vector Embeddings';
COMMENT ON TABLE content_embeddings IS 'Stores content embeddings for semantic search and analysis';
COMMENT ON TABLE user_interest_vectors IS 'User interest profiles as vector embeddings';
COMMENT ON TABLE semantic_search_cache IS 'Cache for frequent semantic search queries';
COMMENT ON TABLE tweet_embeddings IS 'Tweet embeddings for content similarity analysis';

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'PostgreSQL with pgvector initialization completed successfully';
END $$;