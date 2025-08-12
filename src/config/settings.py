"""Configuration settings for the Twitter bot system."""

import os
from typing import List, Optional
from pydantic import BaseSettings, Field
from dotenv import load_dotenv

load_dotenv()


class TwitterConfig(BaseSettings):
    """Twitter API configuration."""
    
    api_key: str = Field(..., env="TWITTER_API_KEY")
    api_secret: str = Field(..., env="TWITTER_API_SECRET")
    access_token: str = Field(..., env="TWITTER_ACCESS_TOKEN")
    access_token_secret: str = Field(..., env="TWITTER_ACCESS_TOKEN_SECRET")
    bearer_token: str = Field(..., env="TWITTER_BEARER_TOKEN")
    bot_username: str = Field(..., env="BOT_USERNAME")


class LLMConfig(BaseSettings):
    """LLM API configuration."""
    
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    preferred_model: str = Field("gpt-4", env="PREFERRED_LLM_MODEL")
    max_tokens: int = Field(1000, env="LLM_MAX_TOKENS")
    temperature: float = Field(0.7, env="LLM_TEMPERATURE")


class DatabaseConfig(BaseSettings):
    """Database configuration."""
    
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    postgres_uri: str = Field(..., env="POSTGRES_URI")
    redis_uri: Optional[str] = Field(None, env="REDIS_URI")
    
    # Connection pool settings
    mongodb_max_pool_size: int = Field(10, env="MONGODB_MAX_POOL_SIZE")
    postgres_max_pool_size: int = Field(10, env="POSTGRES_MAX_POOL_SIZE")


class RateLimitConfig(BaseSettings):
    """Rate limiting configuration."""
    
    # Twitter API rate limits
    max_tweets_per_15min: int = Field(300, env="MAX_TWEETS_PER_15MIN")
    max_likes_per_15min: int = Field(300, env="MAX_LIKES_PER_15MIN")
    max_follows_per_day: int = Field(400, env="MAX_FOLLOWS_PER_DAY")
    max_reposts_per_15min: int = Field(300, env="MAX_REPOSTS_PER_15MIN")
    
    # Bot-specific limits (more conservative)
    max_follows_per_day_bot: int = Field(50, env="MAX_FOLLOWS_PER_DAY")
    max_posts_per_day: int = Field(10, env="MAX_POSTS_PER_DAY")
    max_likes_per_hour: int = Field(30, env="MAX_LIKES_PER_HOUR")
    max_reposts_per_hour: int = Field(15, env="MAX_REPOSTS_PER_HOUR")
    max_comments_per_hour: int = Field(10, env="MAX_COMMENTS_PER_HOUR")
    
    # Timing intervals (in seconds)
    min_action_interval: int = Field(30, env="MIN_ACTION_INTERVAL")
    max_action_interval: int = Field(300, env="MAX_ACTION_INTERVAL")
    
    # Backoff settings
    initial_backoff: int = Field(60, env="INITIAL_BACKOFF")
    max_backoff: int = Field(3600, env="MAX_BACKOFF")
    backoff_multiplier: float = Field(2.0, env="BACKOFF_MULTIPLIER")


class AgentConfig(BaseSettings):
    """Agent behavior configuration."""
    
    # Decision thresholds
    follow_threshold: float = Field(0.7, env="FOLLOW_THRESHOLD")
    like_threshold: float = Field(0.6, env="LIKE_THRESHOLD")
    repost_threshold: float = Field(0.8, env="REPOST_THRESHOLD")
    comment_threshold: float = Field(0.9, env="COMMENT_THRESHOLD")
    
    # Content settings
    content_domains: List[str] = Field(
        ["music", "nola_culture", "genz_trends"], 
        env="CONTENT_DOMAINS"
    )
    min_content_cache_size: int = Field(20, env="MIN_CONTENT_CACHE_SIZE")
    max_content_cache_size: int = Field(100, env="MAX_CONTENT_CACHE_SIZE")
    
    # Target settings
    target_location: str = Field("New Orleans", env="TARGET_LOCATION")
    target_keywords: List[str] = Field(
        ["music", "jazz", "bounce", "krewe", "nola", "new orleans"],
        env="TARGET_KEYWORDS"
    )
    
    # User scoring weights
    location_weight: float = Field(0.3, env="LOCATION_WEIGHT")
    activity_weight: float = Field(0.2, env="ACTIVITY_WEIGHT")
    engagement_weight: float = Field(0.2, env="ENGAGEMENT_WEIGHT")
    relevance_weight: float = Field(0.3, env="RELEVANCE_WEIGHT")


class MonitoringConfig(BaseSettings):
    """Monitoring and logging configuration."""
    
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field("logs/twitter_bot.log", env="LOG_FILE")
    metrics_port: int = Field(8000, env="METRICS_PORT")
    
    # Prometheus settings
    prometheus_enabled: bool = Field(True, env="PROMETHEUS_ENABLED")
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    
    # Health check settings
    health_check_interval: int = Field(300, env="HEALTH_CHECK_INTERVAL")
    
    # Alert settings
    alert_on_rate_limit: bool = Field(True, env="ALERT_ON_RATE_LIMIT")
    alert_on_error_threshold: int = Field(10, env="ALERT_ON_ERROR_THRESHOLD")


class DevelopmentConfig(BaseSettings):
    """Development and debugging configuration."""
    
    debug: bool = Field(False, env="DEBUG")
    development_mode: bool = Field(False, env="DEVELOPMENT_MODE")
    dry_run: bool = Field(False, env="DRY_RUN")
    
    # Testing settings
    mock_twitter_api: bool = Field(False, env="MOCK_TWITTER_API")
    mock_llm_api: bool = Field(False, env="MOCK_LLM_API")
    
    # Logging settings for dev
    verbose_logging: bool = Field(False, env="VERBOSE_LOGGING")
    log_api_requests: bool = Field(False, env="LOG_API_REQUESTS")


class BotConfig(BaseSettings):
    """Main configuration class that combines all sub-configurations."""
    
    twitter: TwitterConfig = TwitterConfig()
    llm: LLMConfig = LLMConfig()
    database: DatabaseConfig = DatabaseConfig()
    rate_limit: RateLimitConfig = RateLimitConfig()
    agents: AgentConfig = AgentConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    development: DevelopmentConfig = DevelopmentConfig()
    
    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def validate_config(self) -> None:
        """Validate configuration settings."""
        # Check required API keys
        if not self.twitter.api_key or not self.twitter.api_secret:
            raise ValueError("Twitter API credentials are required")
        
        if not self.llm.openai_api_key and not self.llm.anthropic_api_key:
            raise ValueError("At least one LLM API key is required")
        
        # Validate thresholds
        for threshold in [
            self.agents.follow_threshold,
            self.agents.like_threshold,
            self.agents.repost_threshold,
            self.agents.comment_threshold,
        ]:
            if not 0 <= threshold <= 1:
                raise ValueError("All thresholds must be between 0 and 1")
        
        # Validate rate limits
        if self.rate_limit.max_follows_per_day_bot > self.rate_limit.max_follows_per_day:
            raise ValueError("Bot follow limit cannot exceed Twitter API limit")
    
    def get_content_generation_prompt(self, content_type: str = "post") -> str:
        """Get domain-specific prompt for content generation."""
        base_context = (
            f"You are an AI assistant creating {content_type}s for a Twitter bot "
            f"focused on {', '.join(self.agents.content_domains)} in {self.agents.target_location}. "
            "Create engaging, authentic content that resonates with GenZ audiences "
            "interested in music and local culture."
        )
        
        if content_type == "post":
            return (
                f"{base_context} "
                "Create an original tweet that's conversational, relevant, and engaging. "
                "Keep it under 280 characters. No hashtags unless naturally relevant. "
                "Focus on local music events, cultural observations, or GenZ trends."
            )
        elif content_type == "comment":
            return (
                f"{base_context} "
                "Create a thoughtful reply that adds value to the conversation. "
                "Be supportive, engaging, and authentic. Keep it conversational and brief."
            )
        else:
            return base_context