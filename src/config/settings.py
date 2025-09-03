"""Configuration settings for the Twitter bot system."""

import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Get the project root directory
project_root = Path(__file__).parent.parent.parent
env_file = project_root / ".env"

# Load environment variables
load_dotenv(env_file)


class TwitterConfig(BaseSettings):
    """Twitter API configuration."""
    model_config = SettingsConfigDict(
        env_file=str(env_file), env_file_encoding="utf-8", extra="ignore"
    )

    api_key: str = Field(
        default_factory=lambda: os.getenv("TWITTER_API_KEY", ""), env="TWITTER_API_KEY"
    )
    api_secret: str = Field(
        default_factory=lambda: os.getenv("TWITTER_API_SECRET", ""),
        env="TWITTER_API_SECRET",
    )
    access_token: str = Field(
        default_factory=lambda: os.getenv("TWITTER_ACCESS_TOKEN", ""),
        env="TWITTER_ACCESS_TOKEN",
    )
    access_token_secret: str = Field(
        default_factory=lambda: os.getenv("TWITTER_ACCESS_TOKEN_SECRET", ""),
        env="TWITTER_ACCESS_TOKEN_SECRET",
    )
    bearer_token: str = Field(
        default_factory=lambda: os.getenv("TWITTER_BEARER_TOKEN", ""),
        env="TWITTER_BEARER_TOKEN",
    )
    bot_username: str = Field(..., env="BOT_USERNAME")


class LLMConfig(BaseSettings):
    """LLM API configuration."""

    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, env="ANTHROPIC_API_KEY")
    preferred_model: str = Field("gpt-4o-mini", env="PREFERRED_LLM_MODEL")
    max_tokens: int = Field(150, env="LLM_MAX_TOKENS")
    temperature: float = Field(0.7, env="LLM_TEMPERATURE")

    # Embedding configuration (using sentence-transformers all-MiniLM-L6-v2)
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(384, env="EMBEDDING_DIMENSIONS")
    embedding_encoding_format: str = Field("float", env="EMBEDDING_ENCODING_FORMAT")

    model_config = SettingsConfigDict(
        env_file=str(env_file), env_file_encoding="utf-8", extra="ignore"
    )


class DatabaseConfig(BaseSettings):
    """Database configuration."""

    mongodb_uri: str = Field(..., env="MONGODB_URI")
    postgres_uri: str = Field(..., env="POSTGRES_URI")
    redis_uri: Optional[str] = Field(None, env="REDIS_URI")

    # Event source database (for RAG queries) - separate from bot operations
    events_postgres_uri: str = Field(..., env="EVENTS_POSTGRES_URI")

    # Connection pool settings
    mongodb_max_pool_size: int = Field(10, env="MONGODB_MAX_POOL_SIZE")
    postgres_max_pool_size: int = Field(10, env="POSTGRES_MAX_POOL_SIZE")
    events_postgres_max_pool_size: int = Field(5, env="EVENTS_POSTGRES_MAX_POOL_SIZE")

    # PostgreSQL-specific settings for cloud databases (Neon)
    postgres_ssl_mode: str = Field("prefer", env="POSTGRES_SSL_MODE")
    postgres_connect_timeout: int = Field(30, env="POSTGRES_CONNECT_TIMEOUT")
    use_local_postgres: bool = Field(True, env="USE_LOCAL_POSTGRES")

    # Event database settings (may also be Neon or different provider)
    events_postgres_ssl_mode: str = Field("prefer", env="EVENTS_POSTGRES_SSL_MODE")
    events_postgres_connect_timeout: int = Field(
        30, env="EVENTS_POSTGRES_CONNECT_TIMEOUT"
    )
    use_local_events_postgres: bool = Field(True, env="USE_LOCAL_EVENTS_POSTGRES")

    model_config = SettingsConfigDict(
        env_file=str(env_file), env_file_encoding="utf-8", extra="ignore"
    )

    def get_postgres_connection_kwargs(self) -> dict:
        """Get PostgreSQL connection kwargs for bot operations database."""
        kwargs = {
            "command_timeout": self.postgres_connect_timeout,
            "server_settings": {"application_name": "fest_vibes_twitter_bot"},
        }

        # Add SSL settings for cloud databases
        if not self.use_local_postgres or "neon.tech" in self.postgres_uri:
            kwargs["ssl"] = self.postgres_ssl_mode

        return kwargs

    def get_events_postgres_connection_kwargs(self) -> dict:
        """Get PostgreSQL connection kwargs for events source database."""
        kwargs = {
            "command_timeout": self.events_postgres_connect_timeout,
            "server_settings": {"application_name": "fest_vibes_rag_client"},
        }

        # Add SSL settings for cloud databases
        if (
            not self.use_local_events_postgres
            or "neon.tech" in self.events_postgres_uri
        ):
            kwargs["ssl"] = self.events_postgres_ssl_mode

        return kwargs


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

    model_config = SettingsConfigDict(
        env_file=str(env_file), env_file_encoding="utf-8", extra="ignore"
    )


class AgentConfig(BaseSettings):
    """Agent behavior configuration."""

    # Decision thresholds
    follow_threshold: float = Field(0.7, env="FOLLOW_THRESHOLD")
    like_threshold: float = Field(0.6, env="LIKE_THRESHOLD")
    repost_threshold: float = Field(0.8, env="REPOST_THRESHOLD")
    comment_threshold: float = Field(0.9, env="COMMENT_THRESHOLD")

    # Content settings
    content_domains: str = Field(
        "music,nola_culture,genz_trends", env="CONTENT_DOMAINS"
    )
    min_content_cache_size: int = Field(5, env="MIN_CONTENT_CACHE_SIZE")
    max_content_cache_size: int = Field(10, env="MAX_CONTENT_CACHE_SIZE")

    # Target settings
    target_location: str = Field("New Orleans", env="TARGET_LOCATION")
    target_keywords: str = Field(
        "music,jazz,bounce,krewe,nola,new orleans", env="TARGET_KEYWORDS"
    )

    # User scoring weights
    location_weight: float = Field(0.3, env="LOCATION_WEIGHT")
    activity_weight: float = Field(0.2, env="ACTIVITY_WEIGHT")
    engagement_weight: float = Field(0.2, env="ENGAGEMENT_WEIGHT")
    relevance_weight: float = Field(0.3, env="RELEVANCE_WEIGHT")

    # RAG settings
    rag_similarity_threshold: float = Field(0.6, env="RAG_SIMILARITY_THRESHOLD")
    rag_max_events_per_search: int = Field(12, env="RAG_MAX_EVENTS_PER_SEARCH")
    rag_max_venues_per_schedule: int = Field(4, env="RAG_MAX_VENUES_PER_SCHEDULE")
    rag_search_days_ahead: int = Field(3, env="RAG_SEARCH_DAYS_AHEAD")
    rag_enable_route_optimization: bool = Field(
        True, env="RAG_ENABLE_ROUTE_OPTIMIZATION"
    )

    model_config = SettingsConfigDict(
        env_file=str(env_file), env_file_encoding="utf-8", extra="ignore"
    )


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

    model_config = SettingsConfigDict(
        env_file=str(env_file), env_file_encoding="utf-8", extra="ignore"
    )


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

    model_config = SettingsConfigDict(
        env_file=str(env_file), env_file_encoding="utf-8", extra="ignore"
    )


class BotConfig(BaseSettings):
    """Main configuration class that combines all sub-configurations."""

    twitter: TwitterConfig = Field(default_factory=TwitterConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)

    model_config = SettingsConfigDict(
        env_file=str(env_file),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

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
