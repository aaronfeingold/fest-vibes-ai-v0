"""Data models for the Twitter bot system."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator
from bson import ObjectId


class PyObjectId(ObjectId):
    """Custom ObjectId for Pydantic models."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema
        return core_schema.with_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def _validate(cls, v, info):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str):
            if ObjectId.is_valid(v):
                return ObjectId(v)
        raise ValueError("Invalid ObjectId")

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema, handler):
        field_schema.update(type="string")
        return field_schema


class FollowStatus(str, Enum):
    """User follow status enumeration."""
    FOLLOWING = "following"
    NOT_FOLLOWING = "not_following"
    BLOCKED = "blocked"
    PENDING = "pending"


class ContentType(str, Enum):
    """Content type enumeration."""
    POST = "post"
    COMMENT = "comment"
    REPOST = "repost"


class ActionType(str, Enum):
    """Engagement action type enumeration."""
    LIKE = "like"
    REPOST = "repost"
    COMMENT = "comment"
    FOLLOW = "follow"
    UNFOLLOW = "unfollow"


class User(BaseModel):
    """User model for MongoDB storage."""

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    twitter_user_id: str = Field(..., description="Twitter user ID")
    username: str = Field(..., description="Twitter username")
    display_name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    follower_count: int = 0
    following_count: int = 0
    tweet_count: int = 0
    last_activity: Optional[datetime] = None
    engagement_score: float = Field(0.0, ge=0.0, le=1.0)
    follow_status: FollowStatus = FollowStatus.NOT_FOLLOWING
    discovered_date: datetime = Field(default_factory=datetime.utcnow)
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)

    # Analytics fields
    mutual_followers: int = 0
    interaction_count: int = 0
    last_interaction: Optional[datetime] = None
    relevance_score: float = 0.0

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class PerformanceMetrics(BaseModel):
    """Performance metrics for content."""

    likes: int = 0
    retweets: int = 0
    replies: int = 0
    impressions: Optional[int] = None
    engagement_rate: float = 0.0


class ContentCache(BaseModel):
    """Content cache model for MongoDB storage."""

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    content_type: ContentType = Field(..., description="Type of content")
    content: str = Field(..., description="Generated content text")
    prompt_used: Optional[str] = None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    used: bool = False
    used_at: Optional[datetime] = None
    performance_metrics: PerformanceMetrics = Field(default_factory=PerformanceMetrics)
    tags: List[str] = Field(default_factory=list)

    # Content metadata
    word_count: int = 0
    character_count: int = 0
    sentiment_score: Optional[float] = None

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class EngagementHistory(BaseModel):
    """Engagement history model for MongoDB storage."""

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    target_tweet_id: str = Field(..., description="Target tweet ID")
    target_user_id: str = Field(..., description="Target user ID")
    action_type: ActionType = Field(..., description="Type of engagement")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = Field(..., description="Whether action was successful")
    response_data: Dict[str, Any] = Field(default_factory=dict)

    # Context information
    content_used: Optional[str] = None
    decision_score: Optional[float] = None
    decision_factors: Dict[str, float] = Field(default_factory=dict)

    # Error information
    error_message: Optional[str] = None
    retry_count: int = 0

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class BotMetrics(BaseModel):
    """Bot metrics model for MongoDB storage."""

    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    metric_type: str = Field(..., description="Type of metric")
    agent: str = Field(..., description="Agent that generated the metric")
    value: float = Field(..., description="Metric value")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# PostgreSQL Models (using Pydantic for validation)

class ContentEmbedding(BaseModel):
    """Content embedding model for PostgreSQL storage."""

    id: Optional[int] = None
    content_hash: str = Field(..., description="SHA-256 hash of content")
    content: str = Field(..., description="Original content text")
    embedding: List[float] = Field(..., description="Vector embedding")
    content_type: str = Field(..., description="Type of content")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class UserInterestVector(BaseModel):
    """User interest vector model for PostgreSQL storage."""

    twitter_user_id: str = Field(..., description="Twitter user ID")
    interest_embedding: List[float] = Field(..., description="Interest vector embedding")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    interaction_count: int = 0
    engagement_score: float = 0.0


class TweetEmbedding(BaseModel):
    """Tweet embedding model for PostgreSQL storage."""

    id: Optional[int] = None
    tweet_id: str = Field(..., description="Twitter tweet ID")
    tweet_text: str = Field(..., description="Tweet text content")
    embedding: List[float] = Field(..., description="Vector embedding")
    author_id: str = Field(..., description="Tweet author ID")
    engagement_metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SemanticSearchResult(BaseModel):
    """Semantic search result model."""

    content_id: int
    content: str
    similarity: float
    metadata: Optional[Dict[str, Any]] = None


class UserSearchResult(BaseModel):
    """User search result model."""

    user_id: str
    similarity: float
    engagement_score: float
    metadata: Optional[Dict[str, Any]] = None


# Request/Response Models

class FollowDecision(BaseModel):
    """Follow decision model."""

    user_id: str
    score: float
    decision: bool
    factors: Dict[str, float]
    confidence: float
    
    @field_validator('user_id', mode='before')
    @classmethod
    def convert_user_id_to_string(cls, v):
        """Convert user_id to string if it's an integer."""
        return str(v)


class ContentGenerationRequest(BaseModel):
    """Content generation request model."""

    content_type: ContentType
    context: Optional[str] = None
    target_audience: Optional[str] = None
    tone: Optional[str] = None
    max_length: int = 280


class EngagementDecision(BaseModel):
    """Engagement decision model."""

    tweet_id: str
    action_type: ActionType
    score: float
    decision: bool
    factors: Dict[str, float]
    confidence: float
    generated_content: Optional[str] = None
    
    @field_validator('tweet_id', mode='before')
    @classmethod
    def convert_tweet_id_to_string(cls, v):
        """Convert tweet_id to string if it's an integer."""
        return str(v)


class UserAnalysis(BaseModel):
    """User analysis result model."""

    user_id: str
    location_score: float
    activity_score: float
    engagement_score: float
    relevance_score: float
    overall_score: float
    recommendation: str
    factors: Dict[str, Any]
