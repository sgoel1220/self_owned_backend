"""Pydantic models for the application"""

from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

class DialogueLine(BaseModel):
    """Single line of dialogue"""
    character: str = Field(..., min_length=1, examples=["character1"])
    dialogue: str = Field(..., min_length=1, examples=["Did you hear that sound?"])

class ScriptRequest(BaseModel):
    """Request model for script generation"""
    story_input: str = Field(
        ...,
        min_length=10,
        description="Story text to convert",
        examples=["A mysterious event occurred in an abandoned mansion on a dark night..."]
    )
    save_to_db: bool = Field(
        default=True,
        description="Whether to save the story summary to database (script is NOT saved)"
    )
    character_a: Optional[str] = Field(
        default=None,
        description="Name for first character",
        examples=["character1"]
    )
    character_b: Optional[str] = Field(
        default=None,
        description="Name for second character",
        examples=["character2"]
    )

    @field_validator('story_input')
    @classmethod
    def validate_story_input(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('story_input cannot be empty or whitespace')
        return v.strip()

class StoryRequest(BaseModel):
    """Request model for saving stories"""
    article_summary: str = Field(
        ...,
        min_length=10,
        description="Story text to save",
        examples=["A chilling tale about mysterious disappearances..."]
    )

    @field_validator('article_summary')
    @classmethod
    def validate_article_summary(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('article_summary cannot be empty or whitespace')
        return v.strip()

class ScriptResponse(BaseModel):
    """Response model for script generation"""
    success: bool
    script: Optional[List[DialogueLine]] = None
    record_id: Optional[int] = None
    saved_to_db: Optional[bool] = None
    db_error: Optional[str] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    message: Optional[str] = None

class StoryResponse(BaseModel):
    """Response model for story saving"""
    success: bool
    message: Optional[str] = None
    record_id: Optional[int] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class BatchScriptResponse(BaseModel):
    """Response model for batch script generation"""
    success: bool
    message: str
    total_rows: int
    processed_count: int = 0
    failed_count: int = 0
    errors: Optional[List[Dict]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class BackgroundPromptBatchResponse(BaseModel):
    """Response model for batch background prompt generation"""
    success: bool
    message: str
    total_rows: int
    processed_count: int = 0
    failed_count: int = 0
    errors: Optional[List[Dict]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class BackgroundImageBatchResponse(BaseModel):
    """Response model for batch background image generation"""
    success: bool
    message: str
    total_rows: int
    processed_count: int = 0
    failed_count: int = 0
    errors: Optional[List[Dict]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class VideoGenerationBatchResponse(BaseModel):
    """Response model for batch video generation"""
    success: bool
    message: str
    total_rows: int
    processed_count: int = 0
    failed_count: int = 0
    errors: Optional[List[Dict]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

from enum import Enum

class ProductionStatus(str, Enum):
    PENDING = "pending"
    SCRIPT_GENERATED = "script_generated"
    PROMPT_GENERATED = "prompt_generated"
    IMAGE_GENERATED = "image_generated"
    VIDEO_GENERATED = "video_generated"
    COMPLETE = "complete"
    FAILED = "failed"

class ProductionSummary(BaseModel):
    id: int
    article_summary: str
    status: str
    created_at: datetime
    video_generated_url: Optional[str] = None

class ProductionDetails(ProductionSummary):
    ai_script: Optional[List[DialogueLine]] = None
    background_image_prompt: Optional[str] = None
    background_image_url: Optional[str] = None
    character_a: Optional[str] = None
    character_b: Optional[str] = None
    background_image_1: Optional[str] = None
    incident_image_1: Optional[str] = None
    incident_image_2: Optional[str] = None
    incident_image_3: Optional[str] = None
    background_audio_url: Optional[str] = None

class ProductionRequest(BaseModel):
    story: str = Field(..., min_length=10, description="The story to generate a video from.")

class RetryStep(str, Enum):
    SCRIPT = "script"
    BACKGROUND_PROMPT = "background_prompt"
    BACKGROUND_IMAGE = "background_image"
    VIDEO = "video"

class RetryResponse(BaseModel):
    success: bool
    message: str

class ProductionUpdateRequest(BaseModel):
    article_summary: Optional[str] = None
    ai_script: Optional[List[DialogueLine]] = None
    background_image_prompt: Optional[str] = None
    background_image_url: Optional[str] = None
    video_generated_url: Optional[str] = None
    character_a: Optional[str] = None
    character_b: Optional[str] = None
    incident_image_1: Optional[str] = None
    incident_image_2: Optional[str] = None
    incident_image_3: Optional[str] = None
    background_audio_url: Optional[str] = None
    background_image_1: Optional[str] = None
