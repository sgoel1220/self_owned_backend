"""
AI Content Automation - Complete Video Generation System
FastAPI server with Supabase integration for viral horror/mystery content creation
Includes script generation, image generation, and video generation with storage upload
Version: 4.0.0
"""

import os
import logging
import tempfile
import gc
import re
import io
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import requests
import json
import base64
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageEnhance
from supabase import create_client, Client
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config(BaseSettings):
    """Application configuration with Pydantic"""
    FLASK_HOST: str = '0.0.0.0'
    FLASK_PORT: int = 5000
    FLASK_DEBUG: bool = False
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_TABLE: str = 'automation'
    SUPABASE_STORAGE_BUCKET: str = 'youtube-automation'
    OPENROUTER_API_KEY: str
    OPENROUTER_BASE_URL: str = 'https://openrouter.ai/api/v1/chat/completions'
    
    # Hugging Face configuration for image generation
    HUGGINGFACE_API_KEY: str
    HUGGINGFACE_IMAGE_URL: str = 'https://router.huggingface.co/nebius/v1/images/generations'
    HUGGINGFACE_MODEL: str = 'black-forest-labs/flux-schnell'
    
    PRIMARY_MODEL: str = 'google/gemini-2.0-flash-exp:free'
    FALLBACK_MODEL: str = 'openai/gpt-3.5-turbo'
    BATCH_SIZE: int = 5
    MAX_RETRIES: int = 3
    TIMEOUT_SECONDS: int = 60
    
    # Video generation defaults
    DEFAULT_FONT_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/font/Inter-VariableFont.ttf"
    DEFAULT_MASK_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/mask/mask.jpg"
    DEFAULT_CHARACTER_1_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/characters/Tradgirl.png"
    DEFAULT_CHARACTER_2_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/characters/Chad.png"
    DEFAULT_AUDIO_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/audio/suspense_dark.mp3"
    
    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=True
    )

config = Config()

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
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

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================
class StoryInsertionError(Exception):
    """Custom exception for story insertion failures"""
    pass

class ImageGenerationError(Exception):
    """Custom exception for image generation failures"""
    pass

class StorageUploadError(Exception):
    """Custom exception for storage upload failures"""
    pass

class ResourceDownloadError(Exception):
    """Exception raised when resource download fails"""
    pass

class VideoGenerationError(Exception):
    """Exception raised when video generation fails"""
    pass

class AudioProcessingError(Exception):
    """Exception raised when audio processing fails"""
    pass

# ============================================================================
# DATABASE MANAGER
# ============================================================================
class SupabaseManager:
    """Manages Supabase database operations"""
    
    def __init__(self):
        self.client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
        self.table = config.SUPABASE_TABLE
        self.storage_bucket = config.SUPABASE_STORAGE_BUCKET
    
    def save_story_only(
        self,
        article_summary: str,
    ) -> dict:
        """
        Save ONLY the story summary to database (no script)
        
        Args:
            article_summary: The story text to save
        
        Returns:
            dict: The inserted record from database
        
        Raises:
            StoryInsertionError: If insertion fails
        """
        try:
            insert_data = {
                "article_summary": article_summary,
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = self.client.table(self.table).insert(insert_data).execute()
            
            if not response.data:
                raise StoryInsertionError("Insert operation returned no data")
            
            record = response.data[0]
            logger.info(f"Successfully saved story summary with ID: {record.get('id')}")
            return record
            
        except Exception as e:
            error_msg = f"Failed to insert story: {str(e)}"
            logger.error(error_msg)
            raise StoryInsertionError(error_msg)
    
    def get_empty_script_rows(self, limit: Optional[int] = None) -> List[dict]:
        """
        Fetch all rows where ai_script is null or empty
        
        Args:
            limit: Optional limit on number of rows to fetch
        
        Returns:
            List of records with empty ai_script
        """
        try:
            query = self.client.table(self.table).select("*").is_("ai_script", None)
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            logger.info(f"Found {len(response.data)} rows with empty ai_script")
            return response.data
            
        except Exception as e:
            logger.error(f"Failed to fetch empty script rows: {e}")
            raise
    
    def update_script(self, record_id: int, script_data: List[DialogueLine]) -> bool:
        """
        Update a specific row with generated script
        
        Args:
            record_id: The ID of the record to update
            script_data: The generated script dialogue lines
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            script_dict = [line.model_dump() for line in script_data]
            
            response = (
                self.client.table(self.table)
                .update({
                    "ai_script": script_dict
                })
                .eq("id", record_id)
                .execute()
            )
            
            if response.data:
                logger.info(f"Successfully updated script for record ID: {record_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update script for record {record_id}: {e}")
            return False
    
    def get_rows_with_null_background_prompt(self, limit: Optional[int] = None) -> List[dict]:
        """
        Fetch all rows where background_image_prompt is null
        
        Args:
            limit: Optional limit on number of rows to fetch
        
        Returns:
            List of records with null background_image_prompt
        """
        try:
            query = self.client.table(self.table).select("*").is_("background_image_prompt", None)
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            logger.info(f"Found {len(response.data)} rows with null background_image_prompt")
            return response.data
            
        except Exception as e:
            logger.error(f"Failed to fetch rows with null background_image_prompt: {e}")
            raise
    
    def update_background_prompt(self, record_id: int, background_prompt: str) -> bool:
        """
        Update a specific row with generated background image prompt
        
        Args:
            record_id: The ID of the record to update
            background_prompt: The generated background image prompt
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = (
                self.client.table(self.table)
                .update({
                    "background_image_prompt": background_prompt
                })
                .eq("id", record_id)
                .execute()
            )
            
            if response.data:
                logger.info(f"Successfully updated background_image_prompt for record ID: {record_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update background_image_prompt for record {record_id}: {e}")
            return False
    
    def get_rows_with_null_background_image(self, limit: Optional[int] = None) -> List[dict]:
        """
        Fetch all rows where background_image_prompt exists but background_image_url is null
        
        Args:
            limit: Optional limit on number of rows to fetch
        
        Returns:
            List of records ready for image generation
        """
        try:
            query = (
                self.client.table(self.table)
                .select("*")
                .not_.is_("background_image_prompt", None)
                .is_("background_image_url", None)
            )
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            logger.info(f"Found {len(response.data)} rows ready for image generation")
            return response.data
            
        except Exception as e:
            logger.error(f"Failed to fetch rows for image generation: {e}")
            raise
    
    def update_background_image_url(self, record_id: int, image_url: str) -> bool:
        """
        Update a specific row with generated background image URL
        
        Args:
            record_id: The ID of the record to update
            image_url: The public URL of the uploaded image
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = (
                self.client.table(self.table)
                .update({
                    "background_image_url": image_url
                })
                .eq("id", record_id)
                .execute()
            )
            
            if response.data:
                logger.info(f"Successfully updated background_image_url for record ID: {record_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update background_image_url for record {record_id}: {e}")
            return False
    
    def get_rows_ready_for_video(self, limit: Optional[int] = None) -> List[dict]:
        """
        Fetch all rows where required fields for video generation exist but video_generated_url is null
        Required fields: ai_script, background_image_url
        
        Args:
            limit: Optional limit on number of rows to fetch
        
        Returns:
            List of records ready for video generation
        """
        try:
            query = (
                self.client.table(self.table)
                .select("*")
                .not_.is_("ai_script", None)
                .not_.is_("background_image_url", None)
                .is_("video_generated_url", None)
            )
            
            if limit:
                query = query.limit(limit)
            
            response = query.execute()
            logger.info(f"Found {len(response.data)} rows ready for video generation")
            return response.data
            
        except Exception as e:
            logger.error(f"Failed to fetch rows for video generation: {e}")
            raise
    
    def update_video_url(self, record_id: int, video_url: str) -> bool:
        """
        Update a specific row with generated video URL
        
        Args:
            record_id: The ID of the record to update
            video_url: The public URL of the uploaded video
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = (
                self.client.table(self.table)
                .update({
                    "video_generated_url": video_url
                })
                .eq("id", record_id)
                .execute()
            )
            
            if response.data:
                logger.info(f"Successfully updated video_generated_url for record ID: {record_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to update video_generated_url for record {record_id}: {e}")
            return False
    
    def upload_image_to_storage(self, image_data: bytes, filename: str) -> str:
        """
        Upload image to Supabase Storage
        
        Args:
            image_data: Binary image data
            filename: Name for the file in storage
        
        Returns:
            str: Public URL of the uploaded image
        
        Raises:
            StorageUploadError: If upload fails
        """
        try:
            # Upload to Supabase Storage
            response = self.client.storage.from_(self.storage_bucket).upload(
                filename,
                image_data,
                {
                    'content-type': 'image/png',
                    'upsert': 'false'
                }
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.storage_bucket).get_public_url(filename)
            
            logger.info(f"Successfully uploaded image: {filename}")
            logger.info(f"Public URL: {public_url}")
            return public_url
            
        except Exception as e:
            error_msg = f"Failed to upload image to storage: {str(e)}"
            logger.error(error_msg)
            raise StorageUploadError(error_msg)
    
    def upload_video_to_storage(self, video_data: bytes, filename: str) -> str:
        """
        Upload video to Supabase Storage
        
        Args:
            video_data: Binary video data
            filename: Name for the file in storage
        
        Returns:
            str: Public URL of the uploaded video
        
        Raises:
            StorageUploadError: If upload fails
        """
        try:
            # Upload to Supabase Storage
            response = self.client.storage.from_(self.storage_bucket).upload(
                filename,
                video_data,
                {
                    'content-type': 'video/mp4',
                    'upsert': 'false'
                }
            )
            
            # Get public URL
            public_url = self.client.storage.from_(self.storage_bucket).get_public_url(filename)
            
            logger.info(f"Successfully uploaded video: {filename}")
            logger.info(f"Public URL: {public_url}")
            return public_url
            
        except Exception as e:
            error_msg = f"Failed to upload video to storage: {str(e)}"
            logger.error(error_msg)
            raise StorageUploadError(error_msg)

# ============================================================================
# SCRIPT GENERATOR
# ============================================================================
class ScriptGenerator:
    """Generates cryptic horror scripts using AI"""
    
    SYSTEM_PROMPT = """You are an expert Cryptic Scriptwriter for viral horror and mystery shorts. Create atmospheric two-character dialogues that hint at terrifying backstories without revealing them directly.

Generate exactly 8 lines of dialogue following this structure:
- Lines 1-2: The Hook (grab attention)
- Lines 3-7: The Mystery (deepen with vague hints)
- Line 8: The Cliffhanger (powerful suspenseful ending by Character 2)

RULES:
1. DO NOT reveal specific names, events, dates, or locations from the source
2. Create natural conversation flow
3. Each line must logically follow the previous
4. Use only "character1" and "character2"
5. Keep dialogue short and punchy
6. Build mystery, not answers

OUTPUT: Return ONLY a JSON array with exactly 8 objects in this format:
[
{"character": "character1", "dialogue": "line 1"},
{"character": "character2", "dialogue": "line 2"},
...
]"""
    
    def __init__(self):
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
    
    def generate_script(self, story_input: str, model: str = None) -> Optional[List[DialogueLine]]:
        """Generate script from story input using AI"""
        model = model or config.PRIMARY_MODEL
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        json_schema = {
            "name": "dialogue_script",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "dialogue_lines": {
                        "type": "array",
                        "description": "Array of exactly 8 dialogue lines",
                        "items": {
                            "type": "object",
                            "properties": {
                                "character": {
                                    "type": "string",
                                    "description": "Character name (character1 or character2)"
                                },
                                "dialogue": {
                                    "type": "string",
                                    "description": "The dialogue line"
                                }
                            },
                            "required": ["character", "dialogue"],
                            "additionalProperties": False
                        },
                        "minItems": 8,
                        "maxItems": 8
                    }
                },
                "required": ["dialogue_lines"],
                "additionalProperties": False
            }
        }
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": f"Convert the below story into a script:\n\n{story_input}"}
            ],
            "temperature": 0.7,
            "max_tokens": 1000,
            "response_format": {
                "type": "json_schema",
                "json_schema": json_schema
            }
        }
        
        try:
            logger.info(f"Generating script with model: {model}")
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=config.TIMEOUT_SECONDS
            )
            
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            logger.info(f"AI Model Output: {content}")
            parsed = json.loads(content)
            
            if isinstance(parsed, dict) and 'dialogue_lines' in parsed:
                dialogue_data = parsed['dialogue_lines']
            elif isinstance(parsed, list):
                dialogue_data = parsed
            else:
                raise ValueError(f"Unexpected response format: {type(parsed)}")
            
            if len(dialogue_data) != 8:
                raise ValueError(f"Expected 8 dialogue lines, got {len(dialogue_data)}")
            
            dialogue_lines = [DialogueLine(**line) for line in dialogue_data]
            logger.info("Script generated successfully")
            return dialogue_lines
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Error response: {e.response.text}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"Script generation failed: {e}")
            return None
    
    def generate_with_retry(self, story_input: str) -> Optional[List[DialogueLine]]:
        """Generate script with retry logic and fallback model"""
        for attempt in range(config.MAX_RETRIES):
            logger.info(f"Attempt {attempt + 1}/{config.MAX_RETRIES} with primary model")
            result = self.generate_script(story_input, config.PRIMARY_MODEL)
            if result:
                return result
        
        logger.warning("Primary model failed, trying fallback")
        for attempt in range(config.MAX_RETRIES):
            logger.info(f"Fallback attempt {attempt + 1}/{config.MAX_RETRIES}")
            result = self.generate_script(story_input, config.FALLBACK_MODEL)
            if result:
                return result
        
        logger.error("All retry attempts exhausted")
        return None

# ============================================================================
# BACKGROUND IMAGE PROMPT GENERATOR
# ============================================================================
class BackgroundImagePromptGenerator:
    """Generates cinematic background image prompts using AI"""
    
    SYSTEM_PROMPT = """You are an expert Background Image Prompt Generator for cinematic, atmospheric scenes. Your task is to create detailed prompts for AI image generation that produce versatile, story-appropriate background images suitable for Instagram (9:16 vertical format).

Generate a single, detailed image prompt that captures the emotional tone and setting of the story WITHOUT showing explicit action or detailed character faces.

PROMPT STRUCTURE:
Your output must follow this structure in order of importance:
1. Image Type: "Cinematic, atmospheric photograph" or "Cinematic scene"
2. Setting/Environment: The primary location or backdrop (architecture, landscapes, anonymous crowds)
3. Mood/Atmosphere: Emotional tone derived from the story (tense, hopeful, mysterious, dramatic, eerie)
4. Lighting: Specific lighting that supports the mood (warm glow, overcast grey, dim shadows, neon glow)
5. Color Palette: Muted and cinematic colors with film grain (warm tones, cool blues, desaturated)
6. Composition: Camera angle and framing (wide shot, environmental backdrop, depth of field)
7. Style Modifiers: "photorealistic", "atmospheric", "cinematic", "softened for text overlay"

REQUIREMENTS:
1. Format: 9:16 vertical canvas (1080 × 1920 px) Instagram-friendly
2. Scene Focus: Environmental backdrop only - avoid sharp detail on people
3. Mood Accuracy: Lighting and atmosphere must reflect the story's emotional tone
4. Detail Level: Subtle and versatile - must work as a background layer
5. Avoid: Clear faces, text, or distracting focal points
6. Color: Muted, cinematic palette with slight film grain for texture
7. Depth: Include visual depth but keep background elements soft

OUTPUT RULES:
1. Generate ONLY the image prompt - no explanations, no JSON, no markdown
2. Prompt must be 80-150 words
3. Always include "9:16 vertical format" or "vertical composition"
4. Always include "suitable for text overlay" or "softened for text overlay"
5. Always end with "photorealistic, no clear faces, no text"
6. Use positive phrasing - describe what SHOULD be in the image
7. Focus on atmosphere and environment, not specific plot details"""
    
    def __init__(self):
        self.api_key = config.OPENROUTER_API_KEY
        self.base_url = config.OPENROUTER_BASE_URL
    
    def generate_prompt(
        self,
        story_input: str,
        script: Optional[str] = None,
        model: str = None
    ) -> Optional[str]:
        """
        Generate background image prompt from story and optional script
        
        Args:
            story_input: The original story text
            script: Optional generated script (JSON string or list of dialogue)
            model: AI model to use (defaults to PRIMARY_MODEL)
        
        Returns:
            Generated image prompt string or None if generation fails
        """
        model = model or config.PRIMARY_MODEL
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        user_message = f"Story: {story_input}"
        if script:
            user_message += f"\n\nScript: {script}"
        user_message += "\n\nGenerate a cinematic background image prompt for this story."
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        try:
            logger.info(f"Generating background image prompt with model: {model}")
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=config.TIMEOUT_SECONDS
            )
            
            response.raise_for_status()
            result = response.json()
            prompt_output = result['choices'][0]['message']['content'].strip()
            
            prompt_output = self._clean_prompt(prompt_output)
            
            logger.info(f"Background image prompt generated successfully")
            logger.info(f"Prompt preview: {prompt_output[:100]}...")
            return prompt_output
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Error response: {e.response.text}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Prompt generation failed: {e}")
            return None
    
    def _clean_prompt(self, prompt: str) -> str:
        """
        Clean the generated prompt by removing markdown and special characters
        
        Args:
            prompt: Raw prompt from AI
        
        Returns:
            Cleaned prompt string
        """
        cleaned = prompt.replace('**', '').replace('`', '').replace('*', '')
        cleaned = cleaned.replace('\n', ' ')
        cleaned = ' '.join(cleaned.split())
        
        if len(cleaned) > 2000:
            cleaned = cleaned[:2000]
        
        return cleaned.strip()
    
    def generate_with_retry(
        self,
        story_input: str,
        script: Optional[str] = None
    ) -> Optional[str]:
        """Generate prompt with retry logic and fallback model"""
        for attempt in range(config.MAX_RETRIES):
            logger.info(f"Attempt {attempt + 1}/{config.MAX_RETRIES} with primary model")
            result = self.generate_prompt(story_input, script, config.PRIMARY_MODEL)
            if result:
                return result
        
        logger.warning("Primary model failed, trying fallback")
        for attempt in range(config.MAX_RETRIES):
            logger.info(f"Fallback attempt {attempt + 1}/{config.MAX_RETRIES}")
            result = self.generate_prompt(story_input, script, config.FALLBACK_MODEL)
            if result:
                return result
        
        logger.error("All retry attempts exhausted")
        return None

# ============================================================================
# BACKGROUND IMAGE GENERATOR
# ============================================================================
class BackgroundImageGenerator:
    """Generates images from prompts using Hugging Face API"""
    
    def __init__(self):
        self.api_key = config.HUGGINGFACE_API_KEY
        self.image_url = config.HUGGINGFACE_IMAGE_URL
        self.model = config.HUGGINGFACE_MODEL
    
    def generate_image(self, prompt: str) -> Optional[bytes]:
        """
        Generate image from prompt using Hugging Face API
        
        Args:
            prompt: The image generation prompt
        
        Returns:
            bytes: Image binary data or None if generation fails
        
        Raises:
            ImageGenerationError: If image generation fails
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "image/png"
        }
        
        payload = {
            "prompt": prompt,
            "model": self.model
        }
        
        try:
            logger.info(f"Generating image with Hugging Face model: {self.model}")
            logger.info(f"Prompt: {prompt[:100]}...")
            
            response = requests.post(
                self.image_url,
                headers=headers,
                json=payload,
                timeout=config.TIMEOUT_SECONDS
            )
            
            response.raise_for_status()
            
            # Check if response is JSON (contains URL) or binary image
            content_type = response.headers.get('Content-Type', '')
            
            if 'application/json' in content_type:
                # Response contains URL to download image
                result = response.json()
                logger.info(f"Hugging Face API response: {result}")
                
                # Extract image URL from response
                image_url = None
                if isinstance(result, dict):
                    if 'data' in result and isinstance(result['data'], list) and len(result['data']) > 0:
                        image_url = result['data'][0].get('url')
                    elif 'url' in result:
                        image_url = result['url']
                
                if not image_url:
                    raise ImageGenerationError("No image URL found in response")
                
                logger.info(f"Downloading image from URL: {image_url}")
                
                # Download the image
                img_response = requests.get(image_url, timeout=config.TIMEOUT_SECONDS)
                img_response.raise_for_status()
                image_data = img_response.content
                
            elif 'image' in content_type:
                # Response is direct binary image
                image_data = response.content
            else:
                raise ImageGenerationError(f"Unexpected content type: {content_type}")
            
            logger.info(f"Image generated successfully. Size: {len(image_data)} bytes")
            return image_data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Image generation API request failed: {str(e)}"
            logger.error(error_msg)
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Error response: {e.response.text}")
            raise ImageGenerationError(error_msg)
        except Exception as e:
            error_msg = f"Image generation failed: {str(e)}"
            logger.error(error_msg)
            raise ImageGenerationError(error_msg)
    
    def generate_with_retry(self, prompt: str) -> Optional[bytes]:
        """Generate image with retry logic"""
        for attempt in range(config.MAX_RETRIES):
            try:
                logger.info(f"Image generation attempt {attempt + 1}/{config.MAX_RETRIES}")
                return self.generate_image(prompt)
            except ImageGenerationError as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt == config.MAX_RETRIES - 1:
                    logger.error("All image generation attempts exhausted")
                    return None
        return None

# ============================================================================
# RESOURCE DOWNLOADER (for video generation)
# ============================================================================
class ResourceDownloader:
    """Handles downloading of images, audio, and fonts from various sources"""
    
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    @staticmethod
    def extract_google_drive_id(url: str) -> Optional[str]:
        """Extract Google Drive file ID from URL"""
        patterns = [
            r'/file/d/([a-zA-Z0-9_-]+)',
            r'id=([a-zA-Z0-9_-]+)',
            r'/d/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    @staticmethod
    def is_google_drive_url(url: str) -> bool:
        """Check if URL is a Google Drive URL"""
        return "drive.google.com" in url
    
    def convert_to_direct_url(self, url: str) -> str:
        """Convert various URL formats to direct download URL"""
        if self.is_google_drive_url(url):
            file_id = self.extract_google_drive_id(url)
            if file_id:
                return f"https://drive.google.com/uc?export=download&id={file_id}"
        return url
    
    def download_with_retry(self, url: str, max_retries: int = 3) -> requests.Response:
        """Download with retry logic and confirmation handling"""
        download_url = self.convert_to_direct_url(url)
        
        for attempt in range(max_retries):
            try:
                response = self.session.get(download_url, headers=self.headers, stream=True, timeout=30)
                
                if response.status_code != 200:
                    if attempt < max_retries - 1:
                        continue
                    raise ResourceDownloadError(f"HTTP {response.status_code} for {url}")
                
                # Handle Google Drive confirmation pages
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' in content_type and self.is_google_drive_url(url):
                    html_content = response.text
                    
                    # Look for confirmation link
                    confirm_pattern = r'href="(/uc\?export=download[^"]*)"'
                    confirm_match = re.search(confirm_pattern, html_content)
                    
                    if confirm_match:
                        confirm_url = "https://drive.google.com" + confirm_match.group(1).replace('&amp;', '&')
                        response = self.session.get(confirm_url, headers=self.headers, stream=True, timeout=30)
                    
                    # Look for confirm token
                    elif 'confirm=' in html_content:
                        confirm_match = re.search(r'confirm=([a-zA-Z0-9_-]+)', html_content)
                        if confirm_match:
                            file_id = self.extract_google_drive_id(url)
                            confirm_token = confirm_match.group(1)
                            confirm_url = f"https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}"
                            response = self.session.get(confirm_url, headers=self.headers, stream=True, timeout=30)
                
                return response
                
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                raise ResourceDownloadError(f"Failed to download from {url}: {str(e)}")
        
        raise ResourceDownloadError(f"Max retries exceeded for {url}")
    
    def download_image(self, url: str, image_type: str = "generic") -> Image.Image:
        """
        Download and process image to memory with appropriate sizing
        
        Args:
            url: Image URL
            image_type: Type of image - "character", "background", "incident", "mask"
        """
        try:
            response = self.download_with_retry(url)
            
            # Load image to memory
            image_buffer = io.BytesIO()
            for chunk in response.iter_content(8192):
                if chunk:
                    image_buffer.write(chunk)
            
            image_buffer.seek(0)
            img = Image.open(image_buffer)
            
            # Apply appropriate sizing based on image type
            if image_type == "character":
                # Characters: reasonable size for processing (will be resized later)
                max_size = (600, 900)
            elif image_type == "background":
                # Background: match canvas width (720) with appropriate height
                max_size = (720, 1280)
            elif image_type == "incident":
                # Incident images: full canvas size
                max_size = (720, 1280)
            elif image_type == "mask":
                # Mask: smaller size is fine, will be resized to match incident images
                max_size = (720, 720)
            else:
                max_size = (800, 800)
            
            # Resize if needed
            if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to appropriate mode
            if image_type == "character":
                # Characters need RGBA for transparency
                img = img.convert('RGBA')
            elif image_type in ["background", "incident", "mask"]:
                # Background and incidents can be RGB
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            else:
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGBA')
                else:
                    img = img.convert('RGB')
            
            print(f"Downloaded {image_type} image: {img.size} from {url[:50]}...")
            return img
            
        except Exception as e:
            raise ResourceDownloadError(f"Failed to download {image_type} image from {url}: {str(e)}")
    
    def download_audio(self, url: str) -> str:
        """Download audio to temporary file"""
        try:
            response = self.download_with_retry(url)
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
                bytes_written = 0
                
                for chunk in response.iter_content(8192):
                    if chunk:
                        temp_file.write(chunk)
                        bytes_written += len(chunk)
            
            if bytes_written == 0:
                raise AudioProcessingError("Audio download resulted in 0 bytes")
            
            if bytes_written < 1000:
                raise AudioProcessingError(f"Audio file too small ({bytes_written} bytes)")
            
            # Validate audio file
            try:
                test_audio = AudioFileClip(temp_path)
                duration = test_audio.duration
                test_audio.close()
                print(f"Audio validated - Duration: {duration:.2f}s, Size: {bytes_written/1024:.2f}KB")
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise AudioProcessingError(f"Invalid audio file: {str(e)}")
            
            return temp_path
            
        except Exception as e:
            raise ResourceDownloadError(f"Failed to download audio from {url}: {str(e)}")
    
    def download_font(self, url: str, font_size: int = 36) -> Tuple[ImageFont.FreeTypeFont, Optional[str]]:
        """Download font and return loaded font object"""
        try:
            response = self.download_with_retry(url)
            
            # Save to temporary file
            temp_dir = tempfile.gettempdir()
            font_filename = f"video_font_{os.getpid()}.ttf"
            temp_path = os.path.join(temp_dir, font_filename)
            
            bytes_written = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(8192):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)
            
            if bytes_written == 0:
                raise Exception("Font download resulted in 0 bytes")
            
            # Load font
            font = ImageFont.truetype(temp_path, font_size)
            print(f"Font loaded from URL - Size: {bytes_written/1024:.2f}KB")
            return font, temp_path
            
        except Exception as e:
            print(f"Failed to load font from URL: {e}, using default font")
            return ImageFont.load_default(), None
    
    def close(self):
        """Close the session"""
        self.session.close()

# ============================================================================
# IMAGE PROCESSOR (for video generation)
# ============================================================================
class ImageProcessor:
    """Handles image processing and manipulation"""
    
    @staticmethod
    def resize_character_image(img: Image.Image, target_size: Tuple[int, int] = (500, 700)) -> Image.Image:
        """Resize character image maintaining aspect ratio and bottom alignment"""
        try:
            img = img.convert("RGBA")
            img_width, img_height = img.size
            target_width, target_height = target_size
            
            # Calculate scaling factor - fit within target size
            scale_factor = min(target_width / img_width, target_height / img_height)
            
            # Calculate new dimensions
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            
            # Resize image with high quality
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create final image with transparent background
            final_img = Image.new('RGBA', target_size, (0, 0, 0, 0))
            
            # Center horizontally, align to bottom
            x_offset = (target_width - new_width) // 2
            y_offset = target_height - new_height
            
            final_img.paste(resized_img, (x_offset, y_offset), resized_img)
            resized_img.close()
            
            print(f"Character resized: {img.size} -> {new_width}x{new_height} in {target_size} canvas")
            return final_img
            
        except Exception as e:
            raise VideoGenerationError(f"Failed to resize character image: {str(e)}")
    
    @staticmethod
    def resize_background_image(img: Image.Image, canvas_size: Tuple[int, int]) -> Image.Image:
        """Resize background image to cover canvas"""
        try:
            target_width, target_height = canvas_size
            
            # Convert to RGB if needed
            if img.mode == 'RGBA':
                # Create white background
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate scaling to cover the canvas
            img_width, img_height = img.size
            scale_width = target_width / img_width
            scale_height = target_height / img_height
            scale = max(scale_width, scale_height)  # Use max to cover entire canvas
            
            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize with high quality
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Crop to canvas size (center crop)
            left = (new_width - target_width) // 2
            top = (new_height - target_height) // 2
            right = left + target_width
            bottom = top + target_height
            
            final_img = resized_img.crop((left, top, right, bottom))
            resized_img.close()
            
            print(f"Background resized and cropped: {img.size} -> {final_img.size}")
            return final_img
            
        except Exception as e:
            raise VideoGenerationError(f"Failed to resize background image: {str(e)}")
    
    @staticmethod
    def resize_incident_image(img: Image.Image, canvas_size: Tuple[int, int]) -> Image.Image:
        """Resize incident image to fit canvas while maintaining aspect ratio"""
        try:
            target_width, target_height = canvas_size
            
            # Convert RGBA to RGB
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (0, 0, 0))
                background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate scaling to fit within canvas
            img_width, img_height = img.size
            scale = min(target_width / img_width, target_height / img_height)
            
            # Calculate new dimensions
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize with high quality
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create canvas and center the image
            final_img = Image.new('RGB', canvas_size, (0, 0, 0))
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            final_img.paste(resized_img, (x_offset, y_offset))
            
            resized_img.close()
            
            print(f"Incident image resized: {img.size} -> {new_width}x{new_height}")
            return final_img
            
        except Exception as e:
            raise VideoGenerationError(f"Failed to resize incident image: {str(e)}")
    
    @staticmethod
    def apply_film_grain(image: Image.Image, mask: Image.Image, grain_strength: float = 0.5) -> Image.Image:
        """Apply film grain effect using mask image"""
        try:
            image = image.convert("RGB")
            mask = mask.convert("RGB")
            
            # Resize mask to match image
            resized_mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            
            # Reduce mask brightness
            enhancer = ImageEnhance.Brightness(resized_mask)
            dimmed_mask = enhancer.enhance(0.5)
            
            # Apply grain
            multiplied = ImageChops.multiply(image, dimmed_mask)
            
            # Blend with original
            blended_img = Image.blend(image, multiplied, alpha=grain_strength)
            
            # Cleanup
            resized_mask.close()
            dimmed_mask.close()
            
            return blended_img
            
        except Exception as e:
            print(f"Failed to apply film grain: {e}")
            return image
    
    @staticmethod
    def generate_text_image(
        text: str,
        font: ImageFont.FreeTypeFont,
        size: Tuple[int, int] = (720, 200),
        text_color: str = 'white',
        bg_color: Tuple[int, int, int, int] = (0, 0, 0, 180)
    ) -> Image.Image:
        """Generate text image with background and stroke effect"""
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        
        # Word wrapping
        lines = []
        words = text.split()
        if not words:
            return img
        
        line = ""
        for word in words:
            test_line = line + word + " " if line else word + " "
            if d.textlength(test_line.strip(), font=font) <= size[0] - 60:
                line = test_line
            else:
                if line:
                    lines.append(line.strip())
                    line = word + " "
                else:
                    lines.append(word)
                    line = ""
        
        if line:
            lines.append(line.strip())
        
        # Calculate dimensions
        try:
            font_size = font.size
        except:
            font_size = 36  # Default if font.size not available
        
        line_height = font_size + 15
        total_text_height = line_height * len(lines)
        max_line_width = max(d.textlength(line, font=font) for line in lines) if lines else 0
        
        # Background dimensions
        padding = 20
        bg_width = max_line_width + (padding * 2)
        bg_height = total_text_height + (padding * 2)
        bg_x = (size[0] - bg_width) // 2
        bg_y = (size[1] - bg_height) // 2
        
        # Draw rounded rectangle background
        background = Image.new('RGBA', size, (0, 0, 0, 0))
        bg_draw = ImageDraw.Draw(background)
        bg_draw.rounded_rectangle(
            [bg_x, bg_y, bg_x + bg_width, bg_y + bg_height],
            radius=10,
            fill=bg_color
        )
        
        img = Image.alpha_composite(img, background)
        d = ImageDraw.Draw(img)
        
        y_start = bg_y + padding
        
        # Draw text with stroke
        stroke_width = 2
        for i, line in enumerate(lines):
            line_width = d.textlength(line, font=font)
            x_text = (size[0] - line_width) // 2
            y_text = y_start + (i * line_height)
            
            # Stroke effect
            for dx in range(-stroke_width, stroke_width + 1):
                for dy in range(-stroke_width, stroke_width + 1):
                    if dx != 0 or dy != 0:
                        d.text((x_text + dx, y_text + dy), line, font=font, fill='black')
            
            # Main text
            d.text((x_text, y_text), line, font=font, fill=text_color)
        
        return img

# ============================================================================
# VIDEO GENERATOR
# ============================================================================
class VideoGenerator:
    """Main video generation class"""
    
    def __init__(self):
        self.downloader = ResourceDownloader()
        self.processor = ImageProcessor()
        self.temp_files = []
    
    def generate_video_from_db_row(self, row: dict) -> Tuple[bytes, float]:
        """
        Generate video from database row data
        
        Args:
            row: Database row containing video generation parameters
        
        Returns:
            Tuple of (video_bytes, duration)
        
        Raises:
            VideoGenerationError: If video generation fails
        """
        try:
            record_id = row.get('id')
            print(f"\n{'='*60}")
            print(f"Starting video generation for record ID: {record_id}")
            print(f"{'='*60}\n")
            
            # Extract parameters from row with defaults
            canvas_width = 720
            canvas_height = 1280
            fps = 24
            scene_duration = 2.0
            font_size = 36
            
            # Get required fields
            ai_script = row.get('ai_script')
            if not ai_script:
                raise VideoGenerationError("ai_script is required")
            
            # Convert script format from character1/character2 to character_1/character_2
            script_lines = []
            if isinstance(ai_script, list):
                for line in ai_script:
                    char = line.get('character', '')
                    # Normalize character names
                    if char.lower() in ['character1', 'character_1']:
                        char = 'character_1'
                    elif char.lower() in ['character2', 'character_2']:
                        char = 'character_2'
                    
                    script_lines.append({
                        'character': char,
                        'dialogue': line.get('dialogue', '')
                    })
            
            background_image_url = row.get('background_image_url')
            if not background_image_url:
                raise VideoGenerationError("background_image_url is required")
            
            # Get optional fields with defaults
            character_1_url = row.get('character_a') or config.DEFAULT_CHARACTER_1_URL
            character_2_url = row.get('character_b') or config.DEFAULT_CHARACTER_2_URL
            
            incident_image_1 = row.get('incident_image_1')
            incident_image_2 = row.get('incident_image_2')
            incident_image_3 = row.get('incident_image_3')
            
            background_audio = row.get('background_audio_url')  # Assuming this column exists
            mask_image = config.DEFAULT_MASK_URL
            font_url = config.DEFAULT_FONT_URL
            if background_audio is None or background_audio == "":
                background_audio = config.DEFAULT_AUDIO_URL
            
            # Download resources
            resources = self._download_resources(
                character_1_url,
                character_2_url,
                background_image_url,
                incident_image_1,
                incident_image_2,
                incident_image_3,
                background_audio,
                mask_image,
                font_url,
                font_size
            )
            
            # Create video
            video_bytes, duration = self._create_video(
                script_lines,
                resources,
                canvas_width,
                canvas_height,
                fps,
                scene_duration
            )
            
            print(f"\n{'='*60}")
            print(f"✅ Video generated successfully!")
            print(f"   Size: {len(video_bytes)/(1024*1024):.2f}MB")
            print(f"   Duration: {duration:.1f}s")
            print(f"{'='*60}\n")
            
            return video_bytes, duration
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ Video generation failed: {str(e)}")
            print(f"{'='*60}\n")
            raise VideoGenerationError(str(e))
        finally:
            self._cleanup()
    
    def _download_resources(
        self,
        character_1_url: str,
        character_2_url: str,
        background_url: str,
        incident_1_url: Optional[str],
        incident_2_url: Optional[str],
        incident_3_url: Optional[str],
        audio_url: Optional[str],
        mask_url: str,
        font_url: str,
        font_size: int
    ) -> dict:
        """Download all required resources"""
        print("📥 Downloading resources...\n")
        
        resources = {}
        
        # Download font
        font, temp_path = self.downloader.download_font(font_url, font_size)
        if temp_path:
            self.temp_files.append(temp_path)
        resources['font'] = font
        
        # Download images
        print("Downloading character images...")
        resources['char1'] = self.downloader.download_image(character_1_url, "character")
        resources['char2'] = self.downloader.download_image(character_2_url, "character")
        
        print("Downloading background image...")
        resources['background'] = self.downloader.download_image(background_url, "background")
        
        # Download incident images
        print("Downloading incident images...")
        for i, img_url in enumerate([incident_1_url, incident_2_url, incident_3_url], 1):
            if img_url:
                resources[f'incident_{i}'] = self.downloader.download_image(img_url, "incident")
        
        # Download mask
        print("Downloading mask image...")
        resources['mask'] = self.downloader.download_image(mask_url, "mask")
        
        # Download audio
        if audio_url:
            print("Downloading background audio...")
            audio_path = self.downloader.download_audio(audio_url)
            self.temp_files.append(audio_path)
            resources['audio_path'] = audio_path
        
        print("\n✅ All resources downloaded successfully\n")
        return resources
    
    def _create_video(
        self,
        script: List[dict],
        resources: dict,
        canvas_width: int,
        canvas_height: int,
        fps: int,
        scene_duration: float
    ) -> Tuple[bytes, float]:
        """Create video from resources"""
        canvas_size = (canvas_width, canvas_height)
        
        # Calculate duration
        num_scenes = len(script)
        num_incidents = sum([1 for k in resources if k.startswith('incident_')])
        video_duration = (num_scenes + num_incidents) * scene_duration
        
        print(f"🎬 Creating video composition:")
        print(f"   Canvas: {canvas_size[0]}x{canvas_size[1]}")
        print(f"   Dialogue scenes: {num_scenes}")
        print(f"   Incident scenes: {num_incidents}")
        print(f"   Total duration: {video_duration:.1f}s\n")
        
        # Process and resize background
        print("Processing background...")
        background_processed = self.processor.resize_background_image(resources['background'], canvas_size)
        background_array = np.array(background_processed)
        background_clip = ImageClip(background_array).with_duration(video_duration).with_position("center")
        
        clips = [background_clip]
        
        # Process character images
        print("Processing character images...")
        char1_processed = self.processor.resize_character_image(resources['char1'])
        char2_processed = self.processor.resize_character_image(resources['char2'])
        
        # Create dialogue scenes
        print(f"\nCreating {num_scenes} dialogue scenes...")
        start_time = 0
        
        for idx, dialogue_dict in enumerate(script, 1):
            is_char1 = dialogue_dict['character'] == 'character_1'
            dialogue_text = dialogue_dict['dialogue']
            
            # Select character
            if is_char1:
                char_img = char1_processed.copy()
            else:
                char_img = char2_processed.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Create character clip
            char_array = np.array(char_img)
            char_clip = ImageClip(char_array).with_duration(scene_duration).with_start(start_time)
            
            # Position character
            char_y_position = canvas_size[1] - char_img.height - 100
            
            if is_char1:
                char_clip = char_clip.with_position((50, char_y_position))
            else:
                char_clip = char_clip.with_position((canvas_size[0] - char_img.width - 50, char_y_position))
            
            # Generate text overlay
            text_img = self.processor.generate_text_image(
                dialogue_text,
                resources['font'],
                size=(canvas_size[0] - 40, 200)
            )
            
            text_array = np.array(text_img)
            text_clip = ImageClip(text_array).with_duration(scene_duration).with_start(start_time).with_position(('center', 150))
            
            clips.extend([char_clip, text_clip])
            
            # Cleanup
            text_img.close()
            if not is_char1:
                char_img.close()
            
            del text_array, char_array
            
            print(f"   Scene {idx}/{num_scenes}: {dialogue_dict['character']} - \"{dialogue_text[:30]}...\"")
            start_time += scene_duration
        
        # Add incident scenes
        if num_incidents > 0:
            print(f"\nCreating {num_incidents} incident scenes...")
            mask = resources.get('mask')
            
            for i in range(1, 4):
                if f'incident_{i}' in resources:
                    incident_img = resources[f'incident_{i}']
                    
                    # Resize incident image to canvas
                    incident_processed = self.processor.resize_incident_image(incident_img, canvas_size)
                    
                    # Apply film grain if mask available
                    if mask:
                        incident_processed = self.processor.apply_film_grain(incident_processed, mask)
                    
                    incident_array = np.array(incident_processed)
                    incident_clip = ImageClip(incident_array).with_duration(scene_duration).with_start(start_time).with_position('center')
                    
                    clips.append(incident_clip)
                    
                    incident_processed.close()
                    del incident_array
                    
                    print(f"   Incident scene {i}")
                    start_time += scene_duration
        
        # Cleanup character images
        char1_processed.close()
        char2_processed.close()
        background_processed.close()
        resources['char1'].close()
        resources['char2'].close()
        resources['background'].close()
        
        del background_array
        
        # Composite video
        print("\n🎞️  Compositing video clips...")
        video = CompositeVideoClip(clips, size=canvas_size)
        
        # Add audio if available
        if 'audio_path' in resources:
            print("🎵 Adding background audio...")
            audio_clip = AudioFileClip(resources['audio_path'])
            
            # Adjust audio duration
            if audio_clip.duration > video_duration:
                print(f"   Trimming audio: {audio_clip.duration:.1f}s -> {video_duration:.1f}s")
                audio_clip = audio_clip.with_duration(video_duration)
            elif audio_clip.duration < video_duration:
                print(f"   Looping audio: {audio_clip.duration:.1f}s -> {video_duration:.1f}s")
                audio_clip = audio_clip.loop(duration=video_duration)
            
            video = video.with_audio(audio_clip)
        
        # Export video
        print("\n💾 Exporting video (this may take a few minutes)...")
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_filename = temp_file.name
            self.temp_files.append(temp_filename)
        
        video.write_videofile(
            temp_filename,
            fps=fps,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=tempfile.mktemp(suffix='.m4a'),
            remove_temp=True,
            preset='superfast',
            threads=os.cpu_count(),
            ffmpeg_params=['-crf', '28', '-movflags', '+faststart'],
            logger=None
        )
        
        # Read video to bytes
        with open(temp_filename, 'rb') as f:
            video_bytes = f.read()
        
        # Cleanup
        video.close()
        if 'audio_path' in resources:
            audio_clip.close()
        
        gc.collect()
        
        return video_bytes, video_duration
    
    def _cleanup(self):
        """Clean up temporary files"""
        print("\n🧹 Cleaning up temporary files...")
        cleaned = 0
        
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    cleaned += 1
                except Exception as e:
                    print(f"   ⚠️  Failed to cleanup {temp_file}: {e}")
        
        if cleaned > 0:
            print(f"   ✅ Cleaned up {cleaned} temporary file(s)")
        
        self.temp_files.clear()
        self.downloader.close()

# ============================================================================
# PROCESSORS
# ============================================================================
class ScriptProcessor:
    """Orchestrates the script generation workflow"""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.generator = ScriptGenerator()
    
    def process_single(self, request_data: ScriptRequest) -> ScriptResponse:
        """
        Process a single story - generates script but saves ONLY story summary to DB
        
        Args:
            request_data: Script generation request
        
        Returns:
            ScriptResponse with generated script and database info
        """
        generated_script = self.generator.generate_with_retry(request_data.story_input)
        
        if not generated_script:
            return ScriptResponse(
                success=False,
                error="Failed to generate script"
            )
        
        result_data = {
            "success": True,
            "script": generated_script,
            "message": "Script generated successfully"
        }
        
        if request_data.save_to_db:
            try:
                story_record = self.db.save_story_only(
                    article_summary=request_data.story_input,
                )
                
                result_data["record_id"] = story_record.get("id")
                result_data["saved_to_db"] = True
                result_data["message"] = "Script generated and story summary saved to database"
                
            except StoryInsertionError as e:
                result_data["saved_to_db"] = False
                result_data["db_error"] = str(e)
                result_data["message"] = "Script generated but failed to save story summary"
        
        return ScriptResponse(**result_data)
    
    def process_batch(self, limit: Optional[int] = None) -> Dict:
        """
        Process all rows with empty ai_script field
        
        Args:
            limit: Optional limit on number of rows to process
        
        Returns:
            Dictionary with processing results
        """
        results = {
            "total_rows": 0,
            "processed_count": 0,
            "failed_count": 0,
            "errors": []
        }
        
        try:
            empty_rows = self.db.get_empty_script_rows(limit)
            results["total_rows"] = len(empty_rows)
            
            if not empty_rows:
                logger.info("No rows with empty ai_script found")
                return results
            
            logger.info(f"Processing {len(empty_rows)} rows...")
            
            for row in empty_rows:
                record_id = row.get("id")
                article_summary = row.get("article_summary")
                
                if not article_summary:
                    results["failed_count"] += 1
                    results["errors"].append({
                        "id": record_id,
                        "error": "Missing article_summary"
                    })
                    continue
                
                generated_script = self.generator.generate_with_retry(article_summary)
                
                if not generated_script:
                    results["failed_count"] += 1
                    results["errors"].append({
                        "id": record_id,
                        "error": "Failed to generate script"
                    })
                    continue
                
                if self.db.update_script(record_id, generated_script):
                    results["processed_count"] += 1
                    logger.info(f"Successfully processed record {record_id}")
                else:
                    results["failed_count"] += 1
                    results["errors"].append({
                        "id": record_id,
                        "error": "Failed to update database"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            results["errors"].append({"error": str(e)})
            return results

class BackgroundPromptProcessor:
    """Orchestrates background prompt generation workflow"""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.generator = BackgroundImagePromptGenerator()
    
    def process_batch(self, limit: Optional[int] = None) -> Dict:
        """
        Process all rows where background_image_prompt is null
        
        Args:
            limit: Optional limit on number of rows to process
        
        Returns:
            Dictionary with processing results
        """
        results = {
            "total_rows": 0,
            "processed_count": 0,
            "failed_count": 0,
            "errors": []
        }
        
        try:
            rows = self.db.get_rows_with_null_background_prompt(limit)
            results["total_rows"] = len(rows)
            
            if not rows:
                logger.info("No rows with null background_image_prompt found")
                return results
            
            logger.info(f"Processing {len(rows)} rows for background image prompts...")
            
            for row in rows:
                record_id = row.get("id")
                article_summary = row.get("article_summary")
                ai_script = row.get("ai_script")
                
                if not article_summary:
                    results["failed_count"] += 1
                    results["errors"].append({
                        "id": record_id,
                        "error": "Missing article_summary"
                    })
                    continue
                
                script_str = None
                if ai_script:
                    if isinstance(ai_script, (list, dict)):
                        script_str = json.dumps(ai_script)
                    else:
                        script_str = str(ai_script)
                
                background_prompt = self.generator.generate_with_retry(
                    article_summary,
                    script_str
                )
                
                if not background_prompt:
                    results["failed_count"] += 1
                    results["errors"].append({
                        "id": record_id,
                        "error": "Failed to generate background prompt"
                    })
                    continue
                
                if self.db.update_background_prompt(record_id, background_prompt):
                    results["processed_count"] += 1
                    logger.info(f"Successfully processed record {record_id}")
                else:
                    results["failed_count"] += 1
                    results["errors"].append({
                        "id": record_id,
                        "error": "Failed to update database"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            results["errors"].append({"error": str(e)})
            return results

class BackgroundImageProcessor:
    """Orchestrates background image generation and upload workflow"""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.image_generator = BackgroundImageGenerator()
    
    def process_batch(self, limit: Optional[int] = None) -> Dict:
        """
        Process all rows where background_image_prompt exists but background_image_url is null
        Generates images and uploads them to Supabase Storage
        
        Args:
            limit: Optional limit on number of rows to process
        
        Returns:
            Dictionary with processing results
        """
        results = {
            "total_rows": 0,
            "processed_count": 0,
            "failed_count": 0,
            "errors": []
        }
        
        try:
            rows = self.db.get_rows_with_null_background_image(limit)
            results["total_rows"] = len(rows)
            
            if not rows:
                logger.info("No rows ready for image generation found")
                return results
            
            logger.info(f"Processing {len(rows)} rows for background image generation...")
            
            for row in rows:
                record_id = row.get("id")
                background_prompt = row.get("background_image_prompt")
                
                if not background_prompt:
                    results["failed_count"] += 1
                    results["errors"].append({
                        "id": record_id,
                        "error": "Missing background_image_prompt"
                    })
                    continue
                
                try:
                    # Step 1: Generate image from prompt
                    logger.info(f"Generating image for record {record_id}...")
                    image_data = self.image_generator.generate_with_retry(background_prompt)
                    
                    if not image_data:
                        raise ImageGenerationError("Failed to generate image")
                    
                    # Step 2: Upload image to Supabase Storage
                    filename = f"background_{record_id}_{int(datetime.utcnow().timestamp())}.png"
                    logger.info(f"Uploading image to Supabase Storage: {filename}")
                    public_url = self.db.upload_image_to_storage(image_data, filename)
                    
                    # Step 3: Update database with image URL
                    if self.db.update_background_image_url(record_id, public_url):
                        results["processed_count"] += 1
                        logger.info(f"Successfully processed record {record_id}")
                        logger.info(f"Image URL: {public_url}")
                    else:
                        raise Exception("Failed to update database with image URL")
                        
                except (ImageGenerationError, StorageUploadError, Exception) as e:
                    results["failed_count"] += 1
                    results["errors"].append({
                        "id": record_id,
                        "error": str(e)
                    })
                    logger.error(f"Failed to process record {record_id}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            results["errors"].append({"error": str(e)})
            return results

class VideoGenerationProcessor:
    """Orchestrates video generation and upload workflow"""
    
    def __init__(self):
        self.db = SupabaseManager()
        self.video_generator = VideoGenerator()
    
    def process_batch(self, limit: Optional[int] = None) -> Dict:
        """
        Process all rows where required fields exist but video_generated_url is null
        Generates videos and uploads them to Supabase Storage
        
        Args:
            limit: Optional limit on number of rows to process
        
        Returns:
            Dictionary with processing results
        """
        results = {
            "total_rows": 0,
            "processed_count": 0,
            "failed_count": 0,
            "errors": []
        }
        
        try:
            rows = self.db.get_rows_ready_for_video(limit)
            results["total_rows"] = len(rows)
            
            if not rows:
                logger.info("No rows ready for video generation found")
                return results
            
            logger.info(f"Processing {len(rows)} rows for video generation...")
            
            for row in rows:
                record_id = row.get("id")
                
                try:
                    # Step 1: Generate video from row data
                    logger.info(f"Generating video for record {record_id}...")
                    video_bytes, duration = self.video_generator.generate_video_from_db_row(row)
                    
                    if not video_bytes:
                        raise VideoGenerationError("Failed to generate video")
                    
                    # Step 2: Upload video to Supabase Storage
                    filename = f"video/video_{record_id}_{int(datetime.utcnow().timestamp())}.mp4"
                    logger.info(f"Uploading video to Supabase Storage: {filename}")
                    public_url = self.db.upload_video_to_storage(video_bytes, filename)
                    
                    # Step 3: Update database with video URL
                    if self.db.update_video_url(record_id, public_url):
                        results["processed_count"] += 1
                        logger.info(f"Successfully processed record {record_id}")
                        logger.info(f"Video URL: {public_url}")
                        logger.info(f"Video duration: {duration:.1f}s")
                    else:
                        raise Exception("Failed to update database with video URL")
                        
                except (VideoGenerationError, StorageUploadError, Exception) as e:
                    results["failed_count"] += 1
                    results["errors"].append({
                        "id": record_id,
                        "error": str(e)
                    })
                    logger.error(f"Failed to process record {record_id}: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            results["errors"].append({"error": str(e)})
            return results

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================
app = FastAPI(
    title="AI Content Automation API",
    description="Complete API for generating scripts, images, and videos with Supabase integration",
    version="4.0.0",
    docs_url="/apidocs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

processor = ScriptProcessor()
bg_prompt_processor = BackgroundPromptProcessor()
bg_image_processor = BackgroundImageProcessor()
video_processor = VideoGenerationProcessor()

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Content Automation API",
        "version": "4.0.0",
        "endpoints": {
            "docs": "/apidocs",
            "health": "/health",
            "generate_script": "/generate-script (POST)",
            "save_story": "/save-story (POST)",
            "batch_scripts": "/batch-generate-scripts (POST)",
            "batch_background_prompts": "/batch-generate-background-prompts (POST)",
            "batch_background_images": "/batch-generate-background-images (POST)",
            "batch_generate_videos": "/batch-generate-videos (POST)"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/generate-script", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    """
    Generate cryptic horror/mystery script from story input
    
    - **story_input**: The story text to convert into script
    - **save_to_db**: Whether to save story summary to database (default: True)
    - **character_a**: Optional name for first character
    - **character_b**: Optional name for second character
    
    Returns generated script and optionally saves story summary to database
    """
    try:
        result = processor.process_single(request)
        return result
    except Exception as e:
        logger.error(f"Script generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/save-story", response_model=StoryResponse)
async def save_story(request: StoryRequest):
    """
    Save a story summary to the database
    
    - **article_summary**: The story text to save
    
    Returns the saved record information
    """
    try:
        db = SupabaseManager()
        record = db.save_story_only(article_summary=request.article_summary)
        
        return StoryResponse(
            success=True,
            message="Story saved successfully",
            record_id=record.get("id")
        )
    except StoryInsertionError as e:
        logger.error(f"Story save error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/batch-generate-scripts", response_model=BatchScriptResponse)
async def batch_generate_scripts(limit: Optional[int] = None):
    """
    Batch process all rows with empty ai_script field
    
    - **limit**: Optional limit on number of rows to process
    
    Processes all rows where ai_script is null and generates scripts for them
    """
    try:
        results = processor.process_batch(limit)
        
        return BatchScriptResponse(
            success=True,
            message=f"Batch processing complete. Processed {results['processed_count']}/{results['total_rows']} rows",
            total_rows=results["total_rows"],
            processed_count=results["processed_count"],
            failed_count=results["failed_count"],
            errors=results["errors"] if results["errors"] else None
        )
    except Exception as e:
        logger.error(f"Batch script generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/batch-generate-background-prompts", response_model=BackgroundPromptBatchResponse)
async def batch_generate_background_prompts(limit: Optional[int] = None):
    """
    Batch generate background image prompts for rows with null background_image_prompt
    
    - **limit**: Optional limit on number of rows to process
    
    Generates cinematic background image prompts for stories
    """
    try:
        results = bg_prompt_processor.process_batch(limit)
        
        return BackgroundPromptBatchResponse(
            success=True,
            message=f"Batch processing complete. Processed {results['processed_count']}/{results['total_rows']} rows",
            total_rows=results["total_rows"],
            processed_count=results["processed_count"],
            failed_count=results["failed_count"],
            errors=results["errors"] if results["errors"] else None
        )
    except Exception as e:
        logger.error(f"Batch background prompt generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/batch-generate-background-images", response_model=BackgroundImageBatchResponse)
async def batch_generate_background_images(limit: Optional[int] = None):
    """
    Batch generate and upload background images for rows with prompts but no images
    
    - **limit**: Optional limit on number of rows to process
    
    Generates images from prompts and uploads to Supabase Storage
    """
    try:
        results = bg_image_processor.process_batch(limit)
        
        return BackgroundImageBatchResponse(
            success=True,
            message=f"Batch processing complete. Processed {results['processed_count']}/{results['total_rows']} rows",
            total_rows=results["total_rows"],
            processed_count=results["processed_count"],
            failed_count=results["failed_count"],
            errors=results["errors"] if results["errors"] else None
        )
    except Exception as e:
        logger.error(f"Batch background image generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/batch-generate-videos", response_model=VideoGenerationBatchResponse)
async def batch_generate_videos(limit: Optional[int] = None):
    """
    Batch generate and upload videos for rows with required fields but no video
    
    - **limit**: Optional limit on number of rows to process
    
    Required fields: ai_script, background_image_url
    Optional fields: character_a, character_b, incident_image_1/2/3, background_audio_url
    
    Generates videos and uploads to Supabase Storage
    """
    try:
        results = video_processor.process_batch(limit)
        
        return VideoGenerationBatchResponse(
            success=True,
            message=f"Batch video generation complete. Processed {results['processed_count']}/{results['total_rows']} rows",
            total_rows=results["total_rows"],
            processed_count=results["processed_count"],
            failed_count=results["failed_count"],
            errors=results["errors"] if results["errors"] else None
        )
    except Exception as e:
        logger.error(f"Batch video generation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

# ============================================================================
# RUN SERVER
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        log_level="info"
    )
