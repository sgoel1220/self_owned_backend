"""
AI Content Automation - Script Generation Module
FastAPI server with Supabase integration for viral horror/mystery content creation
Includes background image prompt generation and image generation with Supabase storage upload
Version: 3.0.0
"""

import os
import logging
from typing import List, Optional, Dict
from datetime import datetime
import requests
from supabase import create_client, Client
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import json
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

import base64
from io import BytesIO

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
    SUPABASE_TABLE: str = 'youtube_scripts'
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


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================


# Pass middleware to FastAPI constructor
app = FastAPI(
    title="AI Script Generator API",
    description="API for generating cryptic horror/mystery scripts",
    version="3.0.0",
    docs_url="/apidocs",
    redoc_url="/redoc",

)

# Define the exact URLs your frontend runs on
origins = [
    "*",
    "http://localhost:3000",    # Example: Local React/Vue/Svelte dev server
    "http://127.0.0.1:3000",   #
    "http://localhost:8080",    # Another common dev port
    "https://your-production-site.com" # Your deployed frontend URL
]

# Add the CORS middleware directly to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # <--- Use the specific list
    allow_credentials=True,    # <--- This is now safe
    allow_methods=["*"],       # You can also get specific, e.g., ["GET", "POST"]
    allow_headers=["*"],       # You can also get specific, e.g., ["Content-Type"]
)


processor = ScriptProcessor()
bg_prompt_processor = BackgroundPromptProcessor()


processor = ScriptProcessor()
bg_prompt_processor = BackgroundPromptProcessor()
bg_image_processor = BackgroundImageProcessor()


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["Health"])
async def root():
    """Root endpoint - API health check"""
    return {
        "message": "AI Script Generator API",
        "version": "3.0.0",
        "status": "running",
        "docs": "/apidocs",
        "endpoints": {
            "script_generation": "/generate-script",
            "story_management": "/save-story",
            "batch_scripts": "/generate-scripts-batch-background",
            "background_prompts": "/generate-background-prompts",
            "background_images": "/generate-background-images"
        },
        "note": "Generate scripts, background prompts, and images for viral horror/mystery content"
    }


@app.post(
    "/generate-script",
    response_model=ScriptResponse,
    status_code=status.HTTP_200_OK,
    tags=["Script Generation"],
    summary="Generate Script from Story Input",
    description="Converts a story text into a cryptic horror/mystery dialogue script. Script is returned but NOT saved to database. Only the story summary is saved."
)
async def generate_script_endpoint(request_data: ScriptRequest):
    """
    Generate a cryptic horror/mystery script from story input.
    
    **Important**: The generated script is returned in the response but is NOT saved to the database.
    Only the story summary and metadata are saved to the database.
    
    - **story_input**: The story text to convert (minimum 10 characters)
    - **save_to_db**: Whether to save the story summary to database (default: true). Note: Only the summary is saved, not the script.
    - **character_a**: Optional custom name for first character
    - **character_b**: Optional custom name for second character
    """
    try:
        result = processor.process_single(request_data)
        status_code = status.HTTP_200_OK if result.success else status.HTTP_500_INTERNAL_SERVER_ERROR
        return JSONResponse(
            status_code=status_code,
            content=result.model_dump(exclude_none=True)
        )
    except Exception as e:
        logger.error(f"Error in generate_script endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/save-story",
    response_model=StoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Story Management"],
    summary="Save Story to Database",
    description="Save a story summary to the database without generating a script"
)
async def save_story_endpoint(request_data: StoryRequest):
    """
    Save a story summary to the database without script generation.
    
    - **article_summary**: The story text to save (minimum 10 characters)
    """
    try:
        story_record = processor.db.save_story_only(
            article_summary=request_data.article_summary,
        )
        
        response = StoryResponse(
            success=True,
            message="Story saved successfully",
            record_id=story_record.get("id")
        )
        return response
        
    except StoryInsertionError as e:
        logger.error(f"Story insertion error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/generate-scripts-batch-background",
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Script Generation"],
    summary="Generate Scripts in Background",
    description="Starts background processing of all rows where ai_script is empty"
)
async def generate_scripts_batch_background(
    background_tasks: BackgroundTasks,
    limit: Optional[int] = None
):
    """
    Start background script generation for all rows where ai_script is empty.
    
    This endpoint immediately returns a response while processing continues in the background.
    Ideal for large datasets.
    
    - **limit**: Optional limit on number of rows to process
    
    **Note**: Check logs or database to monitor progress.
    """
    try:
        background_tasks.add_task(processor.process_batch, limit)
        
        return {
            "success": True,
            "message": "Batch script generation started in background",
            "note": "Check server logs or database for progress updates"
        }
        
    except Exception as e:
        logger.error(f"Error starting background batch: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/generate-background-prompts",
    response_model=BackgroundPromptBatchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Background Prompt Generation"],
    summary="Generate Background Prompts for All Null Rows",
    description="Processes all rows where background_image_prompt is null and generates cinematic background image prompts"
)
async def generate_background_prompts_endpoint(
    background_tasks: BackgroundTasks,
    limit: Optional[int] = None,
    run_in_background: bool = False
):
    """
    Generate background image prompts for all rows where background_image_prompt is null.
    
    This endpoint can run synchronously (default) or asynchronously in the background.
    
    **Parameters:**
    - **limit**: Optional limit on number of rows to process
    - **run_in_background**: If True, runs asynchronously and returns immediately
    """
    try:
        if run_in_background:
            background_tasks.add_task(bg_prompt_processor.process_batch, limit)
            return BackgroundPromptBatchResponse(
                success=True,
                message="Background prompt generation started in background",
                total_rows=0,
                processed_count=0,
                failed_count=0
            )
        else:
            results = bg_prompt_processor.process_batch(limit)
            return BackgroundPromptBatchResponse(
                success=True,
                message="Background prompt generation completed",
                total_rows=results["total_rows"],
                processed_count=results["processed_count"],
                failed_count=results["failed_count"],
                errors=results.get("errors")
            )
        
    except Exception as e:
        logger.error(f"Error in background prompt generation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@app.post(
    "/generate-background-images",
    response_model=BackgroundImageBatchResponse,
    status_code=status.HTTP_200_OK,
    tags=["Background Image Generation"],
    summary="Generate and Upload Background Images",
    description="Generates images from background_image_prompt and uploads to Supabase Storage for all rows where background_image_url is null"
)
async def generate_background_images_endpoint(
    background_tasks: BackgroundTasks,
    limit: Optional[int] = None,
    run_in_background: bool = False
):
    """
    Generate background images and upload to Supabase Storage.
    
    This endpoint processes rows where background_image_prompt exists but background_image_url is null.
    It generates the image using Hugging Face API and uploads to Supabase Storage.
    
    **Parameters:**
    - **limit**: Optional limit on number of rows to process (default: all ready rows)
    - **run_in_background**: If true, processing happens in background and returns immediately (default: false)
    
    **Process:**
    1. Fetches rows with background_image_prompt but no background_image_url
    2. Generates image from prompt using Hugging Face Flux model
    3. Uploads image to Supabase Storage bucket: youtube-automation
    4. Updates database with public image URL
    
    **Note:** This process can take time depending on the number of images to generate.
    """
    try:
        if run_in_background:
            background_tasks.add_task(bg_image_processor.process_batch, limit)
            
            return BackgroundImageBatchResponse(
                success=True,
                message="Background image generation started in background. Check logs for progress.",
                total_rows=0,
                processed_count=0,
                failed_count=0
            )
        else:
            results = bg_image_processor.process_batch(limit)
            
            response = BackgroundImageBatchResponse(
                success=results["processed_count"] > 0 or results["total_rows"] == 0,
                message=f"Processed {results['processed_count']}/{results['total_rows']} rows successfully",
                total_rows=results["total_rows"],
                processed_count=results["processed_count"],
                failed_count=results["failed_count"],
                errors=results.get("errors") if results.get("errors") else None
            )
            
            return response
        
    except Exception as e:
        logger.error(f"Error in generate_background_images endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn

        # Add CORS middleware

    
    uvicorn.run(
        app,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        log_level="info"
    )
