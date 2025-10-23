"""AI generators for scripts, prompts, and images"""

import logging
import json
from typing import List, Optional

import requests

from config import config
from models import DialogueLine
from exceptions import ImageGenerationError

logger = logging.getLogger(__name__)

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