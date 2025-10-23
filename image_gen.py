"""
Video Generation Module - Modularized and Clean
Generates cinematic horror/mystery short videos with dialogue, characters, and effects
Version: 2.0.0 (Refactored with Optimized Image Processing)
"""

import os
import tempfile
import gc
import re
import io
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageEnhance
from pydantic import BaseModel, Field, HttpUrl, field_validator
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class DialogueLine(BaseModel):
    """Single line of dialogue with character identifier"""
    character: str = Field(..., description="Character identifier (character_1 or character_2)")
    dialogue: str = Field(..., min_length=1, description="Dialogue text")
    
    @field_validator('character')
    @classmethod
    def validate_character(cls, v: str) -> str:
        if v not in ['character_1', 'character_2']:
            raise ValueError('character must be either "character_1" or "character_2"')
        return v


class ScriptData(BaseModel):
    """Script containing dialogue lines"""
    script: List[DialogueLine] = Field(..., min_items=1, description="List of dialogue lines")


class VideoGenerationRequest(BaseModel):
    """Request model for video generation"""
    vid_id: str = Field(..., description="Unique video identifier")
    script: ScriptData = Field(..., description="Script with dialogue lines")
    
    character_1: str = Field(..., description="URL to character 1 image")
    character_2: str = Field(..., description="URL to character 2 image")
    background_image: str = Field(..., description="URL to background image")
    
    incident_image_1: Optional[str] = Field(None, description="URL to first incident image")
    incident_image_2: Optional[str] = Field(None, description="URL to second incident image")
    incident_image_3: Optional[str] = Field(None, description="URL to third incident image")
    
    background_audio: Optional[str] = Field(None, description="URL to background audio file")
    mask_image: Optional[str] = Field(None, description="URL to film grain mask image")
    font: Optional[str] = Field(None, description="URL to custom font file")  # Changed from font_url to font
    
    # Video configuration
    canvas_width: int = Field(default=720, description="Video width in pixels")
    canvas_height: int = Field(default=1280, description="Video height in pixels")
    fps: int = Field(default=24, ge=15, le=60, description="Frames per second")
    scene_duration: float = Field(default=2.0, ge=1.0, le=5.0, description="Duration of each scene in seconds")
    font_size: int = Field(default=36, ge=20, le=72, description="Font size for dialogue text")


class VideoGenerationResponse(BaseModel):
    """Response model for video generation"""
    success: bool
    video_bytes: Optional[bytes] = None
    video_size_mb: Optional[float] = None
    duration_seconds: Optional[float] = None
    error: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ============================================================================
# EXCEPTIONS
# ============================================================================

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
# RESOURCE DOWNLOADER
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
# IMAGE PROCESSOR
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
    
    def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResponse:
        """Generate video from request data"""
        try:
            print(f"\n{'='*60}")
            print(f"Starting video generation for: {request.vid_id}")
            print(f"{'='*60}\n")
            
            # Download resources
            resources = self._download_resources(request)
            
            # Create video
            video_bytes, duration = self._create_video(request, resources)
            
            # Calculate size
            size_mb = len(video_bytes) / (1024 * 1024)
            
            print(f"\n{'='*60}")
            print(f"✅ Video generated successfully!")
            print(f"   Size: {size_mb:.2f}MB")
            print(f"   Duration: {duration:.1f}s")
            print(f"{'='*60}\n")
            
            return VideoGenerationResponse(
                success=True,
                video_bytes=video_bytes,
                video_size_mb=round(size_mb, 2),
                duration_seconds=duration
            )
            
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ Video generation failed: {str(e)}")
            print(f"{'='*60}\n")
            return VideoGenerationResponse(
                success=False,
                error=str(e)
            )
        finally:
            self._cleanup()
    
    def _download_resources(self, request: VideoGenerationRequest) -> dict:
        """Download all required resources with appropriate sizing"""
        print("📥 Downloading resources...\n")
        
        resources = {}
        
        # Download font
        if request.font:
            font, temp_path = self.downloader.download_font(request.font, request.font_size)
            if temp_path:
                self.temp_files.append(temp_path)
            resources['font'] = font
        else:
            resources['font'] = ImageFont.load_default()
            print("Using default font")
        
        # Download images with specific types for proper sizing
        print("Downloading character images...")
        resources['char1'] = self.downloader.download_image(request.character_1, "character")
        resources['char2'] = self.downloader.download_image(request.character_2, "character")
        
        print("Downloading background image...")
        resources['background'] = self.downloader.download_image(request.background_image, "background")
        
        # Download incident images
        print("Downloading incident images...")
        for i, img_url in enumerate([request.incident_image_1, request.incident_image_2, request.incident_image_3], 1):
            if img_url:
                resources[f'incident_{i}'] = self.downloader.download_image(img_url, "incident")
        
        # Download mask
        if request.mask_image:
            print("Downloading mask image...")
            resources['mask'] = self.downloader.download_image(request.mask_image, "mask")
        
        # Download audio
        if request.background_audio:
            print("Downloading background audio...")
            audio_path = self.downloader.download_audio(request.background_audio)
            self.temp_files.append(audio_path)
            resources['audio_path'] = audio_path
        
        print("\n✅ All resources downloaded successfully\n")
        return resources
    
    def _create_video(self, request: VideoGenerationRequest, resources: dict) -> Tuple[bytes, float]:
        """Create video from resources"""
        canvas_size = (request.canvas_width, request.canvas_height)
        script = request.script.script
        
        # Calculate duration
        num_scenes = len(script)
        num_incidents = sum([1 for k in resources if k.startswith('incident_')])
        video_duration = (num_scenes + num_incidents) * request.scene_duration
        
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
        for idx, dialogue in enumerate(script, 1):
            is_char1 = dialogue.character == 'character_1'
            
            # Select character
            if is_char1:
                char_img = char1_processed.copy()
            else:
                char_img = char2_processed.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Create character clip
            char_array = np.array(char_img)
            char_clip = ImageClip(char_array).with_duration(request.scene_duration).with_start(start_time)
            
            # Position character
            char_y_position = canvas_size[1] - char_img.height - 100
            if is_char1:
                char_clip = char_clip.with_position((50, char_y_position))
            else:
                char_clip = char_clip.with_position((canvas_size[0] - char_img.width - 50, char_y_position))
            
            # Generate text overlay
            text_img = self.processor.generate_text_image(
                dialogue.dialogue,
                resources['font'],
                size=(canvas_size[0] - 40, 200)
            )
            text_array = np.array(text_img)
            text_clip = ImageClip(text_array).with_duration(request.scene_duration).with_start(start_time).with_position(('center', 150))
            
            clips.extend([char_clip, text_clip])
            
            # Cleanup
            text_img.close()
            if not is_char1:
                char_img.close()
            del text_array, char_array
            
            print(f"   Scene {idx}/{num_scenes}: {dialogue.character} - \"{dialogue.dialogue[:30]}...\"")
            start_time += request.scene_duration
        
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
                    incident_clip = ImageClip(incident_array).with_duration(request.scene_duration).with_start(start_time).with_position('center')
                    clips.append(incident_clip)
                    
                    incident_processed.close()
                    del incident_array
                    
                    print(f"   Incident scene {i}")
                    start_time += request.scene_duration
        
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
            fps=request.fps,
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
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_video_from_dict(data: dict) -> VideoGenerationResponse:
    """Generate video from dictionary data"""
    try:
        request = VideoGenerationRequest(**data)
        generator = VideoGenerator()
        return generator.generate_video(request)
    except Exception as e:
        return VideoGenerationResponse(
            success=False,
            error=f"Invalid request data: {str(e)}"
        )


def generate_video_from_request(request: VideoGenerationRequest) -> VideoGenerationResponse:
    """Generate video from Pydantic request model"""
    generator = VideoGenerator()
    return generator.generate_video(request)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Example with Supabase URLs
    input_data = {
        "vid_id": "recOa9A6Lh9e6uhGQ",
        "script": {
            "script": [
                {"character": "character_1", "dialogue": "That sound... was it real?"},
                {"character": "character_2", "dialogue": "As real as the silence now."},
                {"character": "character_1", "dialogue": "It felt like tearing metal."},
                {"character": "character_2", "dialogue": "Or something breaking apart."},
                {"character": "character_1", "dialogue": "Where are the others?"},
                {"character": "character_2", "dialogue": "They went where we can't follow."},
                {"character": "character_1", "dialogue": "How did we get here?"},
                {"character": "character_2", "dialogue": "The jungle brought us, but it won't let us leave."}
            ]
        },
        "character_1": "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/characters/Tradgirl.png",
        "character_2": "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/characters/Chad.png",
        "background_image": "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/background/background_4_1761132994.png",
        "incident_image_1": "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/background/background_4_1761132994.png",
        "incident_image_2": "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/background/background_4_1761132994.png",
        "incident_image_3": "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/background/background_4_1761132994.png",
        "background_audio": "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/audio/suspense_dark.mp3",
        "mask_image": "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/mask/mask.jpg",
        "font": "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/font/Inter-VariableFont.ttf",
    }
    
    # Generate video
    result = generate_video_from_dict(input_data)
    
    if result.success:
        print(f"✅ Video generated successfully!")
        print(f"   Size: {result.video_size_mb}MB")
        print(f"   Duration: {result.duration_seconds}s")
        
        # Save to file (example)
        with open('output_video.mp4', 'wb') as f:
            f.write(result.video_bytes)
        print("   Saved to: output_video.mp4")
    else:
        print(f"❌ Video generation failed: {result.error}")
