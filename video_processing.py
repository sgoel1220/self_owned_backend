"""Video processing, including resource downloading and image manipulation"""

import os
import logging
import tempfile
import gc
import re
import io
from typing import List, Optional, Tuple

import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops, ImageEnhance
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

from config import config
from exceptions import (
    ResourceDownloadError, VideoGenerationError, AudioProcessingError
)

logger = logging.getLogger(__name__)

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
    def create_gradient_background(
        size: Tuple[int, int],
        start_color: Tuple[int, int, int, int],
        end_color: Tuple[int, int, int, int]
    ) -> Image.Image:
        """Create a vertical gradient background"""
        base = Image.new("RGBA", size, start_color)
        top = Image.new("RGBA", size, end_color)
        mask = Image.new("L", size)
        mask_data = []
        for y in range(size[1]):
            mask_data.extend([int(255 * (y / size[1]))] * size[0])
        mask.putdata(mask_data)
        base.paste(top, (0, 0), mask)
        return base

    @staticmethod
    def generate_text_image(
        text: str,
        font: ImageFont.FreeTypeFont,
        size: Tuple[int, int] = (720, 200),
        text_color: str = '#FFFFFF',
        start_color: Tuple[int, int, int, int] = (25, 25, 112, 180),
        end_color: Tuple[int, int, int, int] = (100, 149, 237, 180)
    ) -> Image.Image:
        """Generate text image with a clean, bold look and a subtle stroke."""
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        # --- Word Wrapping ---
        lines = []
        words = text.split()
        if not words:
            return img

        line = ""
        # A bit of padding on the sides
        max_width = size[0] - 80
        for word in words:
            test_line = f"{line}{word} "
            if d.textlength(test_line.strip(), font=font) <= max_width:
                line = test_line
            else:
                if line:
                    lines.append(line.strip())
                line = f"{word} "
        if line:
            lines.append(line.strip())

        # --- Text Rendering ---
        try:
            # Attempt to get size from a loaded font object
            font_size = font.size
        except AttributeError:
            # Fallback for default fonts
            font_size = 40

        line_height = font_size + 10  # Tighter line spacing
        total_text_height = line_height * len(lines)
        
        # Center the text block vertically
        y_start = (size[1] - total_text_height) // 2

        # --- Draw each line with a stroke ---
        stroke_width = 3  # A slightly thinner stroke
        stroke_fill = 'black'

        for i, line in enumerate(lines):
            line_width = d.textlength(line, font=font)
            x_text = (size[0] - line_width) // 2
            y_text = y_start + (i * line_height)

            # Draw stroke
            d.text((x_text - stroke_width, y_text), line, font=font, fill=stroke_fill)
            d.text((x_text + stroke_width, y_text), line, font=font, fill=stroke_fill)
            d.text((x_text, y_text - stroke_width), line, font=font, fill=stroke_fill)
            d.text((x_text, y_text + stroke_width), line, font=font, fill=stroke_fill)
            
            # Diagonal strokes for better coverage
            d.text((x_text - stroke_width, y_text - stroke_width), line, font=font, fill=stroke_fill)
            d.text((x_text + stroke_width, y_text - stroke_width), line, font=font, fill=stroke_fill)
            d.text((x_text - stroke_width, y_text + stroke_width), line, font=font, fill=stroke_fill)
            d.text((x_text + stroke_width, y_text + stroke_width), line, font=font, fill=stroke_fill)

            # Draw main text
            d.text((x_text, y_text), line, font=font, fill=text_color)

        return img

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
            font_size = 50
            
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
            temp_audiofile=None,
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
        print("clean")
        