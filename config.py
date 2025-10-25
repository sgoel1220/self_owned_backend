"""Application configuration"""

from pydantic_settings import BaseSettings, SettingsConfigDict

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
    DEFAULT_FONT_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/font/Anton-Regular.ttf"
    DEFAULT_MASK_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/mask/mask.jpg"
    DEFAULT_CHARACTER_1_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/characters/Tradgirl.png"
    DEFAULT_CHARACTER_2_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/characters/Chad.png"
    DEFAULT_AUDIO_URL: str = "https://pksxiitjqxyyhnyzssei.supabase.co/storage/v1/object/public/youtube-automation/audio/suspense_dark.mp3"
    
    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=True
    )

config = Config()
