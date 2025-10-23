"""Custom exceptions for the application"""

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
