"""Supabase database manager"""

import logging
from typing import List, Optional
from datetime import datetime

from supabase import create_client, Client

from config import config
from models import DialogueLine
from exceptions import StoryInsertionError, StorageUploadError

logger = logging.getLogger(__name__)

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

    def get_productions(self, limit: int = 100, offset: int = 0) -> List[dict]:
        """
        Fetch a paginated list of all productions, ordered by creation date.
        
        Args:
            limit: The maximum number of records to return.
            offset: The starting position.
        
        Returns:
            A list of production records.
        """
        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Failed to fetch productions: {e}")
            raise

    def get_production_by_id(self, record_id: int) -> Optional[dict]:
        """
        Fetch a single production by its ID.
        
        Args:
            record_id: The ID of the record to fetch.
        
        Returns:
            A single production record or None if not found.
        """
        try:
            response = (
                self.client.table(self.table)
                .select("*")
                .eq("id", record_id)
                .single()
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Failed to fetch production {record_id}: {e}")
            raise

    def update_production(self, record_id: int, data: dict) -> Optional[dict]:
        """
        Update a production record.
        
        Args:
            record_id: The ID of the record to update.
            data: A dictionary of fields to update.
        
        Returns:
            The updated record or None if not found.
        """
        try:
            response = (
                self.client.table(self.table)
                .update(data)
                .eq("id", record_id)
                .execute()
            )
            if response.data:
                logger.info(f"Successfully updated production {record_id}")
                return response.data[0]
            return None
        except Exception as e:
            logger.error(f"Failed to update production {record_id}: {e}")
            raise
