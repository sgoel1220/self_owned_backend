"""Processors for orchestrating the video generation workflow"""

import logging
import json
from typing import Optional, Dict

from database import SupabaseManager
from ai_generators import ScriptGenerator, BackgroundImagePromptGenerator, BackgroundImageGenerator
from video_processing import VideoGenerator
from exceptions import StoryInsertionError, VideoGenerationError
from models import ScriptRequest, ScriptResponse, RetryStep

logger = logging.getLogger(__name__)


class ScriptProcessor:
    """Orchestrates the script generation workflow"""

    def __init__(self):
        self.db = SupabaseManager()
        self.generator = ScriptGenerator()

    def process_single(self, request_data: ScriptRequest) -> ScriptResponse:
        """
        Process a single story - generates script but saves ONLY story summary to DB
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

    def _process_row(self, row: dict) -> Optional[str]:
        record_id = row.get("id")
        article_summary = row.get("article_summary")

        if not article_summary:
            return "Missing article_summary"

        generated_script = self.generator.generate_with_retry(article_summary)

        if not generated_script:
            return "Failed to generate script"

        if self.db.update_script(record_id, generated_script):
            logger.info(f"Successfully processed record {record_id}")
            return None
        else:
            return "Failed to update database"

    def process_batch(self, limit: Optional[int] = None) -> Dict:
        """Process all rows with empty ai_script field"""
        results = {"total_rows": 0, "processed_count": 0, "failed_count": 0, "errors": []}
        try:
            empty_rows = self.db.get_empty_script_rows(limit)
            results["total_rows"] = len(empty_rows)
            if not empty_rows:
                logger.info("No rows with empty ai_script found")
                return results

            logger.info(f"Processing {len(empty_rows)} rows...")
            for row in empty_rows:
                error = self._process_row(row)
                if error:
                    results["failed_count"] += 1
                    results["errors"].append({"id": row.get("id"), "error": error})
                else:
                    results["processed_count"] += 1
            return results
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            results["errors"].append({"error": str(e)})
            return results

    def retry(self, record_id: int) -> Optional[str]:
        """Retry script generation for a single record."""
        row = self.db.get_production_by_id(record_id)
        if not row:
            return f"Record with id {record_id} not found."
        return self._process_row(row)


class BackgroundPromptProcessor:
    """Orchestrates background prompt generation workflow"""

    def __init__(self):
        self.db = SupabaseManager()
        self.generator = BackgroundImagePromptGenerator()

    def _process_row(self, row: dict) -> Optional[str]:
        record_id = row.get("id")
        article_summary = row.get("article_summary")
        ai_script = row.get("ai_script")

        if not article_summary:
            return "Missing article_summary"

        script_str = json.dumps(ai_script) if isinstance(ai_script, (list, dict)) else str(ai_script)
        background_prompt = self.generator.generate_with_retry(article_summary, script_str)

        if not background_prompt:
            return "Failed to generate background prompt"

        if self.db.update_background_prompt(record_id, background_prompt):
            logger.info(f"Successfully processed record {record_id}")
            return None
        else:
            return "Failed to update database"

    def process_batch(self, limit: Optional[int] = None) -> Dict:
        """Process all rows where background_image_prompt is null"""
        results = {"total_rows": 0, "processed_count": 0, "failed_count": 0, "errors": []}
        try:
            rows = self.db.get_rows_with_null_background_prompt(limit)
            results["total_rows"] = len(rows)
            if not rows:
                logger.info("No rows with null background_image_prompt found")
                return results

            logger.info(f"Processing {len(rows)} rows for background image prompts...")
            for row in rows:
                error = self._process_row(row)
                if error:
                    results["failed_count"] += 1
                    results["errors"].append({"id": row.get("id"), "error": error})
                else:
                    results["processed_count"] += 1
            return results
        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            results["errors"].append({"error": str(e)})
            return results

    def retry(self, record_id: int) -> Optional[str]:
        """Retry background prompt generation for a single record."""
        row = self.db.get_production_by_id(record_id)
        if not row:
            return f"Record with id {record_id} not found."
        return self._process_row(row)


class BackgroundImageProcessor:
    """Orchestrates background image generation and upload workflow"""

    def __init__(self):
        self.db = SupabaseManager()
        self.image_generator = BackgroundImageGenerator()

    def _process_row(self, row: dict) -> Optional[str]:
        record_id = row.get("id")
        prompt = row.get("background_image_prompt")

        if not prompt:
            return "Missing background_image_prompt"

        image_data = self.image_generator.generate_with_retry(prompt)

        if not image_data:
            return "Failed to generate image"

        try:
            filename = f"background_{record_id}_{int(datetime.utcnow().timestamp())}.png"
            image_url = self.db.upload_image_to_storage(image_data, filename)

            if self.db.update_background_image_url(record_id, image_url):
                logger.info(f"Successfully processed record {record_id}")
                return None
            else:
                return "Failed to update database with image URL"
        except Exception as e:
            return f"Storage or DB update failed: {e}"

    def process_batch(self, limit: Optional[int] = None) -> Dict:
        """
        Process all rows where background_image_prompt exists but background_image_url is null
        """
        results = {"total_rows": 0, "processed_count": 0, "failed_count": 0, "errors": []}
        try:
            rows = self.db.get_rows_with_null_background_image(limit)
            results["total_rows"] = len(rows)
            if not rows:
                logger.info("No rows ready for image generation found")
                return results

            logger.info(f"Processing {len(rows)} rows for background image generation...")
            for row in rows:
                error = self._process_row(row)
                if error:
                    results["failed_count"] += 1
                    results["errors"].append({"id": row.get("id"), "error": error})
                else:
                    results["processed_count"] += 1
            return results
        except Exception as e:
            logger.error(f"Batch image processing error: {e}")
            results["errors"].append({"error": str(e)})
            return results

    def retry(self, record_id: int) -> Optional[str]:
        """Retry background image generation for a single record."""
        row = self.db.get_production_by_id(record_id)
        if not row:
            return f"Record with id {record_id} not found."
        return self._process_row(row)


class VideoProcessor:
    """Orchestrates the video generation and upload workflow"""

    def __init__(self):
        self.db = SupabaseManager()
        self.video_generator = VideoGenerator()

    def _process_row(self, row: dict) -> Optional[str]:
        record_id = row.get("id")
        try:
            video_bytes, _ = self.video_generator.generate_video_from_db_row(row)
            filename = f"video_{record_id}_{int(datetime.utcnow().timestamp())}.mp4"
            video_url = self.db.upload_video_to_storage(video_bytes, filename)

            if self.db.update_video_url(record_id, video_url):
                logger.info(f"Successfully processed and uploaded video for record {record_id}")
                return None
            else:
                raise VideoGenerationError("Failed to update database with video URL")
        except Exception as e:
            logger.error(f"Failed to process video for record {record_id}: {e}")
            return str(e)

    def process_batch(self, limit: Optional[int] = None) -> Dict:
        """Process all rows ready for video generation"""
        results = {"total_rows": 0, "processed_count": 0, "failed_count": 0, "errors": []}
        try:
            rows = self.db.get_rows_ready_for_video(limit)
            results["total_rows"] = len(rows)
            if not rows:
                logger.info("No rows ready for video generation found")
                return results

            logger.info(f"Processing {len(rows)} rows for video generation...")
            for row in rows:
                error = self._process_row(row)
                if error:
                    results["failed_count"] += 1
                    results["errors"].append({"id": row.get("id"), "error": error})
                else:
                    results["processed_count"] += 1
            return results
        except Exception as e:
            logger.error(f"Batch video processing error: {e}")
            results["errors"].append({"error": str(e)})
            return results

    def retry(self, record_id: int) -> Optional[str]:
        """Retry video generation for a single record."""
        row = self.db.get_production_by_id(record_id)
        if not row:
            return f"Record with id {record_id} not found."
        return self._process_row(row)


class ProductionProcessor:
    """Handles high-level production tasks like retrying steps."""

    def __init__(self):
        self.db = SupabaseManager()
        self.script_processor = ScriptProcessor()
        self.prompt_processor = BackgroundPromptProcessor()
        self.image_processor = BackgroundImageProcessor()
        self.video_processor = VideoProcessor()

    def retry_step(self, record_id: int, step: RetryStep) -> bool:
        """
        Retries a specific step in the production pipeline for a given record.
        """
        logger.info(f"Retrying step '{step.value}' for record {record_id}")
        error = None
        if step == RetryStep.SCRIPT:
            error = self.script_processor.retry(record_id)
        elif step == RetryStep.BACKGROUND_PROMPT:
            error = self.prompt_processor.retry(record_id)
        elif step == RetryStep.BACKGROUND_IMAGE:
            error = self.image_processor.retry(record_id)
        elif step == RetryStep.VIDEO:
            error = self.video_processor.retry(record_id)

        if error:
            logger.error(f"Retry failed for step '{step.value}' on record {record_id}: {error}")
            return False
        
        logger.info(f"Successfully retried step '{step.value}' for record {record_id}")
        return True
