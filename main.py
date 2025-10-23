"""
AI Content Automation - Complete Video Generation System
FastAPI server with Supabase integration for viral horror/mystery content creation
Includes script generation, image generation, and video generation with storage upload
Version: 4.0.0
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from config import config
from models import (
    ScriptRequest, StoryRequest, ScriptResponse, StoryResponse, 
    BatchScriptResponse, BackgroundPromptBatchResponse, 
    BackgroundImageBatchResponse, VideoGenerationBatchResponse
)
from exceptions import StoryInsertionError
from database import SupabaseManager
from processors import (
    ScriptProcessor, BackgroundPromptProcessor, 
    BackgroundImageProcessor, VideoProcessor,
    ProductionProcessor
)
from models import (
    ProductionRequest, ProductionSummary, ProductionDetails, RetryStep, RetryResponse, ProductionUpdateRequest
)

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Content Automation API",
    description="Automated system for generating viral horror/mystery video content",
    version="4.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Helper Functions
# ============================================================================

def get_production_status(record: dict) -> str:
    if record.get("video_generated_url"):
        return "complete"
    if record.get("background_image_url"):
        return "video_pending"
    if record.get("background_image_prompt"):
        return "image_pending"
    if record.get("ai_script"):
        return "prompt_pending"
    if record.get("article_summary"):
        return "script_pending"
    return "unknown"

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/productions", response_model=list[ProductionSummary], tags=["Production"])
def get_productions(limit: int = 100, offset: int = 0):
    """Get a list of all video productions."""
    try:
        db = SupabaseManager()
        records = db.get_productions(limit, offset)
        return [ProductionSummary(
            id=r.get("id"),
            article_summary=r.get("article_summary"),
            status=get_production_status(r),
            created_at=r.get("created_at"),
            video_generated_url=r.get("video_generated_url")
        ) for r in records]
    except Exception as e:
        logger.error(f"Error in /productions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/productions/{production_id}", response_model=ProductionDetails, tags=["Production"])
def get_production_details(production_id: int):
    """Get the full details of a single video production."""
    try:
        db = SupabaseManager()
        record = db.get_production_by_id(production_id)
        if not record:
            raise HTTPException(status_code=404, detail="Production not found")
        
        return ProductionDetails(
            id=record.get("id"),
            article_summary=record.get("article_summary"),
            status=get_production_status(record),
            created_at=record.get("created_at"),
            video_generated_url=record.get("video_generated_url"),
            ai_script=record.get("ai_script"),
            background_image_prompt=record.get("background_image_prompt"),
            background_image_url=record.get("background_image_url"),
            character_a=record.get("character_a"),
            character_b=record.get("character_b"),
            background_image_1=record.get("background_image_1"),
            incident_image_1=record.get("incident_image_1"),
            incident_image_2=record.get("incident_image_2"),
            incident_image_3=record.get("incident_image_3"),
            background_audio_url=record.get("background_audio_url")
        )
    except Exception as e:
        logger.error(f"Error in /productions/{production_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/productions/{production_id}/status", tags=["Production"])
def get_production_status_endpoint(production_id: int):
    """Get the status of a single video production."""
    try:
        db = SupabaseManager()
        record = db.get_production_by_id(production_id)
        if not record:
            raise HTTPException(status_code=404, detail="Production not found")
        
        status = get_production_status(record)
        video_url = record.get("video_generated_url")
        
        return {"status": status, "video_url": video_url}
    except Exception as e:
        logger.error(f"Error in /productions/{production_id}/status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/productions", response_model=ProductionSummary, tags=["Production"])
def create_production(request: ProductionRequest):
    """Create a new video production."""
    try:
        db = SupabaseManager()
        record = db.save_story_only(request.story)
        
        def run_pipeline():
            try:
                ScriptProcessor().retry(record.get("id"))
                BackgroundPromptProcessor().retry(record.get("id"))
                BackgroundImageProcessor().retry(record.get("id"))
                VideoProcessor().retry(record.get("id"))
            except Exception as e:
                logger.error(f"Error in background pipeline for record {record.get('id')}: {e}", exc_info=True)

        run_pipeline()

        # Re-fetch the record to get the updated status and video URL
        record = db.get_production_by_id(record.get("id"))

        return ProductionSummary(
            id=record.get("id"),
            article_summary=record.get("article_summary"),
            status=get_production_status(record),
            created_at=record.get("created_at"),
            video_generated_url=record.get("video_generated_url")
        )
    except StoryInsertionError as e:
        logger.error(f"Story insertion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in /productions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/productions/{production_id}/retry/{step}", response_model=RetryResponse, tags=["Production"])
def retry_production_step(production_id: int, step: RetryStep):
    """Retry a specific step of the production pipeline."""
    
    processor = ProductionProcessor()
    processor.retry_step(production_id, step)

    return RetryResponse(success=True, message=f"Successfully retried step: {step.value}")

@app.put("/productions/{production_id}", response_model=ProductionDetails, tags=["Production"])
def update_production(production_id: int, request: ProductionUpdateRequest):
    """Update a production record."""
    try:
        db = SupabaseManager()
        updated_record = db.update_production(production_id, request.model_dump(exclude_unset=True))
        if not updated_record:
            raise HTTPException(status_code=404, detail="Production not found")
        
        return ProductionDetails(
            id=updated_record.get("id"),
            article_summary=updated_record.get("article_summary"),
            status=get_production_status(updated_record),
            created_at=updated_record.get("created_at"),
            video_generated_url=updated_record.get("video_generated_url"),
            ai_script=updated_record.get("ai_script"),
            background_image_prompt=updated_record.get("background_image_prompt"),
            background_image_url=updated_record.get("background_image_url")
        )
    except Exception as e:
        logger.error(f"Error in /productions/{production_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/generate-script", response_model=ScriptResponse, tags=["Generation"], deprecated=True)
def generate_script(request: ScriptRequest):
    """Generate a script from a story and optionally save the story summary to the database"""
    try:
        processor = ScriptProcessor()
        response = processor.process_single(request)
        return response
    except Exception as e:
        logger.error(f"Error in /generate-script: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/save-story", response_model=StoryResponse, tags=["Database"], deprecated=True)
def save_story(request: StoryRequest):
    """Save a story summary to the database"""
    try:
        db = SupabaseManager()
        record = db.save_story_only(request.article_summary)
        return StoryResponse(
            success=True,
            message="Story summary saved successfully",
            record_id=record.get('id')
        )
    except StoryInsertionError as e:
        logger.error(f"Story insertion failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in /save-story: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/batch/generate-scripts", response_model=BatchScriptResponse, tags=["Batch Processing"])
def batch_generate_scripts(limit: Optional[int] = None):
    """Process all rows with empty ai_script field"""
    try:
        processor = ScriptProcessor()
        result = processor.process_batch(limit)
        return BatchScriptResponse(
            success=True,
            message="Batch script generation completed",
            **result
        )
    except Exception as e:
        logger.error(f"Error in /batch/generate-scripts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/batch/generate-background-prompts", response_model=BackgroundPromptBatchResponse, tags=["Batch Processing"])
def batch_generate_background_prompts(limit: Optional[int] = None):
    """Process all rows where background_image_prompt is null"""
    try:
        processor = BackgroundPromptProcessor()
        result = processor.process_batch(limit)
        return BackgroundPromptBatchResponse(
            success=True,
            message="Batch background prompt generation completed",
            **result
        )
    except Exception as e:
        logger.error(f"Error in /batch/generate-background-prompts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/batch/generate-background-images", response_model=BackgroundImageBatchResponse, tags=["Batch Processing"])
def batch_generate_background_images(limit: Optional[int] = None):
    """Process all rows ready for background image generation"""
    try:
        processor = BackgroundImageProcessor()
        result = processor.process_batch(limit)
        return BackgroundImageBatchResponse(
            success=True,
            message="Batch background image generation completed",
            **result
        )
    except Exception as e:
        logger.error(f"Error in /batch/generate-background-images: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/batch/generate-videos", response_model=VideoGenerationBatchResponse, tags=["Batch Processing"])
def batch_generate_videos(limit: Optional[int] = None):
    """Process all rows ready for video generation"""
    try:
        processor = VideoProcessor()
        result = processor.process_batch(limit)
        return VideoGenerationBatchResponse(
            success=True,
            message="Batch video generation completed",
            **result
        )
    except Exception as e:
        logger.error(f"Error in /batch/generate-videos: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.FLASK_HOST, port=config.FLASK_PORT)