"""
Additional API endpoints for Favorite Prompts with R2 Storage
These endpoints should be added to main.py
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
import base64
import logging
from database.prompt_service import FavoritePromptService

logger = logging.getLogger(__name__)

# Pydantic models for API
class CreateFavoritePromptRequest(BaseModel):
    title: str
    prompt: str
    negative_prompt: Optional[str] = None
    description: Optional[str] = None
    
    # Generation parameters
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    sampler: str
    clip_skip: Optional[int] = None
    model_filename: Optional[str] = None
    loras_used: Optional[dict] = None
    
    # Organization
    tags: List[str] = []
    category: Optional[str] = None
    is_public: bool = False
    
    # Base64 encoded image data
    image_data: str  # Base64 encoded image

class UpdateFavoritePromptRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[str] = None
    is_public: Optional[bool] = None

class FavoritePromptResponse(BaseModel):
    id: str
    title: str
    prompt: str
    negative_prompt: Optional[str]
    description: Optional[str]
    image_url: str
    thumbnail_url: Optional[str]
    
    # Generation parameters
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    sampler: str
    clip_skip: Optional[int]
    model: Optional[dict]
    loras_used: Optional[dict]
    
    # Organization & Analytics
    tags: List[str]
    category: Optional[str]
    is_public: bool
    is_featured: bool
    usage_count: int
    like_count: int
    view_count: int
    created_at: str
    updated_at: str

class FavoritePromptsListResponse(BaseModel):
    prompts: List[FavoritePromptResponse]
    total_count: int
    limit: int
    offset: int
    has_more: bool

# Initialize service
prompt_service = FavoritePromptService()

# API Endpoints to add to main.py
"""
@app.post("/favorite-prompts", response_model=FavoritePromptResponse)
async def create_favorite_prompt(request: CreateFavoritePromptRequest):
    \"\"\"Create a new favorite prompt with image upload\"\"\"
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_data)
        
        # Prepare prompt data
        prompt_data = {
            "title": request.title,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "description": request.description,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "sampler": request.sampler,
            "clip_skip": request.clip_skip,
            "model_filename": request.model_filename,
            "loras_used": request.loras_used,
            "tags": request.tags,
            "category": request.category,
            "is_public": request.is_public
        }
        
        # Create favorite prompt
        result = await prompt_service.create_favorite_prompt(prompt_data, image_bytes)
        
        return FavoritePromptResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to create favorite prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/favorite-prompts", response_model=FavoritePromptsListResponse)
async def get_favorite_prompts(
    limit: int = 50,
    offset: int = 0,
    category: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated tags
    is_public: Optional[bool] = None,
    search: Optional[str] = None
):
    \"\"\"Get favorite prompts with filtering and pagination\"\"\"
    try:
        # Parse tags if provided
        tag_list = None
        if tags:
            tag_list = [tag.strip() for tag in tags.split(",")]
        
        result = await prompt_service.get_favorite_prompts(
            limit=limit,
            offset=offset,
            category=category,
            tags=tag_list,
            is_public=is_public,
            search=search
        )
        
        return FavoritePromptsListResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to get favorite prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/favorite-prompts/{prompt_id}", response_model=FavoritePromptResponse)
async def get_favorite_prompt(prompt_id: str):
    \"\"\"Get a specific favorite prompt by ID\"\"\"
    try:
        result = await prompt_service.get_favorite_prompt_by_id(prompt_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Favorite prompt not found")
        
        return FavoritePromptResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get favorite prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/favorite-prompts/{prompt_id}", response_model=FavoritePromptResponse)
async def update_favorite_prompt(prompt_id: str, request: UpdateFavoritePromptRequest):
    \"\"\"Update a favorite prompt\"\"\"
    try:
        updates = {k: v for k, v in request.dict().items() if v is not None}
        
        result = await prompt_service.update_favorite_prompt(prompt_id, updates)
        
        if not result:
            raise HTTPException(status_code=404, detail="Favorite prompt not found")
        
        return FavoritePromptResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update favorite prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/favorite-prompts/{prompt_id}")
async def delete_favorite_prompt(prompt_id: str):
    \"\"\"Delete a favorite prompt and its images\"\"\"
    try:
        success = await prompt_service.delete_favorite_prompt(prompt_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Favorite prompt not found")
        
        return {"message": "Favorite prompt deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete favorite prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/favorite-prompts/{prompt_id}/like")
async def toggle_like_prompt(prompt_id: str, like: bool = True):
    \"\"\"Like or unlike a favorite prompt\"\"\"
    try:
        new_count = await prompt_service.toggle_like(prompt_id, like)
        return {
            "prompt_id": prompt_id,
            "liked": like,
            "like_count": new_count
        }
        
    except Exception as e:
        logger.error(f"Failed to toggle like: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/favorite-prompts/{prompt_id}/use")
async def mark_prompt_used(prompt_id: str):
    \"\"\"Mark a prompt as used (increment usage count)\"\"\"
    try:
        success = await prompt_service.increment_usage(prompt_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Favorite prompt not found")
        
        return {"message": "Usage count incremented"}
        
    except Exception as e:
        logger.error(f"Failed to mark prompt as used: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/popular-prompts")
async def get_popular_prompts(limit: int = 20):
    \"\"\"Get most popular public prompts\"\"\"
    try:
        prompts = await prompt_service.get_popular_prompts(limit=limit)
        return {"prompts": prompts}
        
    except Exception as e:
        logger.error(f"Failed to get popular prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/favorite-prompts/{prompt_id}/reuse")
async def get_prompt_for_reuse(prompt_id: str):
    \"\"\"Get prompt parameters for reusing in generation\"\"\"
    try:
        prompt = await prompt_service.get_favorite_prompt_by_id(prompt_id)
        
        if not prompt:
            raise HTTPException(status_code=404, detail="Favorite prompt not found")
        
        # Mark as used
        await prompt_service.increment_usage(prompt_id)
        
        # Return generation parameters
        return {
            "prompt": prompt["prompt"],
            "negative_prompt": prompt["negative_prompt"],
            "width": prompt["width"],
            "height": prompt["height"],
            "steps": prompt["steps"],
            "guidance_scale": prompt["guidance_scale"],
            "seed": prompt["seed"],
            "sampler": prompt["sampler"],
            "clip_skip": prompt["clip_skip"],
            "loras_used": prompt["loras_used"],
            "model": prompt["model"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get prompt for reuse: {e}")
        raise HTTPException(status_code=500, detail=str(e))
""" 