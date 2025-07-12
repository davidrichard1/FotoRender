"""
Favorite Prompt Database Service for Foto Render
Manages favorite prompts with R2 image storage integration
"""

import logging
from typing import List, Dict, Optional, Any
from datetime import datetime
from sqlalchemy import select, update, delete, func, or_
from sqlalchemy.orm import selectinload
from .sqlalchemy_client import get_db_client, DatabaseClient
from .sqlalchemy_models import AiModel, FavoritePrompt
from storage import R2StorageClient, ImageProcessor

logger = logging.getLogger(__name__)

class FavoritePromptService:
    """
    Service for managing favorite prompts with R2 image storage
    """
    
    def __init__(self, db_client: Optional[DatabaseClient] = None):
        self.db_client = db_client
        self.r2_client = R2StorageClient()
        self.image_processor = ImageProcessor()
    
    async def _get_client(self):
        if self.db_client is None:
            self.db_client = await get_db_client()
        return self.db_client
    
    async def create_favorite_prompt(self, prompt_data: Dict[str, Any], 
                                   image_data: bytes) -> Dict[str, Any]:
        """
        Create a new favorite prompt with image upload to R2
        
        Args:
            prompt_data: Prompt information and generation parameters
            image_data: Generated image data to upload
        
        Returns:
            Created favorite prompt information
        """
        try:
            client = await self._get_client()
            
            # Upload image to R2
            upload_result = self.r2_client.upload_image(
                image_data=image_data,
                folder="prompts",
                content_type="image/png",
                generate_thumbnail=True
            )
            
            # Connect to model if provided
            model_id = None
            if prompt_data.get("model_filename"):
                async with client.get_session() as session:
                    result = await session.execute(
                        select(AiModel).where(AiModel.filename == prompt_data["model_filename"])
                    )
                    model = result.scalar_one_or_none()
                    if model:
                        model_id = model.id
            
            # Create database record
            async with client.get_session() as session:
                favorite = FavoritePrompt(
                    title=prompt_data["title"],
                    prompt=prompt_data["prompt"],
                    negative_prompt=prompt_data.get("negative_prompt"),
                    description=prompt_data.get("description"),
                    width=prompt_data["width"],
                    height=prompt_data["height"],
                    steps=prompt_data["steps"],
                    guidance_scale=prompt_data["guidance_scale"],
                    seed=prompt_data["seed"],
                    sampler=prompt_data["sampler"],
                    clip_skip=prompt_data.get("clip_skip"),
                    loras_used=prompt_data.get("loras_used"),
                    
                    # R2 image information
                    image_r2_key=upload_result["r2_key"],
                    image_r2_url=upload_result["r2_url"],
                    image_cdn_url=upload_result["cdn_url"],
                    image_width=upload_result["width"],
                    image_height=upload_result["height"],
                    image_size_bytes=upload_result["size_bytes"],
                    image_format="png",
                    
                    # Thumbnail information
                    thumbnail_r2_key=upload_result["thumbnail"]["r2_key"] if upload_result["thumbnail"] else None,
                    thumbnail_cdn_url=upload_result["thumbnail"]["cdn_url"] if upload_result["thumbnail"] else None,
                    
                    # Organization
                    tags=prompt_data.get("tags", []),
                    category=prompt_data.get("category"),
                    is_public=prompt_data.get("is_public", False),
                    is_featured=False,
                    
                    # Model relationship
                    model_id=model_id
                )
                
                session.add(favorite)
                await session.commit()
                await session.refresh(favorite)
            
            logger.info(f"✅ Created favorite prompt: {favorite.title}")
            
            return {
                "id": favorite.id,
                "title": favorite.title,
                "prompt": favorite.prompt,
                "image_url": favorite.image_cdn_url or favorite.image_r2_url,
                "thumbnail_url": favorite.thumbnail_cdn_url,
                "r2_key": favorite.image_r2_key,
                "created_at": favorite.created_at.isoformat(),
                "tags": favorite.tags,
                "category": favorite.category,
                "is_public": favorite.is_public,
                "usage_count": favorite.usage_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to create favorite prompt: {e}")
            # Clean up uploaded image if database creation failed
            try:
                if 'upload_result' in locals():
                    self.r2_client.delete_image(upload_result["r2_key"])
            except:
                pass
            raise
    
    async def get_favorite_prompts(self, 
                                 limit: int = 50,
                                 offset: int = 0,
                                 category: Optional[str] = None,
                                 tags: Optional[List[str]] = None,
                                 is_public: Optional[bool] = None,
                                 search: Optional[str] = None) -> Dict[str, Any]:
        """
        Get favorite prompts with filtering and pagination
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            category: Filter by category
            tags: Filter by tags (any match)
            is_public: Filter by public status
            search: Search in title, prompt, and description
        
        Returns:
            Dict with prompts and pagination info
        """
        try:
            client = await self._get_client()
            
            async with client.get_session() as session:
                # Build query
                query = select(FavoritePrompt).options(selectinload(FavoritePrompt.model))
                
                # Apply filters
                if category:
                    query = query.where(FavoritePrompt.category == category)
                if is_public is not None:
                    query = query.where(FavoritePrompt.is_public == is_public)
                if tags:
                    query = query.where(FavoritePrompt.tags.contains(tags))
                if search:
                    search_filter = or_(
                        FavoritePrompt.title.ilike(f"%{search}%"),
                        FavoritePrompt.prompt.ilike(f"%{search}%"),
                        FavoritePrompt.description.ilike(f"%{search}%")
                    )
                    query = query.where(search_filter)
                
                # Order and pagination
                query = query.order_by(FavoritePrompt.created_at.desc()).offset(offset).limit(limit)
                
                result = await session.execute(query)
                prompts = result.scalars().all()
                
                # Get total count
                count_query = select(func.count(FavoritePrompt.id))
                if category:
                    count_query = count_query.where(FavoritePrompt.category == category)
                if is_public is not None:
                    count_query = count_query.where(FavoritePrompt.is_public == is_public)
                if tags:
                    count_query = count_query.where(FavoritePrompt.tags.contains(tags))
                if search:
                    count_query = count_query.where(search_filter)
                
                count_result = await session.execute(count_query)
                total_count = count_result.scalar() or 0
            
            # Format response
            formatted_prompts = []
            for prompt in prompts:
                formatted_prompts.append({
                    "id": prompt.id,
                    "title": prompt.title,
                    "prompt": prompt.prompt,
                    "negative_prompt": prompt.negative_prompt,
                    "description": prompt.description,
                    "image_url": prompt.image_cdn_url or prompt.image_r2_url,
                    "thumbnail_url": prompt.thumbnail_cdn_url,
                    "width": prompt.width,
                    "height": prompt.height,
                    "steps": prompt.steps,
                    "guidance_scale": prompt.guidance_scale,
                    "seed": prompt.seed,
                    "sampler": prompt.sampler,
                    "clip_skip": prompt.clip_skip,
                    "loras_used": prompt.loras_used,
                    "model": {
                        "filename": prompt.model.filename,
                        "display_name": prompt.model.display_name
                    } if prompt.model else None,
                    "tags": prompt.tags,
                    "category": prompt.category,
                    "is_public": prompt.is_public,
                    "is_featured": prompt.is_featured,
                    "usage_count": prompt.usage_count,
                    "like_count": prompt.like_count,
                    "view_count": prompt.view_count,
                    "created_at": prompt.created_at.isoformat(),
                    "updated_at": prompt.updated_at.isoformat()
                })
            
            return {
                "prompts": formatted_prompts,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get favorite prompts: {e}")
            raise
    
    async def get_favorite_prompt_by_id(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific favorite prompt by ID
        
        Args:
            prompt_id: Prompt ID
        
        Returns:
            Prompt information or None if not found
        """
        try:
            client = await self._get_client()
            
            async with client.get_session() as session:
                # Get prompt with model
                result = await session.execute(
                    select(FavoritePrompt).where(FavoritePrompt.id == prompt_id)
                    .options(selectinload(FavoritePrompt.model))
                )
                prompt = result.scalar_one_or_none()
                
                if not prompt:
                    return None
                
                # Increment view count
                prompt.view_count += 1
                await session.commit()
            
            return {
                "id": prompt.id,
                "title": prompt.title,
                "prompt": prompt.prompt,
                "negative_prompt": prompt.negative_prompt,
                "description": prompt.description,
                "image_url": prompt.image_cdn_url or prompt.image_r2_url,
                "thumbnail_url": prompt.thumbnail_cdn_url,
                "width": prompt.width,
                "height": prompt.height,
                "steps": prompt.steps,
                "guidance_scale": prompt.guidance_scale,
                "seed": prompt.seed,
                "sampler": prompt.sampler,
                "clip_skip": prompt.clip_skip,
                "loras_used": prompt.loras_used,
                "model": {
                    "filename": prompt.model.filename,
                    "display_name": prompt.model.display_name
                } if prompt.model else None,
                "tags": prompt.tags,
                "category": prompt.category,
                "is_public": prompt.is_public,
                "is_featured": prompt.is_featured,
                "usage_count": prompt.usage_count,
                "like_count": prompt.like_count,
                "view_count": prompt.view_count,
                "created_at": prompt.created_at.isoformat(),
                "updated_at": prompt.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get favorite prompt: {e}")
            raise
    
    async def update_favorite_prompt(self, prompt_id: str, 
                                   updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Update a favorite prompt
        
        Args:
            prompt_id: Prompt ID
            updates: Fields to update
        
        Returns:
            Updated prompt information
        """
        try:
            client = await self._get_client()
            
            # Only allow certain fields to be updated
            allowed_fields = [
                "title", "description", "tags", "category", 
                "is_public", "is_featured"
            ]
            
            update_data = {k: v for k, v in updates.items() if k in allowed_fields}
            
            if not update_data:
                return await self.get_favorite_prompt_by_id(prompt_id)
            
            async with client.get_session() as session:
                result = await session.execute(
                    select(FavoritePrompt).where(FavoritePrompt.id == prompt_id)
                    .options(selectinload(FavoritePrompt.model))
                )
                prompt = result.scalar_one_or_none()
                
                if not prompt:
                    return None
                
                # Apply updates
                for key, value in update_data.items():
                    setattr(prompt, key, value)
                
                await session.commit()
                await session.refresh(prompt)
            
            logger.info(f"✅ Updated favorite prompt: {prompt.title}")
            
            return {
                "id": prompt.id,
                "title": prompt.title,
                "image_url": prompt.image_cdn_url or prompt.image_r2_url,
                "tags": prompt.tags,
                "category": prompt.category,
                "is_public": prompt.is_public,
                "is_featured": prompt.is_featured,
                "updated_at": prompt.updated_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to update favorite prompt: {e}")
            raise
    
    async def delete_favorite_prompt(self, prompt_id: str) -> bool:
        """
        Delete a favorite prompt and its associated R2 images
        
        Args:
            prompt_id: Prompt ID
        
        Returns:
            True if successful
        """
        try:
            client = await self._get_client()
            
            async with client.get_session() as session:
                # Get prompt info first
                result = await session.execute(
                    select(FavoritePrompt).where(FavoritePrompt.id == prompt_id)
                )
                prompt = result.scalar_one_or_none()
                
                if not prompt:
                    return False
                
                # Delete from database first
                await session.delete(prompt)
                await session.commit()
                
                # Delete from R2 storage
                if prompt.image_r2_key:
                    self.r2_client.delete_image(
                        prompt.image_r2_key, 
                        delete_thumbnail=True
                    )
            
            logger.info(f"✅ Deleted favorite prompt: {prompt.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete favorite prompt: {e}")
            return False
    
    async def increment_usage(self, prompt_id: str) -> bool:
        """
        Increment usage count when prompt is reused
        
        Args:
            prompt_id: Prompt ID
        
        Returns:
            True if successful
        """
        try:
            client = await self._get_client()
            
            async with client.get_session() as session:
                result = await session.execute(
                    select(FavoritePrompt).where(FavoritePrompt.id == prompt_id)
                )
                prompt = result.scalar_one_or_none()
                
                if prompt:
                    prompt.usage_count += 1
                    prompt.last_used = datetime.now()
                    await session.commit()
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to increment usage: {e}")
            return False
    
    async def toggle_like(self, prompt_id: str, increment: bool = True) -> int:
        """
        Toggle like count for a prompt
        
        Args:
            prompt_id: Prompt ID
            increment: True to like, False to unlike
        
        Returns:
            New like count
        """
        try:
            client = await self._get_client()
            
            async with client.get_session() as session:
                result = await session.execute(
                    select(FavoritePrompt).where(FavoritePrompt.id == prompt_id)
                )
                prompt = result.scalar_one_or_none()
                
                if prompt:
                    prompt.like_count += 1 if increment else -1
                    await session.commit()
                    return prompt.like_count
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Failed to toggle like: {e}")
            return 0
    
    async def get_popular_prompts(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get most popular prompts based on usage and likes
        
        Args:
            limit: Maximum number of results
        
        Returns:
            List of popular prompts
        """
        try:
            client = await self._get_client()
            
            async with client.get_session() as session:
                query = (
                    select(FavoritePrompt)
                    .where(FavoritePrompt.is_public == True)
                    .options(selectinload(FavoritePrompt.model))
                    .order_by(
                        FavoritePrompt.like_count.desc(),
                        FavoritePrompt.usage_count.desc(),
                        FavoritePrompt.view_count.desc()
                    )
                    .limit(limit)
                )
                
                result = await session.execute(query)
                prompts = result.scalars().all()
            
            formatted_prompts = []
            for prompt in prompts:
                prompt_text = prompt.prompt
                if len(prompt_text) > 200:
                    prompt_text = prompt_text[:200] + "..."
                    
                formatted_prompts.append({
                    "id": prompt.id,
                    "title": prompt.title,
                    "prompt": prompt_text,
                    "image_url": prompt.image_cdn_url or prompt.image_r2_url,
                    "thumbnail_url": prompt.thumbnail_cdn_url,
                    "tags": prompt.tags,
                    "category": prompt.category,
                    "usage_count": prompt.usage_count,
                    "like_count": prompt.like_count,
                    "view_count": prompt.view_count,
                    "created_at": prompt.created_at.isoformat()
                })
            
            return formatted_prompts
            
        except Exception as e:
            logger.error(f"❌ Failed to get popular prompts: {e}")
            return [] 