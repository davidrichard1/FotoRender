"""
LoRA management module for Foto Render
Replaces LoRA-related functions from main.py with database-driven approach
"""

import os
import logging
from typing import List, Dict, Optional, Any
from sqlalchemy import select, func
from database.sqlalchemy_client import get_db_client, LoraService
from database.sqlalchemy_models import Lora, LoraCompatibility, AiModel

logger = logging.getLogger(__name__)

class LoraManager:
    """
    LoRA manager that provides database-driven LoRA management
    Replaces get_available_loras() and related functions from main.py
    """
    
    def __init__(self, loras_dir: str = "loras"):
        self.loras_dir = loras_dir
        self.db_client = None
    
    async def _get_service(self) -> LoraService:
        """Get LoRA service with database client"""
        if not self.db_client:
            self.db_client = await get_db_client()
        return LoraService(self.db_client)
    
    async def get_available_loras(self, filter_by_current_model: bool = False, 
                                 current_model_filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available LoRAs from database with filesystem fallback
        Replaces get_available_loras() from main.py
        """
        try:
            service = await self._get_service()
            
            # Get all LoRAs from database
            loras = await service.get_all_loras(
                active_only=True,
                compatible_with_model=current_model_filename if filter_by_current_model else None
            )
            
            # Convert to dictionary format with correct attribute names
            return [
                {
                    "id": lora.id,
                    "filename": lora.filename,
                    "display_name": lora.display_name,
                    "description": lora.description,
                    "category": lora.category.value if hasattr(lora.category, 'value') else str(lora.category) if lora.category else "GENERAL",
                    "is_nsfw": lora.is_nsfw,
                    "trigger_words": lora.trigger_words or [],
                    "default_scale": lora.default_scale,
                    "recommended_min": lora.recommended_min,
                    "recommended_max": lora.recommended_max,
                    "min_scale": lora.min_scale,
                    "max_scale": lora.max_scale,
                    "file_path": lora.file_path,
                    "file_size": lora.file_size,
                    "usage_count": lora.usage_count,
                    "last_used": lora.last_used.isoformat() if lora.last_used else None,
                    "tags": lora.tags or [],
                    "source": lora.source
                }
                for lora in loras
            ]
        except Exception as e:
            logger.error(f"Error getting LoRAs from database: {e}")
            return []
    
    async def sync_loras_from_filesystem(self) -> Dict[str, Any]:
        """
        Sync LoRAs from filesystem to database
        Should be called on startup to ensure database is up to date
        """
        # This would need to be implemented in the LoraService
        logger.warning("sync_loras_from_filesystem not yet implemented")
        return {"synced": 0, "errors": 0}
    
    def get_lora_base_classification(self, lora_filename: str) -> List[str]:
        """
        Legacy function - basic classification for backwards compatibility with main.py
        """
        # Simple classification based on filename patterns
        classifications = []
        filename_lower = lora_filename.lower()
        
        # Character/anatomy types
        if any(pattern in filename_lower for pattern in ['anthro', 'furry', 'animal']):
            classifications.append('anthro')
        
        # Style types
        if any(pattern in filename_lower for pattern in ['realistic', 'photo', 'real']):
            classifications.append('realistic')
        elif any(pattern in filename_lower for pattern in ['anime', 'cartoon', 'art']):
            classifications.append('stylized')
        
        # NSFW detection - including main.py references
        if any(pattern in filename_lower for pattern in ['nsfw', 'nature', 'cock', 'adult', 'naturecock']):
            classifications.append('nsfw')
            
        return classifications if classifications else ['general']
    
    def get_lora_guidelines_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Legacy function - basic guidelines for backwards compatibility with main.py
        """
        classifications = self.get_lora_base_classification(filename)
        
        # Default guidelines based on classification
        guidelines = {
            "recommended_scale": 0.8,
            "min_scale": 0.1,
            "max_scale": 1.5,
            "trigger_words": [],
            "description": f"LoRA based on {filename}"
        }
        
        # Adjust based on classifications
        if 'nsfw' in classifications:
            guidelines["trigger_words"] = ["explicit content", "adult themes"]
            guidelines["recommended_scale"] = 0.6
        
        if 'realistic' in classifications:
            guidelines["recommended_scale"] = 0.9
            guidelines["trigger_words"].extend(["photorealistic", "detailed"])
        
        return guidelines
    
    async def get_loras_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get LoRAs filtered by category
        New database-driven functionality
        """
        all_loras = await self.get_available_loras()
        return [lora for lora in all_loras if lora.get("category", "").upper() == category.upper()]
    
    async def get_nsfw_loras(self, include_nsfw: bool = False) -> List[Dict[str, Any]]:
        """
        Get LoRAs filtered by NSFW content
        Database now handles NSFW classification including "n1-naturecock" references
        """
        all_loras = await self.get_available_loras()
        if include_nsfw:
            return all_loras
        else:
            return [lora for lora in all_loras if not lora.get("is_nsfw", False)]
    
    async def get_compatible_loras(self, model_base_type: str) -> List[Dict[str, Any]]:
        """
        Get LoRAs compatible with specific model base type
        """
        try:
            if not self.db_client:
                self.db_client = await get_db_client()
            
            async with self.db_client.get_session() as session:
                # Query LoRAs that are compatible with the specified base model type
                query = (
                    select(Lora)
                    .join(LoraCompatibility, Lora.id == LoraCompatibility.lora_id)
                    .join(AiModel, LoraCompatibility.model_id == AiModel.id)
                    .where(AiModel.base_model == model_base_type.upper())
                    .where(Lora.is_active == True)
                )
                
                result = await session.execute(query)
                loras = result.scalars().all()
                
                return [
                    {
                        "id": lora.id,
                        "filename": lora.filename,
                        "display_name": lora.display_name,
                        "description": lora.description,
                        "category": lora.category.value if hasattr(lora.category, 'value') else str(lora.category) if lora.category else "GENERAL",
                        "is_nsfw": lora.is_nsfw,
                        "trigger_words": lora.trigger_words or [],
                        "default_scale": lora.default_scale,
                        "recommended_min": lora.recommended_min,
                        "recommended_max": lora.recommended_max,
                        "file_path": lora.file_path,
                        "usage_count": lora.usage_count
                    }
                    for lora in loras
                ]
        except Exception as e:
            logger.error(f"Error getting compatible LoRAs: {e}")
            return []
    
    async def get_lora_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific LoRA by filename
        """
        all_loras = await self.get_available_loras()
        for lora in all_loras:
            if lora["filename"] == filename:
                return lora
        return None
    
    async def search_loras(self, query: str, search_fields: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search LoRAs by various fields
        """
        if search_fields is None:
            search_fields = ["display_name", "description", "trigger_words", "tags"]
        
        all_loras = await self.get_available_loras()
        query_lower = query.lower()
        matching_loras = []
        
        for lora in all_loras:
            for field in search_fields:
                if field in lora:
                    field_value = lora[field]
                    if isinstance(field_value, str) and query_lower in field_value.lower():
                        matching_loras.append(lora)
                        break
                    elif isinstance(field_value, list):
                        for item in field_value:
                            if isinstance(item, str) and query_lower in item.lower():
                                matching_loras.append(lora)
                                break
        
        return matching_loras
    
    async def get_popular_loras(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get most popular LoRAs based on usage statistics
        """
        all_loras = await self.get_available_loras()
        
        # Sort by usage count (descending)
        sorted_loras = sorted(
            all_loras, 
            key=lambda x: x.get("usage_count", 0), 
            reverse=True
        )
        
        return sorted_loras[:limit]
    
    async def update_lora_usage(self, filename: str):
        """
        Update LoRA usage statistics
        """
        try:
            if not self.db_client:
                self.db_client = await get_db_client()
            
            async with self.db_client.get_session() as session:
                result = await session.execute(
                    select(Lora).where(Lora.filename == filename)
                )
                lora = result.scalar_one_or_none()
                if lora:
                    lora.usage_count += 1
                    lora.last_used = func.now()
                    await session.commit()
        except Exception as e:
            logger.warning(f"Could not update usage for LoRA {filename}: {e}")
    
    async def get_lora_statistics(self) -> Dict[str, Any]:
        """
        Get LoRA usage and category statistics
        """
        all_loras = await self.get_available_loras()
        
        total_loras = len(all_loras)
        nsfw_count = len([l for l in all_loras if l.get("is_nsfw", False)])
        
        # Count by category
        category_counts = {}
        for lora in all_loras:
            category = lora.get("category", "GENERAL")
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Get most used
        most_used = sorted(
            all_loras, 
            key=lambda x: x.get("usage_count", 0), 
            reverse=True
        )[:5]
        
        return {
            "total_loras": total_loras,
            "nsfw_count": nsfw_count,
            "sfw_count": total_loras - nsfw_count,
            "category_counts": category_counts,
            "most_used": [{"filename": l["filename"], "usage_count": l.get("usage_count", 0)} for l in most_used]
        } 