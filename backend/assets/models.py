"""
Model management module for Foto Render
Replaces model-related functions from main.py with database-driven approach
"""

import os
import logging
from typing import List, Dict, Optional, Any
from database.services import ModelManager as DatabaseModelManager

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Model manager that provides database-driven model management
    Replaces get_available_models() and related functions from main.py
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.db_manager = DatabaseModelManager(models_dir)
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available models from database with filesystem fallback
        Replaces get_available_models() from main.py
        """
        return await self.db_manager.get_available_models()
    
    async def sync_models_from_filesystem(self) -> Dict[str, Any]:
        """
        Sync models from filesystem to database
        Should be called on startup to ensure database is up to date
        """
        return await self.db_manager.sync_models()
    
    async def update_model_usage(self, filename: str):
        """
        Update model usage statistics
        Call this when a model is used for generation
        """
        await self.db_manager.update_usage(filename)
    
    def get_model_base_classification(self, model_filename: str) -> str:
        """
        Legacy function - now redirects to database classification
        Kept for backwards compatibility
        """
        # This function is now handled by the database service
        # but kept for compatibility with existing code
        return self.db_manager._classify_base_model_legacy(model_filename)
    
    async def get_models_by_base_type(self, base_type: str) -> List[Dict[str, Any]]:
        """
        Get models filtered by base type
        New database-driven functionality
        """
        all_models = await self.get_available_models()
        return [model for model in all_models if model.get("base_model") == base_type]
    
    async def get_model_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific model by filename
        """
        all_models = await self.get_available_models()
        for model in all_models:
            if model["filename"] == filename:
                return model
        return None
    
    async def get_nsfw_models(self, include_nsfw: bool = False) -> List[Dict[str, Any]]:
        """
        Get models filtered by NSFW content
        """
        all_models = await self.get_available_models()
        if include_nsfw:
            return all_models
        else:
            return [model for model in all_models if not model.get("is_nsfw", False)]
    
    async def get_recommended_model(self, preference: str = "quality") -> Optional[Dict[str, Any]]:
        """
        Get recommended model based on preference
        """
        models = await self.get_available_models()
        
        if not models:
            return None
        
        # Preference-based selection
        if preference == "quality":
            # Prefer NoobAI models for quality
            for model in models:
                if model.get("base_model") == "noobai" and not model.get("is_nsfw", False):
                    return model
        elif preference == "fast":
            # Return first available model
            return models[0]
        
        # Fallback to first model
        return models[0] 