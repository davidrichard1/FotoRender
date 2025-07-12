"""
Upscaler management module for Foto Render
Replaces upscaler-related functions from main.py with database-driven approach
"""

import logging
from typing import List, Dict, Optional, Any
from database.services import UpscalerManager as DatabaseUpscalerManager

logger = logging.getLogger(__name__)

class UpscalerManager:
    """
    Upscaler manager that provides database-driven upscaler management
    Replaces get_available_upscalers() and related functions from main.py
    """
    
    def __init__(self, upscalers_dir: str = "upscalers"):
        self.upscalers_dir = upscalers_dir
        self.db_manager = DatabaseUpscalerManager(upscalers_dir)
    
    async def get_available_upscalers(self) -> List[Dict[str, Any]]:
        """
        Get available upscalers from database with filesystem fallback
        Replaces get_available_upscalers() from main.py
        """
        return await self.db_manager.get_available_upscalers()
    
    async def get_upscalers_by_scale(self, scale_factor: int) -> List[Dict[str, Any]]:
        """
        Get upscalers filtered by scale factor
        """
        all_upscalers = await self.get_available_upscalers()
        return [upscaler for upscaler in all_upscalers if upscaler.get("scale_factor") == scale_factor]
    
    async def get_upscaler_by_filename(self, filename: str) -> Optional[Dict[str, Any]]:
        """
        Get specific upscaler by filename
        """
        all_upscalers = await self.get_available_upscalers()
        for upscaler in all_upscalers:
            if upscaler["filename"] == filename:
                return upscaler
        return None 