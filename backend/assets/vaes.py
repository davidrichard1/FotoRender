"""
VAE management module for Foto Render
Replaces VAE-related functions from main.py with database-driven approach
"""

import logging
from typing import List, Dict, Optional, Any
from database.services import VaeManager as DatabaseVaeManager

logger = logging.getLogger(__name__)

class VaeManager:
    """
    VAE manager that provides database-driven VAE management
    Replaces get_available_vaes() and related functions from main.py
    """
    
    def __init__(self, vaes_dir: str = "vaes"):
        self.vaes_dir = vaes_dir
        self.db_manager = DatabaseVaeManager(vaes_dir)
    
    async def get_available_vaes(self) -> List[Dict[str, Any]]:
        """
        Get available VAEs from database with filesystem fallback
        Replaces get_available_vaes() from main.py
        """
        return await self.db_manager.get_available_vaes()
    
    async def get_default_vae(self) -> Optional[Dict[str, Any]]:
        """
        Get the default VAE
        """
        all_vaes = await self.get_available_vaes()
        for vae in all_vaes:
            if vae.get("is_default", False):
                return vae
        return None 