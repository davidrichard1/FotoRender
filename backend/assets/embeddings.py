"""
Embedding management module for Foto Render
Replaces embedding-related functions from main.py with database-driven approach
"""

import logging
from typing import List, Dict, Optional, Any
from database.services import EmbeddingManager as DatabaseEmbeddingManager

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """
    Embedding manager that provides database-driven embedding management
    Replaces get_available_embeddings() and related functions from main.py
    """
    
    def __init__(self, embeddings_dir: str = "embeddings"):
        self.embeddings_dir = embeddings_dir
        self.db_manager = DatabaseEmbeddingManager(embeddings_dir)
    
    async def get_available_embeddings(self) -> List[Dict[str, Any]]:
        """
        Get available embeddings from database with filesystem fallback
        Replaces get_available_embeddings() from main.py
        """
        return await self.db_manager.get_available_embeddings()
    
    async def sync_embeddings_from_filesystem(self) -> Dict[str, Any]:
        """
        Sync embeddings from filesystem to database
        """
        # Implementation would go here
        return {"status": "not_implemented", "message": "Embedding sync not yet implemented"}
    
    async def get_embeddings_by_type(self, embedding_type: str) -> List[Dict[str, Any]]:
        """
        Get embeddings filtered by type
        """
        all_embeddings = await self.get_available_embeddings()
        return [emb for emb in all_embeddings if emb.get("type") == embedding_type]
    
    async def get_negative_embeddings(self) -> List[Dict[str, Any]]:
        """
        Get negative embeddings specifically
        """
        return await self.get_embeddings_by_type("negative") 