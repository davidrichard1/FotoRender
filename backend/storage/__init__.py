"""
Storage package for Foto Render
Handles Cloudflare R2 storage operations for images and thumbnails
"""

from .r2_client import R2StorageClient
from .image_processor import ImageProcessor

__all__ = [
    'R2StorageClient',
    'ImageProcessor',
] 