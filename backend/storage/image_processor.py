"""
Image processing utilities for Foto Render
Handles thumbnail generation, resizing, and format conversion
"""

import logging
from typing import Tuple, Union, Optional, Protocol, Any
from io import BytesIO

# PIL Protocol definitions for proper typing
class PILImageProtocol(Protocol):
    width: int
    height: int
    mode: str
    format: Optional[str]
    info: dict[str, Any]
    size: Tuple[int, int]
    
    def open(self, fp: Union[str, BytesIO]) -> 'PILImageProtocol': ...
    def new(self, mode: str, size: Tuple[int, int], color: Any = ...) -> 'PILImageProtocol': ...
    def convert(self, mode: str) -> 'PILImageProtocol': ...
    def thumbnail(self, size: Tuple[int, int], resample: Any = ...) -> None: ...
    def resize(self, size: Tuple[int, int], resample: Any = ...) -> 'PILImageProtocol': ...
    def paste(self, im: 'PILImageProtocol', box: Any = ..., mask: Any = ...) -> None: ...
    def split(self) -> Tuple[Any, ...]: ...
    def save(self, fp: BytesIO, **kwargs: Any) -> None: ...

class ResamplingProtocol(Protocol):
    LANCZOS: Any

# Runtime imports with proper fallback
try:
    from PIL import Image, ImageOps
    from PIL.Image import Resampling
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    # Create fallback that will raise proper error when used
    class _PILNotAvailable:
        def __getattr__(self, name: str) -> Any:
            raise ImportError("Pillow library is required for image processing. Install with: pip install Pillow")
    
    Image = _PILNotAvailable()  # type: ignore
    ImageOps = _PILNotAvailable()  # type: ignore  
    Resampling = _PILNotAvailable()  # type: ignore

logger = logging.getLogger(__name__)

class ImageProcessor:
    """
    Image processing utilities for thumbnail generation and resizing
    """
    
    def __init__(self):
        pass
    
    def create_thumbnail(self, image_data: Union[bytes, BytesIO], 
                        size: Tuple[int, int] = (512, 512),
                        quality: int = 85,
                        format: str = "PNG") -> bytes:
        """
        Create a thumbnail from image data
        
        Args:
            image_data: Original image data
            size: Thumbnail size as (width, height)
            quality: JPEG quality (ignored for PNG)
            format: Output format (PNG, JPEG, WEBP)
        
        Returns:
            Thumbnail image data as bytes
        """
        if not PIL_AVAILABLE:
            raise ImportError("Pillow library is required for image processing. Install with: pip install Pillow")
        
        try:
            # Load image
            if isinstance(image_data, BytesIO):
                image_data.seek(0)
                img = Image.open(image_data)
            else:
                img = Image.open(BytesIO(image_data))
            
            # Convert to RGB if necessary (for JPEG output)
            if format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
                # Create white background for transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            
            # Create thumbnail with proper aspect ratio
            img.thumbnail(size, Resampling.LANCZOS)
            
            # Save to BytesIO
            output = BytesIO()
            save_kwargs: dict[str, Any] = {"format": format.upper()}
            
            if format.upper() == "JPEG":
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True
            elif format.upper() == "PNG":
                save_kwargs["optimize"] = True
            elif format.upper() == "WEBP":
                save_kwargs["quality"] = quality
                save_kwargs["method"] = 6  # Higher compression
            
            img.save(output, **save_kwargs)
            
            thumbnail_data = output.getvalue()
            output.close()
            
            logger.info(f"✅ Created thumbnail: {img.size} -> {len(thumbnail_data)} bytes")
            return thumbnail_data
            
        except Exception as e:
            logger.error(f"❌ Failed to create thumbnail: {e}")
            raise
    
    def resize_image(self, image_data: Union[bytes, BytesIO],
                    target_size: Tuple[int, int],
                    maintain_aspect: bool = True,
                    quality: int = 85,
                    format: Optional[str] = None) -> bytes:
        """
        Resize an image to target dimensions
        
        Args:
            image_data: Original image data
            target_size: Target size as (width, height)
            maintain_aspect: Whether to maintain aspect ratio
            quality: Output quality for lossy formats
            format: Output format (None to keep original)
        
        Returns:
            Resized image data as bytes
        """
        try:
            # Load image
            if isinstance(image_data, BytesIO):
                image_data.seek(0)
                img = Image.open(image_data)
            else:
                img = Image.open(BytesIO(image_data))
            
            original_format = img.format or "PNG"
            output_format = format or original_format
            
            # Resize image
            if maintain_aspect:
                img.thumbnail(target_size, Resampling.LANCZOS)
            else:
                img = img.resize(target_size, Resampling.LANCZOS)
            
            # Handle format conversion
            if output_format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
                # Create white background for transparency
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            
            # Save to BytesIO
            output = BytesIO()
            save_kwargs: dict[str, Any] = {"format": output_format.upper()}
            
            if output_format.upper() == "JPEG":
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True
            elif output_format.upper() == "PNG":
                save_kwargs["optimize"] = True
            elif output_format.upper() == "WEBP":
                save_kwargs["quality"] = quality
                save_kwargs["method"] = 6
            
            img.save(output, **save_kwargs)
            
            resized_data = output.getvalue()
            output.close()
            
            logger.info(f"✅ Resized image: {img.size} -> {len(resized_data)} bytes")
            return resized_data
            
        except Exception as e:
            logger.error(f"❌ Failed to resize image: {e}")
            raise
    
    def convert_format(self, image_data: Union[bytes, BytesIO],
                      target_format: str,
                      quality: int = 85) -> bytes:
        """
        Convert image to different format
        
        Args:
            image_data: Original image data
            target_format: Target format (PNG, JPEG, WEBP)
            quality: Quality for lossy formats
        
        Returns:
            Converted image data as bytes
        """
        try:
            # Load image
            if isinstance(image_data, BytesIO):
                image_data.seek(0)
                img = Image.open(image_data)
            else:
                img = Image.open(BytesIO(image_data))
            
            # Handle transparency for JPEG
            if target_format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            
            # Save in target format
            output = BytesIO()
            save_kwargs: dict[str, Any] = {"format": target_format.upper()}
            
            if target_format.upper() == "JPEG":
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True
            elif target_format.upper() == "PNG":
                save_kwargs["optimize"] = True
            elif target_format.upper() == "WEBP":
                save_kwargs["quality"] = quality
                save_kwargs["method"] = 6
            
            img.save(output, **save_kwargs)
            
            converted_data = output.getvalue()
            output.close()
            
            logger.info(f"✅ Converted image to {target_format}: {len(converted_data)} bytes")
            return converted_data
            
        except Exception as e:
            logger.error(f"❌ Failed to convert image format: {e}")
            raise
    
    def get_image_info(self, image_data: Union[bytes, BytesIO]) -> dict[str, Any]:
        """
        Get information about an image
        
        Args:
            image_data: Image data to analyze
        
        Returns:
            Dict with image information
        """
        try:
            if isinstance(image_data, BytesIO):
                image_data.seek(0)
                img = Image.open(image_data)
            else:
                img = Image.open(BytesIO(image_data))
            
            return {
                "width": img.width,
                "height": img.height,
                "format": img.format,
                "mode": img.mode,
                "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
                "size_bytes": len(image_data) if isinstance(image_data, bytes) else len(image_data.getvalue())
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get image info: {e}")
            return {}
    
    def optimize_for_web(self, image_data: Union[bytes, BytesIO],
                        max_width: int = 1920,
                        max_height: int = 1920,
                        quality: int = 85,
                        format: str = "WEBP") -> bytes:
        """
        Optimize image for web delivery
        
        Args:
            image_data: Original image data
            max_width: Maximum width
            max_height: Maximum height
            quality: Output quality
            format: Output format (WEBP recommended)
        
        Returns:
            Optimized image data as bytes
        """
        try:
            # Load image
            if isinstance(image_data, BytesIO):
                image_data.seek(0)
                img = Image.open(image_data)
            else:
                img = Image.open(BytesIO(image_data))
            
            # Resize if needed
            if img.width > max_width or img.height > max_height:
                img.thumbnail((max_width, max_height), Resampling.LANCZOS)
            
            # Convert to RGB for JPEG, keep RGBA for PNG/WEBP
            if format.upper() == "JPEG" and img.mode in ("RGBA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = background
            
            # Save optimized
            output = BytesIO()
            save_kwargs: dict[str, Any] = {"format": format.upper()}
            
            if format.upper() == "JPEG":
                save_kwargs["quality"] = quality
                save_kwargs["optimize"] = True
                save_kwargs["progressive"] = True
            elif format.upper() == "PNG":
                save_kwargs["optimize"] = True
            elif format.upper() == "WEBP":
                save_kwargs["quality"] = quality
                save_kwargs["method"] = 6
                save_kwargs["lossless"] = False
            
            img.save(output, **save_kwargs)
            
            optimized_data = output.getvalue()
            output.close()
            
            original_size = len(image_data) if isinstance(image_data, bytes) else len(image_data.getvalue())
            compression_ratio = (original_size - len(optimized_data)) / original_size * 100
            
            logger.info(f"✅ Optimized image: {original_size} -> {len(optimized_data)} bytes ({compression_ratio:.1f}% reduction)")
            return optimized_data
            
        except Exception as e:
            logger.error(f"❌ Failed to optimize image: {e}")
            raise 