"""
Cloudflare R2 Storage Client for Foto Render
Handles image uploads, downloads, and management in R2 bucket
"""

import os
import uuid
import logging
from typing import Optional, Dict, Any, Tuple, Union, List, TYPE_CHECKING, Protocol
from io import BytesIO
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError, NoCredentialsError
from PIL import Image as PILImage

# Proper S3 client protocol - defines the methods we actually use
class S3ClientProtocol(Protocol):
    def put_object(self, **kwargs: Any) -> Dict[str, Any]: ...
    def delete_object(self, **kwargs: Any) -> Dict[str, Any]: ...
    def head_object(self, **kwargs: Any) -> Dict[str, Any]: ...
    def list_objects_v2(self, **kwargs: Any) -> Dict[str, Any]: ...
    def generate_presigned_url(self, operation: str, Params: Dict[str, Any], ExpiresIn: int) -> str: ...
    def head_bucket(self, **kwargs: Any) -> Dict[str, Any]: ...

# Type alias for clean code
S3Client = S3ClientProtocol

from database_config import get_r2_config, get_r2_credentials

logger = logging.getLogger(__name__)

class R2StorageClient:
    """
    Cloudflare R2 storage client for managing images and thumbnails
    """
    
    def __init__(self):
        self.config = get_r2_config()
        self.credentials = get_r2_credentials()
        self._client: Optional[S3Client] = None
        self._setup_client()
    
    def _setup_client(self):
        """Initialize the S3-compatible client for R2"""
            
        try:
            credentials = self.credentials
            
            if not credentials["access_key_id"] or not credentials["secret_access_key"]:
                logger.warning("⚠️ R2 credentials not found. Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY in environment")
                return
            
            # Configure S3 client for Cloudflare R2
            client = boto3.client(
                "s3",
                endpoint_url=self.config["ENDPOINT_URL"],
                aws_access_key_id=credentials["access_key_id"],
                aws_secret_access_key=credentials["secret_access_key"],
                region_name=credentials["region"],
                config=Config(
                    signature_version="v4",
                    s3={
                        "addressing_style": "path"  # Required for R2
                    }
                )
            )
            
            # Test connection
            client.head_bucket(Bucket=self.config["BUCKET_NAME"])
            
            # Store client after successful connection
            self._client = client  # type: ignore  # boto3 client implements our protocol
            logger.info(f"✅ R2 client connected to bucket: {self.config['BUCKET_NAME']}")
            
        except NoCredentialsError:
            logger.error("❌ R2 credentials not provided")
            self._client = None
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.error(f"❌ R2 bucket '{self.config['BUCKET_NAME']}' not found")
            else:
                logger.error(f"❌ R2 connection failed: {e}")
            self._client = None
        except Exception as e:
            logger.error(f"❌ R2 setup failed: {e}")
            self._client = None
    
    def is_available(self) -> bool:
        """Check if R2 client is available and configured"""
        return self._client is not None
    
    def _get_client(self) -> S3Client:
        """Get the client with proper type checking"""
        if self._client is None:
            raise RuntimeError("R2 client not available")
        return self._client
    
    def upload_image(self, image_data: Union[bytes, BytesIO], 
                    filename: Optional[str] = None,
                    folder: str = "prompts",
                    content_type: str = "image/png",
                    generate_thumbnail: bool = True) -> Dict[str, Any]:
        """
        Upload an image to R2 storage
        
        Args:
            image_data: Image data as bytes or BytesIO
            filename: Custom filename (generates UUID if None)
            folder: R2 folder path (prompts, thumbnails, etc.)
            content_type: MIME type of the image
            generate_thumbnail: Whether to generate and upload thumbnail
        
        Returns:
            Dict with upload information including URLs and keys
        """
        if not self.is_available():
            raise RuntimeError("R2 storage client not available")
        
        client = self._get_client()
        
        try:
            # Generate filename if not provided
            if filename is None:
                file_extension = self._get_extension_from_content_type(content_type)
                filename = f"{uuid.uuid4()}{file_extension}"
            
            # Ensure filename has proper extension
            if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                filename = f"{filename}.png"
            
            # Create R2 key
            r2_key = f"{folder}/{filename}"
            
            # Convert image_data to bytes if needed
            if isinstance(image_data, BytesIO):
                image_bytes = image_data.getvalue()
            else:
                image_bytes = image_data
            
            # Get image metadata
            metadata = self._get_image_metadata(image_bytes)
            
            # Upload main image
            client.put_object(
                Bucket=self.config["BUCKET_NAME"],
                Key=r2_key,
                Body=image_bytes,
                ContentType=content_type,
                Metadata={
                    "width": str(metadata["width"]),
                    "height": str(metadata["height"]),
                    "size_bytes": str(len(image_bytes)),
                    "upload_source": "foto-render"
                }
            )
            
            # Generate URLs
            r2_url = f"{self.config['PUBLIC_URL_BASE']}/{r2_key}"
            cdn_url = f"{self.config['CDN_URL_BASE']}/{r2_key}" if self.config.get("CDN_URL_BASE") else None
            
            result = {
                "r2_key": r2_key,
                "r2_url": r2_url,
                "cdn_url": cdn_url,
                "filename": filename,
                "folder": folder,
                "content_type": content_type,
                "size_bytes": len(image_bytes),
                "width": metadata["width"],
                "height": metadata["height"],
                "thumbnail": None
            }
            
            # Generate and upload thumbnail if requested
            if generate_thumbnail:
                try:
                    thumbnail_info = self._upload_thumbnail(image_bytes, filename, content_type)
                    result["thumbnail"] = thumbnail_info
                except Exception as e:
                    logger.warning(f"⚠️ Failed to generate thumbnail for {filename}: {e}")
            
            logger.info(f"✅ Uploaded image to R2: {r2_key}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to upload image to R2: {e}")
            raise
    
    def _upload_thumbnail(self, image_bytes: bytes, original_filename: str, content_type: str) -> Dict[str, Any]:
        """Generate and upload thumbnail version"""
        from .image_processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Generate thumbnail
        thumbnail_data = processor.create_thumbnail(
            image_bytes, 
            size=self.config["THUMBNAIL_SIZE"]
        )
        
        # Create thumbnail filename
        name, ext = os.path.splitext(original_filename)
        thumbnail_filename = f"{name}_thumb{ext}"
        
        # Upload thumbnail
        thumbnail_key = f"{self.config['THUMBNAILS_PATH']}/{thumbnail_filename}"
        
        client = self._get_client()
        
        client.put_object(
            Bucket=self.config["BUCKET_NAME"],
            Key=thumbnail_key,
            Body=thumbnail_data,
            ContentType=content_type,
            Metadata={
                "original_file": original_filename,
                "thumbnail": "true",
                "upload_source": "foto-render"
            }
        )
        
        # Generate thumbnail URLs
        thumbnail_r2_url = f"{self.config['PUBLIC_URL_BASE']}/{thumbnail_key}"
        thumbnail_cdn_url = f"{self.config['CDN_URL_BASE']}/{thumbnail_key}" if self.config.get("CDN_URL_BASE") else None
        
        return {
            "r2_key": thumbnail_key,
            "r2_url": thumbnail_r2_url,
            "cdn_url": thumbnail_cdn_url,
            "filename": thumbnail_filename,
            "size_bytes": len(thumbnail_data)
        }
    
    def delete_image(self, r2_key: str, delete_thumbnail: bool = True) -> bool:
        """
        Delete an image from R2 storage
        
        Args:
            r2_key: R2 object key to delete
            delete_thumbnail: Whether to also delete associated thumbnail
        
        Returns:
            True if deletion successful
        """
        if not self.is_available():
            logger.warning("R2 storage client not available for deletion")
            return False
        
        client = self._get_client()
        
        try:
            # Delete main image
            client.delete_object(
                Bucket=self.config["BUCKET_NAME"],
                Key=r2_key
            )
            
            logger.info(f"✅ Deleted image from R2: {r2_key}")
            
            # Delete thumbnail if requested
            if delete_thumbnail:
                try:
                    # Derive thumbnail key from main image key
                    folder, filename = r2_key.rsplit("/", 1)
                    name, ext = os.path.splitext(filename)
                    thumbnail_key = f"{self.config['THUMBNAILS_PATH']}/{name}_thumb{ext}"
                    
                    client.delete_object(
                        Bucket=self.config["BUCKET_NAME"],
                        Key=thumbnail_key
                    )
                    
                    logger.info(f"✅ Deleted thumbnail from R2: {thumbnail_key}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Could not delete thumbnail {thumbnail_key}: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to delete image from R2: {e}")
            return False
    
    def get_image_info(self, r2_key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata about an image in R2 storage
        
        Args:
            r2_key: R2 object key
        
        Returns:
            Image metadata dict or None if not found
        """
        if not self.is_available():
            return None
        
        client = self._get_client()
        
        try:
            response = client.head_object(
                Bucket=self.config["BUCKET_NAME"],
                Key=r2_key
            )
            
            metadata = response.get("Metadata", {})
            
            return {
                "r2_key": r2_key,
                "size_bytes": response["ContentLength"],
                "content_type": response["ContentType"],
                "last_modified": response["LastModified"],
                "width": int(metadata.get("width", 0)),
                "height": int(metadata.get("height", 0)),
                "etag": response["ETag"],
                "metadata": metadata
            }
            
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.warning(f"Image not found in R2: {r2_key}")
            else:
                logger.error(f"Error getting image info from R2: {e}")
            return None
    
    def generate_presigned_url(self, r2_key: str, expiration: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for direct access to an R2 object
        
        Args:
            r2_key: R2 object key
            expiration: URL expiration time in seconds
        
        Returns:
            Presigned URL or None if failed
        """
        if not self.is_available():
            return None
        
        client = self._get_client()
        
        try:
            url = client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": self.config["BUCKET_NAME"],
                    "Key": r2_key
                },
                ExpiresIn=expiration
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    def list_images(self, folder: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        """
        List images in a specific folder
        
        Args:
            folder: Folder prefix to search
            limit: Maximum number of results
        
        Returns:
            List of image information dicts
        """
        if not self.is_available():
            return []
        
        client = self._get_client()
        
        try:
            response = client.list_objects_v2(
                Bucket=self.config["BUCKET_NAME"],
                Prefix=folder,
                MaxKeys=limit
            )
            
            images = []
            for obj in response.get("Contents", []):
                # Skip directories
                if obj["Key"].endswith("/"):
                    continue
                
                # Only include image files
                if not self._is_image_file(obj["Key"]):
                    continue
                
                images.append({
                    "r2_key": obj["Key"],
                    "size_bytes": obj["Size"],
                    "last_modified": obj["LastModified"],
                    "etag": obj["ETag"],
                    "r2_url": f"{self.config['PUBLIC_URL_BASE']}/{obj['Key']}",
                    "cdn_url": f"{self.config['CDN_URL_BASE']}/{obj['Key']}" if self.config.get("CDN_URL_BASE") else None
                })
            
            return images
            
        except Exception as e:
            logger.error(f"Failed to list images from R2: {e}")
            return []
    
    def _get_image_metadata(self, image_bytes: bytes) -> Dict[str, int]:
        """Extract metadata from image bytes"""
        
        try:
            with PILImage.open(BytesIO(image_bytes)) as img:
                return {
                    "width": img.width,
                    "height": img.height
                }
        except Exception as e:
            logger.warning(f"Could not extract image metadata: {e}")
            return {"width": 0, "height": 0}
    
    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Get file extension from MIME type"""
        type_map = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp"
        }
        return type_map.get(content_type, ".png")
    
    def _is_image_file(self, filename: str) -> bool:
        """Check if filename is an image file"""
        image_extensions = ['.png', '.jpg', '.jpeg', '.webp', '.gif']
        return any(filename.lower().endswith(ext) for ext in image_extensions) 