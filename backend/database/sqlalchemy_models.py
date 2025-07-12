"""
SQLAlchemy Models for Foto Render
Converted from Prisma schema with full typing support
"""

from datetime import datetime
from typing import List, Optional, Any, Dict
from enum import Enum
import uuid

from sqlalchemy import (
    String, Boolean, Integer, BigInteger, Float, DateTime, JSON, Text,
    ForeignKey, Index, UniqueConstraint, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID, TEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models"""
    pass


# Enums
class ModelType(str, Enum):
    SDXL = "SDXL"
    SD15 = "SD15"
    FLUX = "FLUX"
    OTHER = "OTHER"


class BaseModel(str, Enum):
    NOOBAI = "NOOBAI"
    PONYXL = "PONYXL"
    ILLUSTRIOUS = "ILLUSTRIOUS"
    SDXL_BASE = "SDXL_BASE"
    SD15_BASE = "SD15_BASE"
    OTHER = "OTHER"


class LoraCategory(str, Enum):
    GENERAL = "GENERAL"
    DETAIL_ENHANCEMENT = "DETAIL_ENHANCEMENT"
    STYLE = "STYLE"
    CHARACTER = "CHARACTER"
    CONCEPT = "CONCEPT"
    CLOTHING = "CLOTHING"
    POSE = "POSE"
    LIGHTING = "LIGHTING"
    EXPERIMENTAL = "EXPERIMENTAL"
    NSFW = "NSFW"


class VaeType(str, Enum):
    SDXL = "SDXL"
    SD15 = "SD15"
    GENERAL = "GENERAL"


class UpscalerType(str, Enum):
    GENERAL = "GENERAL"
    REAL_ESRGAN = "REAL_ESRGAN"
    ESRGAN = "ESRGAN"
    ANIME = "ANIME"
    PHOTO = "PHOTO"
    FACE = "FACE"


# Core Models
class AiModel(Base):
    __tablename__ = "AiModel"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column("displayName", String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_type: Mapped[ModelType] = mapped_column("modelType", String, default=ModelType.SDXL)
    base_model: Mapped[BaseModel] = mapped_column("baseModel", String, default=BaseModel.SDXL_BASE)
    capability: Mapped[str] = mapped_column(String, default="text-to-image")

    # File information
    file_path: Mapped[str] = mapped_column("filePath", String, nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column("fileSize", BigInteger, nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Configuration
    is_nsfw: Mapped[bool] = mapped_column("isNsfw", Boolean, default=False)
    is_active: Mapped[bool] = mapped_column("isActive", Boolean, default=True)
    default_sampler: Mapped[str] = mapped_column("defaultSampler", String, default="DPM++ 2M SDE Karras")

    # Metadata
    author: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, default="local")
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)

    # Configuration JSON
    supported_dimensions: Mapped[Optional[Dict[str, Any]]] = mapped_column("supportedDimensions", JSON, nullable=True)
    recommended_settings: Mapped[Optional[Dict[str, Any]]] = mapped_column("recommendedSettings", JSON, nullable=True)

    # Analytics
    usage_count: Mapped[int] = mapped_column("usageCount", Integer, default=0)
    last_used: Mapped[Optional[datetime]] = mapped_column("lastUsed", DateTime, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column("updatedAt", DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    compatible_loras: Mapped[List["LoraCompatibility"]] = relationship("LoraCompatibility", back_populates="model", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_aimodel_model_type", "modelType"),
        Index("idx_aimodel_base_model", "baseModel"),
        Index("idx_aimodel_is_active", "isActive"),
        Index("idx_aimodel_is_nsfw", "isNsfw"),
        Index("idx_aimodel_usage_count", "usageCount"),
    )


class Lora(Base):
    __tablename__ = "Lora"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column("displayName", String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[LoraCategory] = mapped_column(String, default=LoraCategory.GENERAL)

    # File information
    file_path: Mapped[str] = mapped_column("filePath", String, nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column("fileSize", Integer, nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Configuration
    is_nsfw: Mapped[bool] = mapped_column("isNsfw", Boolean, default=False)
    is_active: Mapped[bool] = mapped_column("isActive", Boolean, default=True)

    # Scale configuration
    default_scale: Mapped[float] = mapped_column("defaultScale", Float, default=1.0)
    min_scale: Mapped[float] = mapped_column("minScale", Float, default=-5.0)
    max_scale: Mapped[float] = mapped_column("maxScale", Float, default=5.0)
    recommended_min: Mapped[float] = mapped_column("recommendedMin", Float, default=0.5)
    recommended_max: Mapped[float] = mapped_column("recommendedMax", Float, default=1.5)

    # Usage information
    usage_tips: Mapped[Optional[str]] = mapped_column("usageTips", Text, nullable=True)
    trigger_words: Mapped[List[str]] = mapped_column("triggerWords", ARRAY(String), default=list)

    # Metadata
    author: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, default="local")
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)

    # Analytics
    usage_count: Mapped[int] = mapped_column("usageCount", Integer, default=0)
    last_used: Mapped[Optional[datetime]] = mapped_column("lastUsed", DateTime, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column("updatedAt", DateTime, default=func.now(), onupdate=func.now())

    # Relationships
    compatible_models: Mapped[List["LoraCompatibility"]] = relationship("LoraCompatibility", back_populates="lora", cascade="all, delete-orphan")

    __table_args__ = (
        Index("idx_lora_category", "category"),
        Index("idx_lora_is_active", "isActive"),
        Index("idx_lora_is_nsfw", "isNsfw"),
        Index("idx_lora_usage_count", "usageCount"),
    )


class LoraCompatibility(Base):
    __tablename__ = "LoraCompatibility"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    lora_id: Mapped[str] = mapped_column("loraId", ForeignKey("Lora.id", ondelete="CASCADE"), nullable=False)
    model_id: Mapped[str] = mapped_column("modelId", ForeignKey("AiModel.id", ondelete="CASCADE"), nullable=False)

    # Compatibility metadata
    is_verified: Mapped[bool] = mapped_column("isVerified", Boolean, default=False)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamp
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=func.now())

    # Relationships
    lora: Mapped["Lora"] = relationship("Lora", back_populates="compatible_models")
    model: Mapped["AiModel"] = relationship("AiModel", back_populates="compatible_loras")

    __table_args__ = (
        UniqueConstraint("loraId", "modelId", name="uq_lora_model"),
        Index("idx_lora_compatibility_lora_id", "loraId"),
        Index("idx_lora_compatibility_model_id", "modelId"),
    )


class Vae(Base):
    __tablename__ = "Vae"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column("displayName", String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    vae_type: Mapped[VaeType] = mapped_column("vaeType", String, default=VaeType.SDXL)

    # File information
    file_path: Mapped[str] = mapped_column("filePath", String, nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column("fileSize", Integer, nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Configuration
    is_active: Mapped[bool] = mapped_column("isActive", Boolean, default=True)
    is_default: Mapped[bool] = mapped_column("isDefault", Boolean, default=False)

    # Metadata
    author: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, default="local")
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)

    # Analytics
    usage_count: Mapped[int] = mapped_column("usageCount", Integer, default=0)
    last_used: Mapped[Optional[datetime]] = mapped_column("lastUsed", DateTime, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column("updatedAt", DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_vae_vae_type", "vaeType"),
        Index("idx_vae_is_active", "isActive"),
        Index("idx_vae_is_default", "isDefault"),
        Index("idx_vae_usage_count", "usageCount"),
    )


class Upscaler(Base):
    __tablename__ = "Upscaler"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    model_name: Mapped[str] = mapped_column("modelName", String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    upscaler_type: Mapped[UpscalerType] = mapped_column("upscalerType", String, default=UpscalerType.GENERAL)

    # File information
    file_path: Mapped[str] = mapped_column("filePath", String, nullable=False)
    file_format: Mapped[str] = mapped_column("fileFormat", String, nullable=False)
    file_size: Mapped[Optional[int]] = mapped_column("fileSize", Integer, nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Configuration
    is_active: Mapped[bool] = mapped_column("isActive", Boolean, default=True)
    scale_factor: Mapped[int] = mapped_column("scaleFactor", Integer, default=4)

    # Performance settings
    tile_size: Mapped[int] = mapped_column("tileSize", Integer, default=512)
    tile_pad: Mapped[int] = mapped_column("tilePad", Integer, default=10)
    supports_half: Mapped[bool] = mapped_column("supportsHalf", Boolean, default=True)

    # Metadata
    author: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    website: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, default="local")
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)

    # Analytics
    usage_count: Mapped[int] = mapped_column("usageCount", Integer, default=0)
    last_used: Mapped[Optional[datetime]] = mapped_column("lastUsed", DateTime, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column("updatedAt", DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_upscaler_upscaler_type", "upscalerType"),
        Index("idx_upscaler_is_active", "isActive"),
        Index("idx_upscaler_scale_factor", "scaleFactor"),
        Index("idx_upscaler_usage_count", "usageCount"),
    )


class Sampler(Base):
    __tablename__ = "Sampler"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column("displayName", String, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Configuration
    scheduler_class: Mapped[str] = mapped_column("schedulerClass", String, nullable=False)
    config: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)

    # Metadata
    is_active: Mapped[bool] = mapped_column("isActive", Boolean, default=True)
    is_default: Mapped[bool] = mapped_column("isDefault", Boolean, default=False)

    # Performance notes
    speed_rating: Mapped[Optional[int]] = mapped_column("speedRating", Integer, default=3)
    quality_rating: Mapped[Optional[int]] = mapped_column("qualityRating", Integer, default=3)

    # Analytics
    usage_count: Mapped[int] = mapped_column("usageCount", Integer, default=0)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column("updatedAt", DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_sampler_is_active", "isActive"),
        Index("idx_sampler_is_default", "isDefault"),
        Index("idx_sampler_usage_count", "usageCount"),
    )


class SystemConfig(Base):
    __tablename__ = "SystemConfig"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    value: Mapped[str] = mapped_column(String, nullable=False)
    type: Mapped[str] = mapped_column(String, default="string")

    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_secret: Mapped[bool] = mapped_column("isSecret", Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column("updatedAt", DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_systemconfig_category", "category"),
        Index("idx_systemconfig_key", "key"),
    )


class FavoritePrompt(Base):
    __tablename__ = "FavoritePrompt"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title: Mapped[str] = mapped_column(String, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    negative_prompt: Mapped[Optional[str]] = mapped_column("negativePrompt", Text, nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Generation parameters
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    steps: Mapped[int] = mapped_column(Integer, nullable=False)
    guidance_scale: Mapped[float] = mapped_column("guidanceScale", Float, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    sampler: Mapped[str] = mapped_column(String, nullable=False)
    clip_skip: Mapped[Optional[int]] = mapped_column("clipSkip", Integer, nullable=True)
    loras_used: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column("lorasUsed", JSON, nullable=True)

    # R2 image information
    image_r2_key: Mapped[str] = mapped_column("imageR2Key", String, nullable=False)
    image_r2_url: Mapped[str] = mapped_column("imageR2Url", String, nullable=False)
    image_cdn_url: Mapped[Optional[str]] = mapped_column("imageCdnUrl", String, nullable=True)
    image_width: Mapped[int] = mapped_column("imageWidth", Integer, nullable=False)
    image_height: Mapped[int] = mapped_column("imageHeight", Integer, nullable=False)
    image_size_bytes: Mapped[int] = mapped_column("imageSizeBytes", Integer, nullable=False)
    image_format: Mapped[str] = mapped_column("imageFormat", String, default="png")

    # Thumbnail information
    thumbnail_r2_key: Mapped[Optional[str]] = mapped_column("thumbnailR2Key", String, nullable=True)
    thumbnail_cdn_url: Mapped[Optional[str]] = mapped_column("thumbnailCdnUrl", String, nullable=True)

    # Organization
    tags: Mapped[List[str]] = mapped_column(ARRAY(String), default=list)
    category: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    is_public: Mapped[bool] = mapped_column("isPublic", Boolean, default=False)
    is_featured: Mapped[bool] = mapped_column("isFeatured", Boolean, default=False)

    # Analytics
    usage_count: Mapped[int] = mapped_column("usageCount", Integer, default=0)
    like_count: Mapped[int] = mapped_column("likeCount", Integer, default=0)
    view_count: Mapped[int] = mapped_column("viewCount", Integer, default=0)
    last_used: Mapped[Optional[datetime]] = mapped_column("lastUsed", DateTime, nullable=True)

    # Model relationship
    model_id: Mapped[Optional[str]] = mapped_column("modelId", ForeignKey("AiModel.id", ondelete="SET NULL"), nullable=True)
    model: Mapped[Optional["AiModel"]] = relationship("AiModel")

    # Timestamps
    created_at: Mapped[datetime] = mapped_column("createdAt", DateTime, default=func.now())
    updated_at: Mapped[datetime] = mapped_column("updatedAt", DateTime, default=func.now(), onupdate=func.now())

    __table_args__ = (
        Index("idx_favoriteprompt_is_public", "isPublic"),
        Index("idx_favoriteprompt_is_featured", "isFeatured"),
        Index("idx_favoriteprompt_category", "category"),
        Index("idx_favoriteprompt_model_id", "modelId"),
        Index("idx_favoriteprompt_usage_count", "usageCount"),
        Index("idx_favoriteprompt_like_count", "likeCount"),
        Index("idx_favoriteprompt_view_count", "viewCount"),
        Index("idx_favoriteprompt_created_at", "createdAt"),
    ) 