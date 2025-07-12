"""
Database package for Foto Render
Provides SQLAlchemy models and services for managing AI assets
"""

from .sqlalchemy_client import (
    get_db_client, 
    DatabaseClient,
    ModelService,
    LoraService,
    VaeService,
    UpscalerService,
    SamplerService,
    ConfigService
)
from .sqlalchemy_models import (
    Base,
    AiModel,
    Lora,
    LoraCompatibility,
    Vae,
    Upscaler,
    Sampler,
    SystemConfig,
    FavoritePrompt,
    ModelType,
    BaseModel,
    LoraCategory,
    VaeType,
    UpscalerType
)

__all__ = [
    # Client
    'get_db_client',
    'DatabaseClient',
    # Services
    'ModelService',
    'LoraService', 
    'VaeService',
    'UpscalerService',
    'SamplerService',
    'ConfigService',
    # Models
    'Base',
    'AiModel',
    'Lora',
    'LoraCompatibility',
    'Vae',
    'Upscaler',
    'Sampler',
    'SystemConfig',
    'FavoritePrompt',
    # Enums
    'ModelType',
    'BaseModel',
    'LoraCategory',
    'VaeType',
    'UpscalerType',
] 