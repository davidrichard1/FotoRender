"""
High-level service managers for Foto Render
Combines database operations with filesystem scanning and legacy compatibility
"""

import os
import glob
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from .sqlalchemy_client import (
    get_db_client, ModelService as AiModelService, LoraService, 
    VaeService, UpscalerService, SamplerService, ConfigService as SystemConfigService, DatabaseClient
)
from .sqlalchemy_models import AiModel, Lora, LoraCompatibility, Vae, Upscaler, Sampler, SystemConfig

logger = logging.getLogger(__name__)

class ModelManager:
    """High-level manager for AI models - now uses singleton DB client"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        # No longer store db_client instance - use singleton
    
    async def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get available models with database metadata and filesystem fallback
        Maintains compatibility with the original main.py format
        """
        try:
            # Get singleton database client
            db_client = await get_db_client()
            service = AiModelService(db_client)
            db_models = await service.get_all_models(active_only=False)  # Get ALL models, not just active ones
            
            if db_models:
                logger.info(f"🗄️ Found {len(db_models)} models in database")
                for i, model in enumerate(db_models):
                    logger.info(f"  Model {i+1}: {model.filename} (active: {model.is_active})")
                
                # Convert database models to legacy format
                models = []
                for db_model in db_models:
                    # Always use correct path: models/filename
                    model_path = os.path.join(self.models_dir, db_model.filename)
                    
                    # Safely handle enum values that might already be strings
                    model_type = db_model.model_type.value if hasattr(db_model.model_type, 'value') else str(db_model.model_type)
                    base_model = db_model.base_model.value if hasattr(db_model.base_model, 'value') else str(db_model.base_model)
                    
                    model_dict = {
                        "filename": db_model.filename,
                        "display_name": db_model.display_name,
                        "path": model_path,  # Always use models/filename
                        "type": model_type.lower(),
                        "capability": db_model.capability,
                        "source": db_model.source,
                        "base_model": base_model.lower().replace("_", "-"),
                        "description": db_model.description,
                        "is_nsfw": db_model.is_nsfw,
                        "file_size_mb": db_model.file_size,
                        "usage_count": db_model.usage_count,
                        "last_used": db_model.last_used.isoformat() if db_model.last_used else None,
                    }
                    models.append(model_dict)  # Include ALL models from database
                    logger.info(f"✅ Added model: {db_model.filename}")
                
                logger.info(f"🗄️ Final models list contains {len(models)} models")
                return models
            else:
                # Fallback to filesystem scanning if no database models
                logger.warning("No models in database, falling back to filesystem scan")
                return await self._scan_filesystem_models()
                
        except Exception as e:
            logger.error(f"Error getting models from database: {e}")
            logger.info("Falling back to filesystem scan")
            return await self._scan_filesystem_models()
    
    async def _scan_filesystem_models(self) -> List[Dict[str, Any]]:
        """Fallback filesystem scanning (legacy compatibility)"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        model_files = glob.glob(os.path.join(self.models_dir, "*.safetensors"))
        
        for file_path in model_files:
            filename = os.path.basename(file_path)
            
            # Skip Flux models
            if any(flux_name in filename.lower() for flux_name in ["flux", "flux-dev", "flux-kontext"]):
                continue
            
            # Legacy classification logic
            base_model = self._classify_base_model_legacy(filename)
            display_name = filename.replace(".safetensors", "").replace("_", " ").replace("-", " ")
            
            models.append({
                "filename": filename,
                "display_name": display_name,
                "path": file_path,
                "type": "sdxl",
                "capability": "text-to-image",
                "source": "local",
                "base_model": base_model,
                "description": None,
                "is_nsfw": self._detect_nsfw_legacy(filename),
                "file_size_mb": None,
                "usage_count": 0,
                "last_used": None,
            })
        
        return models
    
    def _classify_base_model_legacy(self, filename: str) -> str:
        """Legacy base model classification"""
        filename_lower = filename.lower()
        
        if any(noobai in filename_lower for noobai in [
            "yiffymix_v61arenoobxl", "yiffymix_v62noobxl",
            "indigofurrymixXL_noobaieps6", "indigofurrymixXL_noobaiepsreal5"
        ]):
            return "noobai"
        elif any(pony in filename_lower for pony in ["waianiNSFWPONYXL", "ponyxl", "pony"]):
            return "ponyxl"
        elif any(illustrious in filename_lower for illustrious in [
            "wai-nsfw-illustrious-sdxl", "illustrious", "yiffinhell"
        ]):
            return "illustrious"
        else:
            return "sdxl_base"
    
    def _detect_nsfw_legacy(self, filename: str) -> bool:
        """Legacy NSFW detection"""
        filename_lower = filename.lower()
        return any(nsfw in filename_lower for nsfw in ["nsfw", "adult", "xxx"])
    
    async def sync_models(self) -> Dict[str, Any]:
        """Sync models from filesystem to database"""
        # This would need to be implemented in the ModelService
        logger.warning("sync_models not yet implemented")
        return {"synced": 0, "errors": 0}
    
    async def update_usage(self, filename: str):
        """Update model usage statistics"""
        try:
            # Get singleton database client
            db_client = await get_db_client()
            
            async with db_client.get_session() as session:
                result = await session.execute(
                    select(AiModel).where(AiModel.filename == filename)
                )
                model = result.scalar_one_or_none()
                if model:
                    model.usage_count += 1
                    model.last_used = func.now()
                    await session.commit()
        except Exception as e:
            logger.warning(f"Could not update usage for model {filename}: {e}")

class LoraManager:
    """High-level manager for LoRA models"""
    
    def __init__(self, loras_dir: str = "loras"):
        self.loras_dir = loras_dir
        # No longer store db_client instance - use singleton
    
    async def get_available_loras(self, filter_by_current_model: bool = False, 
                                 current_model_filename: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get available LoRAs with database metadata and filesystem fallback
        Maintains compatibility with the original main.py format
        """
        try:
            # Get singleton database client
            db_client = await get_db_client()
            service = LoraService(db_client)
            
            compatible_with = current_model_filename if filter_by_current_model else None
            logger.info(f"🎨 Fetching LoRAs from database - filter_by_current_model: {filter_by_current_model}, compatible_with: {compatible_with}")
            
            db_loras = await service.get_all_loras(active_only=False, compatible_with_model=compatible_with)  # Get ALL LoRAs, not just active ones
            
            # Always use database results when available, even if empty list
            logger.info(f"🎨 Found {len(db_loras)} LoRAs in database")
            
            if filter_by_current_model and len(db_loras) == 0:
                logger.info(f"🚫 No LoRAs compatible with model {current_model_filename} - returning empty list (no fallback)")
                return []
            
            if len(db_loras) > 0:
                for i, lora in enumerate(db_loras):
                    logger.info(f"  LoRA {i+1}: {lora.filename} (active: {lora.is_active})")
            
            # Convert to legacy format
            loras = []
            for db_lora in db_loras:
                # Always use correct path: loras/filename
                lora_path = os.path.join(self.loras_dir, db_lora.filename)
                
                # Safely handle enum values that might already be strings
                category = db_lora.category.value if hasattr(db_lora.category, 'value') else str(db_lora.category)
                
                lora_dict = {
                    "filename": db_lora.filename,
                    "display_name": db_lora.display_name,
                    "path": lora_path,  # Always use loras/filename
                    "size_mb": db_lora.file_size or 0,
                    "type": "lora",
                    "category": category.lower().replace("_", "-"),
                    "supported_models": ["sdxl"],
                    "compatible_bases": self._get_compatible_bases_from_db(db_lora),
                    "source": db_lora.source,
                    "is_nsfw": db_lora.is_nsfw,
                    # Scale configuration
                    "scale_config": {
                        "default_scale": db_lora.default_scale,
                        "recommended_scale": db_lora.default_scale,
                        "min_scale": db_lora.min_scale,
                        "max_scale": db_lora.max_scale,
                        "optimal_range": [db_lora.recommended_min, db_lora.recommended_max]
                    },
                    # Usage information
                    "usage_tips": {
                        "description": db_lora.description or "LoRA model",
                        "range_info": f"Supported range: {db_lora.min_scale} to {db_lora.max_scale}",
                        "category_info": f"{category.replace('_', ' ').title()} LoRA",
                        "trigger_words": db_lora.trigger_words or []
                    },
                    "usage_count": db_lora.usage_count,
                    "last_used": db_lora.last_used.isoformat() if db_lora.last_used else None,
                }
                loras.append(lora_dict)  # Include ALL LoRAs from database
            
            # Only fallback to filesystem when not filtering AND no database LoRAs found
            if not filter_by_current_model and len(db_loras) == 0:
                logger.warning("No LoRAs in database and not filtering - falling back to filesystem scan")
                return await self._scan_filesystem_loras()
            
            return loras
                
        except Exception as e:
            logger.error(f"❌ ERROR getting LoRAs from database: {e}")
            logger.error(f"❌ Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Full traceback: {traceback.format_exc()}")
            # Don't fallback to filesystem if filtering is enabled
            if filter_by_current_model:
                logger.error(f"Filtering enabled for model {current_model_filename}, returning empty list instead of filesystem fallback")
                return []
            logger.warning("Database failed and not filtering - falling back to filesystem scan")
            return await self._scan_filesystem_loras()
    
    def _get_compatible_bases_from_db(self, db_lora) -> List[str]:
        """Extract compatible base models from database relationships"""
        compatible_bases = []
        try:
            for compatibility in db_lora.compatible_models:
                # Safely handle enum values that might already be strings
                base_model_enum = compatibility.model.base_model
                base_model = base_model_enum.value if hasattr(base_model_enum, 'value') else str(base_model_enum)
                base_model = base_model.lower().replace("_", "-")
                if base_model not in compatible_bases:
                    compatible_bases.append(base_model)
        except Exception as e:
            logger.warning(f"Could not access compatibility data for LoRA {db_lora.filename}: {e}")
            # Fall back to filename-based classification
            compatible_bases = self._get_lora_base_classification_legacy(db_lora.filename)
        
        # If no specific compatibility, assume general compatibility
        if not compatible_bases:
            compatible_bases = ["noobai", "ponyxl", "illustrious", "sdxl-base"]
        
        return compatible_bases
    
    async def _scan_filesystem_loras(self) -> List[Dict[str, Any]]:
        """Fallback filesystem scanning"""
        loras = []
        
        if not os.path.exists(self.loras_dir):
            return loras
        
        lora_files = glob.glob(os.path.join(self.loras_dir, "*.safetensors"))
        
        for file_path in lora_files:
            filename = os.path.basename(file_path)
            display_name = filename.replace(".safetensors", "").replace("_", " ").replace("-", " ")
            
            # Legacy classification
            compatible_bases = self._get_lora_base_classification_legacy(filename)
            guidelines = self._get_lora_guidelines_legacy(filename)
            
            try:
                file_size = os.path.getsize(file_path)
                file_size_mb = round(file_size / (1024 * 1024), 1)
            except:
                file_size_mb = 0
            
            loras.append({
                "filename": filename,
                "display_name": display_name,
                "path": file_path,
                "size_mb": file_size_mb,
                "type": "lora",
                "category": guidelines["category"],
                "supported_models": ["sdxl"],
                "compatible_bases": compatible_bases,
                "source": "local",
                "is_nsfw": self._detect_nsfw_legacy(filename),
                "scale_config": {
                    "default_scale": guidelines["recommended_scale"],
                    "recommended_scale": guidelines["recommended_scale"],
                    "min_scale": guidelines["scale_range"]["min"],
                    "max_scale": guidelines["scale_range"]["max"],
                    "optimal_range": [
                        max(guidelines["scale_range"]["min"], guidelines["recommended_scale"] - 0.3),
                        min(guidelines["scale_range"]["max"], guidelines["recommended_scale"] + 0.3)
                    ]
                },
                "usage_tips": {
                    "description": guidelines["description"],
                    "range_info": f"Supported range: {guidelines['scale_range']['min']} to {guidelines['scale_range']['max']}",
                    "category_info": f"{guidelines['category'].replace('_', ' ').title()} LoRA",
                    "compatibility_info": f"Compatible with: {', '.join(compatible_bases)}"
                },
                "usage_count": 0,
                "last_used": None,
            })
        
        return loras
    
    def _get_lora_base_classification_legacy(self, filename: str) -> List[str]:
        """Legacy LoRA base classification"""
        filename_lower = filename.lower()
        
        # NoobAI-specific LoRAs
        if any(noobai_lora in filename_lower for noobai_lora in [
            "n1-monster hunter", "n1-naturecock", "n1-toothless"
        ]):
            return ["noobai"]
        
        # Illustrious-specific LoRAs
        if any(illustrious_lora in filename_lower for illustrious_lora in ["sweaty_v_5"]):
            return ["illustrious"]
        
        # General compatibility
        return ["noobai", "ponyxl", "illustrious", "sdxl-base"]
    
    def _get_lora_guidelines_legacy(self, filename: str) -> Dict[str, Any]:
        """Legacy LoRA guidelines"""
        filename_lower = filename.lower()
        
        # Detail enhancement LoRAs
        if any(word in filename_lower for word in ["detail", "details"]):
            return {
                "recommended_scale": 0.7,
                "scale_range": {"min": -5.0, "max": 5.0},
                "description": "Detail enhancer - use 0.5-1.0 range, can go lower for subtle effects",
                "category": "detail_enhancement"
            }
        
        # Skin tone sliders
        elif any(word in filename_lower for word in ["skin_tone", "skintone"]) and "slider" in filename_lower:
            return {
                "recommended_scale": 0.0,
                "scale_range": {"min": -5.0, "max": 5.0},
                "description": "Skin tone slider - supports -5 to +5 range",
                "category": "skin_tone_slider"
            }
        
        # Default
        return {
            "recommended_scale": 1.0,
            "scale_range": {"min": -5.0, "max": 5.0},
            "description": "LoRA - typical range 0.5-1.5, full range available",
            "category": "general"
        }
    
    def _detect_nsfw_legacy(self, filename: str) -> bool:
        """Legacy NSFW detection"""
        filename_lower = filename.lower()
        nsfw_indicators = ["nsfw", "adult", "xxx", "erotic", "hentai", "naturecock"]
        return any(indicator in filename_lower for indicator in nsfw_indicators)

class EmbeddingManager:
    """High-level manager for embeddings"""
    
    def __init__(self, embeddings_dir: str = "embeddings"):
        self.embeddings_dir = embeddings_dir
        # No longer store db_client instance - use singleton
    
    async def get_available_embeddings(self) -> List[Dict[str, Any]]:
        """Get available embeddings with database metadata and filesystem fallback"""
        try:
            # Note: No EmbeddingService implemented yet in SQLAlchemy
            # For now, just return filesystem scan
            return await self._scan_filesystem_embeddings()
                
        except Exception as e:
            logger.error(f"Error getting embeddings from database: {e}")
            return await self._scan_filesystem_embeddings()
    
    async def _scan_filesystem_embeddings(self) -> List[Dict[str, Any]]:
        """Fallback filesystem scanning"""
        embeddings = []
        
        if not os.path.exists(self.embeddings_dir):
            return embeddings
        
        embedding_files = glob.glob(os.path.join(self.embeddings_dir, "*.safetensors"))
        
        for file_path in embedding_files:
            filename = os.path.basename(file_path)
            trigger_word = filename.replace(".safetensors", "").lower()
            display_name = filename.replace(".safetensors", "").replace("_", " ").replace("-", " ")
            
            try:
                file_size = os.path.getsize(file_path)
                file_size_kb = round(file_size / 1024, 1)
            except:
                file_size_kb = 0
            
            embeddings.append({
                "filename": filename,
                "trigger_word": trigger_word,
                "display_name": display_name,
                "path": file_path,
                "category": "negative" if "negative" in filename.lower() else "positive",
                "type": "negative" if "negative" in filename.lower() else "positive",
                "size_kb": file_size_kb,
                "format": "safetensors",
                "supported_models": ["sdxl"],
                "source": "local",
                "is_nsfw": False,
                "description": f"Embedding for {trigger_word}",
                "usage_count": 0,
            })
        
        return embeddings

class VaeManager:
    """High-level manager for VAE models"""
    
    def __init__(self, vaes_dir: str = "vaes"):
        self.vaes_dir = vaes_dir
        # No longer store db_client instance - use singleton
    
    async def get_available_vaes(self) -> List[Dict[str, Any]]:
        """Get available VAEs with database metadata and filesystem fallback"""
        try:
            # Get singleton database client
            db_client = await get_db_client()
            service = VaeService(db_client)
            db_vaes = await service.get_all_vaes(active_only=False)  # Get ALL VAEs, not just active ones
            
            if db_vaes:
                vaes = []
                for db_vae in db_vaes:
                    if os.path.exists(db_vae.file_path):
                        # Safely handle enum values that might already be strings
                        vae_type = db_vae.vae_type.value if hasattr(db_vae.vae_type, 'value') else str(db_vae.vae_type)
                        
                        vae_dict = {
                            "filename": db_vae.filename,
                            "display_name": db_vae.display_name,
                            "path": db_vae.file_path,
                            "size_mb": db_vae.file_size or 0,
                            "type": vae_type.lower(),
                            "supported_models": ["sdxl"] if vae_type == "SDXL" else ["sd15"],
                            "source": db_vae.source,
                            "is_default": db_vae.is_default,
                            "description": db_vae.description,
                        }
                        vaes.append(vae_dict)
                
                return vaes
            else:
                return await self._scan_filesystem_vaes()
                
        except Exception as e:
            logger.error(f"Error getting VAEs from database: {e}")
            return await self._scan_filesystem_vaes()
    
    async def _scan_filesystem_vaes(self) -> List[Dict[str, Any]]:
        """Fallback filesystem scanning"""
        vaes = []
        
        if not os.path.exists(self.vaes_dir):
            return vaes
        
        vae_files = glob.glob(os.path.join(self.vaes_dir, "*.safetensors"))
        
        for file_path in vae_files:
            filename = os.path.basename(file_path)
            display_name = filename.replace(".safetensors", "").replace("_", " ").replace("-", " ")
            
            try:
                file_size = os.path.getsize(file_path)
                file_size_mb = round(file_size / (1024 * 1024), 1)
            except:
                file_size_mb = 0
            
            vaes.append({
                "filename": filename,
                "display_name": display_name,
                "path": file_path,
                "size_mb": file_size_mb,
                "type": "sdxl",
                "supported_models": ["sdxl"],
                "source": "local",
                "is_default": "sdxlVAE" in filename,
                "description": None,
            })
        
        return vaes

class UpscalerManager:
    """High-level manager for upscaler models"""
    
    def __init__(self, upscalers_dir: str = "upscalers"):
        self.upscalers_dir = upscalers_dir
        # No longer store db_client instance - use singleton
    
    async def get_available_upscalers(self) -> List[Dict[str, Any]]:
        """Get available upscalers with database metadata and filesystem fallback"""
        try:
            # Get singleton database client
            db_client = await get_db_client()
            service = UpscalerService(db_client)
            db_upscalers = await service.get_all_upscalers(active_only=False)  # Get ALL upscalers, not just active ones
            
            if db_upscalers:
                upscalers = []
                for db_upscaler in db_upscalers:
                    if os.path.exists(db_upscaler.file_path):
                        # Safely handle enum values that might already be strings
                        upscaler_type = db_upscaler.upscaler_type.value if hasattr(db_upscaler.upscaler_type, 'value') else str(db_upscaler.upscaler_type)
                        
                        upscaler_dict = {
                            "filename": db_upscaler.filename,
                            "model_name": db_upscaler.model_name,
                            "path": db_upscaler.file_path,
                            "category": "general",  # Could be derived from path
                            "type": upscaler_type.lower().replace("_", "-"),
                            "scale_factor": db_upscaler.scale_factor,
                            "size_mb": db_upscaler.file_size or 0,
                            "format": db_upscaler.file_format,
                            "source": db_upscaler.source,
                            "supports_pickle": db_upscaler.file_format in ["pt", "pth"],
                            "tile_size": db_upscaler.tile_size,
                            "tile_pad": db_upscaler.tile_pad,
                            "supports_half": db_upscaler.supports_half,
                        }
                        upscalers.append(upscaler_dict)
                
                return upscalers
            else:
                return await self._scan_filesystem_upscalers()
                
        except Exception as e:
            logger.error(f"Error getting upscalers from database: {e}")
            return await self._scan_filesystem_upscalers()
    
    async def _scan_filesystem_upscalers(self) -> List[Dict[str, Any]]:
        """Fallback filesystem scanning"""
        upscalers = []
        
        if not os.path.exists(self.upscalers_dir):
            return upscalers
        
        upscaler_files = glob.glob(os.path.join(self.upscalers_dir, "*.pth")) + \
                        glob.glob(os.path.join(self.upscalers_dir, "*.pt"))
        
        for file_path in upscaler_files:
            filename = os.path.basename(file_path)
            model_name = filename.replace(".pth", "").replace(".pt", "").replace("_", " ").replace("-", " ")
            
            try:
                file_size = os.path.getsize(file_path)
                file_size_mb = round(file_size / (1024 * 1024), 1)
            except:
                file_size_mb = 0
            
            # Determine scale factor and type
            scale_factor = 4  # default
            upscaler_type = "general"
            
            if "x2" in filename.lower():
                scale_factor = 2
            elif "x4" in filename.lower():
                scale_factor = 4
            elif "x8" in filename.lower():
                scale_factor = 8
            
            if "anime" in filename.lower():
                upscaler_type = "anime"
            elif "photo" in filename.lower():
                upscaler_type = "photo"
            elif "face" in filename.lower():
                upscaler_type = "face"
            
            upscalers.append({
                "filename": filename,
                "model_name": model_name,
                "path": file_path,
                "category": "general",
                "type": upscaler_type,
                "scale_factor": scale_factor,
                "size_mb": file_size_mb,
                "format": filename.split(".")[-1],
                "source": "local",
                "supports_pickle": True,
                "tile_size": 512,
                "tile_pad": 10,
                "supports_half": True,
            })
        
        return upscalers 