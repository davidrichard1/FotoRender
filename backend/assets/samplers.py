"""
Sampler management module for Foto Render
Replaces sampler-related functions from main.py with database-driven approach
"""

import logging
from typing import List, Dict, Optional, Any
from sqlalchemy import select
from database.sqlalchemy_client import get_db_client, SamplerService
from database.sqlalchemy_models import Sampler

logger = logging.getLogger(__name__)

class SamplerManager:
    """
    Sampler manager that provides database-driven sampler management
    Replaces SAMPLERS dict and get_scheduler() from main.py
    """
    
    def __init__(self):
        self.db_client = None
        self._cached_samplers = None
    
    async def _get_service(self) -> SamplerService:
        """Get Sampler service with database client"""
        if not self.db_client:
            self.db_client = await get_db_client()
        return SamplerService(self.db_client)
    
    async def get_available_samplers(self) -> List[Dict[str, Any]]:
        """
        Get available samplers from database
        Replaces SAMPLERS dict from main.py
        """
        try:
            service = await self._get_service()
            db_samplers = await service.get_all_samplers()
            
            samplers = []
            for db_sampler in db_samplers:
                sampler_dict = {
                    "name": db_sampler.name,
                    "display_name": db_sampler.display_name,
                    "description": db_sampler.description,
                    "scheduler_class": db_sampler.scheduler_class,
                    "config": db_sampler.config,
                    "is_default": db_sampler.is_default,
                    "speed_rating": db_sampler.speed_rating,
                    "quality_rating": db_sampler.quality_rating,
                    "usage_count": db_sampler.usage_count,
                }
                samplers.append(sampler_dict)
            
            # Cache the samplers for sync access
            self._cached_samplers = samplers
            return samplers
            
        except Exception as e:
            logger.error(f"Error getting samplers from database: {e}")
            # Fallback to hardcoded samplers if database fails
            fallback_samplers = await self._get_fallback_samplers()
            self._cached_samplers = fallback_samplers
            return fallback_samplers

    def get_available_samplers_sync(self) -> List[Dict[str, Any]]:
        """
        Get available samplers synchronously (for use in non-async contexts)
        Uses cached samplers or fallback if cache is empty
        """
        if self._cached_samplers is not None:
            return self._cached_samplers
        
        # If no cache available, return hardcoded fallback
        logger.warning("No cached samplers available, using hardcoded fallback")
        return [
            {
                "name": "DPM++ 2M SDE Karras",
                "display_name": "DPM++ 2M SDE Karras",
                "description": "DPM++ 2nd order multistep SDE sampler with Karras noise schedule",
                "scheduler_class": "DPMSolverMultistepScheduler",
                "config": {
                    "algorithm_type": "sde-dpmsolver++",
                    "use_karras_sigmas": True,
                    "solver_order": 2
                },
                "is_default": True,
                "speed_rating": 4,
                "quality_rating": 5,
                "usage_count": 0,
            }
        ]
    
    async def get_default_sampler(self) -> Optional[Dict[str, Any]]:
        """
        Get the default sampler
        """
        try:
            service = await self._get_service()
            db_sampler = await service.get_default_sampler()
            if db_sampler:
                return {
                    "name": db_sampler.name,
                    "display_name": db_sampler.display_name,
                    "description": db_sampler.description,
                    "scheduler_class": db_sampler.scheduler_class,
                    "config": db_sampler.config,
                    "is_default": True,
                    "speed_rating": db_sampler.speed_rating,
                    "quality_rating": db_sampler.quality_rating,
                }
            
            # Fallback to DPM++ 2M SDE Karras
            samplers = await self.get_available_samplers()
            for sampler in samplers:
                if sampler["name"] == "DPM++ 2M SDE Karras":
                    return sampler
            
            return samplers[0] if samplers else None
            
        except Exception as e:
            logger.error(f"Error getting default sampler: {e}")
            return None
    
    async def get_sampler_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific sampler by name
        Replaces lookup in SAMPLERS dict from main.py
        """
        samplers = await self.get_available_samplers()
        for sampler in samplers:
            if sampler["name"] == name:
                return sampler
        return None
    
    async def get_scheduler_config(self, sampler_name: str, original_scheduler_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get scheduler configuration for a sampler
        Replaces get_scheduler() function from main.py
        """
        sampler = await self.get_sampler_by_name(sampler_name)
        
        if not sampler:
            logger.warning(f"Unknown sampler: {sampler_name}, using DPM++ 2M SDE Karras")
            sampler = await self.get_sampler_by_name("DPM++ 2M SDE Karras")
        
        if sampler:
            # Merge original config with sampler-specific config
            merged_config = {**original_scheduler_config, **sampler["config"]}
            return {
                "scheduler_class": sampler["scheduler_class"],
                "config": merged_config
            }
        
        # Ultimate fallback
        return {
            "scheduler_class": "DPMSolverMultistepScheduler",
            "config": {
                **original_scheduler_config,
                "algorithm_type": "sde-dpmsolver++",
                "use_karras_sigmas": True,
                "solver_order": 2
            }
        }
    
    async def update_sampler_usage(self, sampler_name: str):
        """
        Update sampler usage statistics
        """
        try:
            if not self.db_client:
                self.db_client = await get_db_client()
            
            async with self.db_client.get_session() as session:
                result = await session.execute(
                    select(Sampler).where(Sampler.name == sampler_name)
                )
                sampler = result.scalar_one_or_none()
                if sampler:
                    sampler.usage_count += 1
                    await session.commit()
        except Exception as e:
            logger.warning(f"Could not update usage for sampler {sampler_name}: {e}")
    
    async def get_sampler_by_class(self, scheduler_class: str) -> Optional[Dict[str, Any]]:
        """
        Get sampler by scheduler class name
        """
        samplers = await self.get_available_samplers()
        for sampler in samplers:
            if sampler["scheduler_class"] == scheduler_class:
                return sampler
        return None
    
    async def get_samplers_by_rating(self, min_quality: int = 4, min_speed: int = 3) -> List[Dict[str, Any]]:
        """
        Get samplers filtered by quality and speed ratings
        """
        samplers = await self.get_available_samplers()
        return [
            sampler for sampler in samplers 
            if (sampler.get("quality_rating", 0) >= min_quality and 
                sampler.get("speed_rating", 0) >= min_speed)
        ]
    
    async def get_sampler_statistics(self) -> Dict[str, Any]:
        """
        Get sampler usage statistics
        """
        samplers = await self.get_available_samplers()
        
        total_samplers = len(samplers)
        total_usage = sum(s.get("usage_count", 0) for s in samplers)
        
        # Find most used
        most_used = sorted(
            samplers, 
            key=lambda x: x.get("usage_count", 0), 
            reverse=True
        )
        
        # Average ratings
        avg_quality = sum(s.get("quality_rating", 0) for s in samplers) / max(total_samplers, 1)
        avg_speed = sum(s.get("speed_rating", 0) for s in samplers) / max(total_samplers, 1)
        
        return {
            "total_samplers": total_samplers,
            "total_usage": total_usage,
            "most_used": most_used[:3],
            "avg_quality_rating": round(avg_quality, 2),
            "avg_speed_rating": round(avg_speed, 2),
            "default_sampler": await self.get_default_sampler()
        }
    
    async def _get_fallback_samplers(self) -> List[Dict[str, Any]]:
        """
        Fallback samplers if database is unavailable
        """
        return [
            {
                "name": "DPM++ 2M SDE Karras",
                "display_name": "DPM++ 2M SDE Karras",
                "description": "DPM++ 2nd order multistep SDE sampler with Karras noise schedule",
                "scheduler_class": "DPMSolverMultistepScheduler",
                "config": {
                    "algorithm_type": "sde-dpmsolver++",
                    "use_karras_sigmas": True,
                    "solver_order": 2
                },
                "is_default": True,
                "speed_rating": 4,
                "quality_rating": 5,
                "usage_count": 0,
            },
            {
                "name": "DPM++ 2SA Karras",
                "display_name": "DPM++ 2SA Karras",
                "description": "DPM++ 2nd order sampler with Karras noise schedule",
                "scheduler_class": "DPMSolverMultistepScheduler",
                "config": {
                    "algorithm_type": "dpmsolver++",
                    "use_karras_sigmas": True,
                    "solver_order": 2
                },
                "is_default": False,
                "speed_rating": 4,
                "quality_rating": 5,
                "usage_count": 0,
            },
            {
                "name": "DPM++ 2M",
                "display_name": "DPM++ 2M",
                "description": "DPM++ 2nd order multistep sampler",
                "scheduler_class": "DPMSolverMultistepScheduler",
                "config": {
                    "algorithm_type": "dpmsolver++",
                    "solver_order": 2
                },
                "is_default": False,
                "speed_rating": 4,
                "quality_rating": 4,
                "usage_count": 0,
            },
            {
                "name": "Euler a",
                "display_name": "Euler Ancestral",
                "description": "Euler ancestral sampler",
                "scheduler_class": "EulerAncestralDiscreteScheduler",
                "config": {},
                "is_default": False,
                "speed_rating": 5,
                "quality_rating": 3,
                "usage_count": 0,
            },
            {
                "name": "Euler",
                "display_name": "Euler",
                "description": "Euler sampler",
                "scheduler_class": "EulerDiscreteScheduler",
                "config": {},
                "is_default": False,
                "speed_rating": 5,
                "quality_rating": 3,
                "usage_count": 0,
            },
            {
                "name": "DDIM",
                "display_name": "DDIM",
                "description": "Denoising Diffusion Implicit Models sampler",
                "scheduler_class": "DDIMScheduler",
                "config": {},
                "is_default": False,
                "speed_rating": 3,
                "quality_rating": 4,
                "usage_count": 0,
            },
            {
                "name": "LMS",
                "display_name": "LMS",
                "description": "Linear multistep sampler",
                "scheduler_class": "LMSDiscreteScheduler",
                "config": {},
                "is_default": False,
                "speed_rating": 3,
                "quality_rating": 3,
                "usage_count": 0,
            }
        ] 