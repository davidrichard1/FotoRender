"""
API v2 - Stateless, queue-based image generation API
Replaces the monolithic main.py with a scalable architecture
"""

import asyncio
import logging
import os
import uuid
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from queue_system import QueueManager, JobStatus
from sqlalchemy import select
import json

# Add database imports
from database.sqlalchemy_client import get_db_client
from database.sqlalchemy_models import SystemConfig, AiModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Image Generation API v2", version="2.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global queue manager
queue_manager = QueueManager()

# Pydantic models
class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    seed: int = -1
    sampler: str = "DPM++ 2M SDE Karras"
    model_name: str  # Add the missing model_name field!
    loras: list = []
    clip_skip: Optional[int] = None
    user_id: Optional[str] = None

class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    estimated_wait_time: Optional[int] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    eta: Optional[int] = None
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

class ApiModel(BaseModel):
    filename: str
    display_name: str
    path: str
    type: str = "checkpoint"
    capability: str = "general"
    source: str = "local"
    base_model: str = "SDXL"
    description: Optional[str] = None
    is_gated: bool = False
    file_size_mb: Optional[float] = None
    usage_count: int = 0
    last_used: Optional[str] = None

class ApiLora(BaseModel):
    filename: str
    display_name: str
    path: str
    category: str = "general"
    description: Optional[str] = None
    is_gated: bool = False
    trigger_words: Optional[List[str]] = None
    default_scale: float = 1.0
    min_scale: float = 0.0
    max_scale: float = 2.0
    usage_count: int = 0
    last_used: Optional[str] = None

class ModelSwitchRequest(BaseModel):
    model_name: str

# Add VAE request models
class VaeLoadRequest(BaseModel):
    vae_filename: str

# Add after the existing endpoints, before the end of the file
@app.post("/load-vae")
async def load_vae(request: VaeLoadRequest):
    """Load a specific VAE model"""
    try:
        vae_path = os.path.join("vaes", request.vae_filename)
        
        if not os.path.exists(vae_path):
            raise HTTPException(
                status_code=404, 
                detail=f"VAE file not found: {request.vae_filename}"
            )
        
        # In a real implementation, you would load the VAE into your model here
        # For now, we'll just return success
        return {
            "status": "success",
            "message": f"VAE {request.vae_filename} loaded successfully",
            "loaded_vae": request.vae_filename
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load VAE: {str(e)}")

@app.post("/reset-vae")
async def reset_vae():
    """Reset VAE to default"""
    try:
        # In a real implementation, you would reset the VAE in your model here
        # For now, we'll just return success
        return {
            "status": "success", 
            "message": "VAE reset to default successfully",
            "loaded_vae": "default"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset VAE: {str(e)}")

# Helper functions for asset management
def scan_models_directory():
    """Scan the models directory for available models"""
    models_dir = "models"
    models = []
    
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith(('.safetensors', '.ckpt', '.bin')):
                file_path = os.path.join(models_dir, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                models.append({
                    "filename": file,
                    "display_name": file.replace('.safetensors', '').replace('.ckpt', '').replace('.bin', ''),
                    "path": file_path,
                    "type": "checkpoint",
                    "capability": "general",
                    "source": "local",
                    "base_model": "SDXL",
                    "description": None,
                    "is_gated": False,
                    "file_size_mb": round(file_size, 2),
                    "usage_count": 0,
                    "last_used": None
                })
    
    return models

def scan_loras_directory():
    """Scan the loras directory for available LoRAs"""
    loras_dir = "loras"
    loras = []
    
    if os.path.exists(loras_dir):
        for file in os.listdir(loras_dir):
            if file.endswith('.safetensors'):
                file_path = os.path.join(loras_dir, file)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                
                loras.append({
                    "filename": file,
                    "display_name": file.replace('.safetensors', ''),
                    "path": file_path,
                    "category": "general",
                    "is_gated": False,
                    "usage_count": 0,
                    "last_used": None,
                    "size_mb": round(file_size / (1024 * 1024), 1),
                    # Nested structure expected by frontend
                    "scale_config": {
                        "min_scale": -5.0,
                        "max_scale": 5.0,
                        "recommended_scale": 1.0
                    },
                    "usage_tips": {
                        "description": f"LoRA model: {file.replace('.safetensors', '')}",
                        "trigger_words": []
                    }
                })
    
    return loras

def scan_vaes_directory():
    """Scan the vaes directory for available VAEs"""
    vaes_dir = "vaes"
    vaes = []
    
    if os.path.exists(vaes_dir):
        for file in os.listdir(vaes_dir):
            if file.endswith('.safetensors'):
                file_path = os.path.join(vaes_dir, file)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                display_name = file.replace('.safetensors', '')
                
                vaes.append({
                    "id": display_name.lower().replace(' ', '-').replace('_', '-'),  # Generate unique ID
                    "filename": file,
                    "displayName": display_name,
                    "description": f"VAE model: {display_name}",
                    "vaeType": "SDXL" if "sdxl" in file.lower() else "SD15" if "sd15" in file.lower() else "GENERAL",
                    "isActive": False,  # Default to inactive
                    "isDefault": "default" in file.lower(),  # Check if this is a default VAE
                    "fileSize": file_size,
                    "author": None,
                    "version": None,
                    "website": None,
                    "tags": [],
                    "usageCount": 0,
                    "lastUsed": None,
                    "createdAt": "2024-01-01T00:00:00Z",  # Placeholder
                    "updatedAt": "2024-01-01T00:00:00Z",  # Placeholder
                    "path": file_path
                })
    
    return vaes

def scan_upscalers_directory():
    """Scan the upscalers directory for available upscalers"""
    upscalers_dir = "upscalers"
    upscalers = []
    
    if os.path.exists(upscalers_dir):
        for file in os.listdir(upscalers_dir):
            if file.endswith(('.pth', '.pt')):
                file_path = os.path.join(upscalers_dir, file)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                display_name = file.replace('.pth', '').replace('.pt', '')
                
                # Determine upscaler type from filename
                upscaler_type = "REALESRGAN"
                if "esrgan" in file.lower():
                    upscaler_type = "ESRGAN" 
                elif "realesrgan" in file.lower():
                    upscaler_type = "REALESRGAN"
                elif "waifu" in file.lower():
                    upscaler_type = "WAIFU2X"
                else:
                    upscaler_type = "OTHER"
                
                # Determine scale factor from filename  
                scale_factor = 4  # Default
                if "x2" in file.lower():
                    scale_factor = 2
                elif "x4" in file.lower():
                    scale_factor = 4
                elif "x8" in file.lower():
                    scale_factor = 8
                
                upscalers.append({
                    "filename": file,
                    "model_name": display_name,
                    "scale_factor": scale_factor,
                    "type": upscaler_type,
                    "path": file_path,
                    "file_size": file_size
                })
    
    return upscalers

def get_available_samplers():
    """Get list of available samplers with full metadata"""
    samplers_data = [
        {
            "name": "DPM++ 2M SDE Karras",
            "category": "DPM",
            "speed": "MEDIUM",
            "quality": "HIGH",
            "steps": {"min": 15, "max": 50, "recommended": 30}
        },
        {
            "name": "DPM++ 2M SDE",
            "category": "DPM", 
            "speed": "MEDIUM",
            "quality": "HIGH",
            "steps": {"min": 15, "max": 50, "recommended": 30}
        },
        {
            "name": "DPM++ 2M",
            "category": "DPM",
            "speed": "FAST",
            "quality": "MEDIUM",
            "steps": {"min": 10, "max": 30, "recommended": 20}
        },
        {
            "name": "DPM++ SDE Karras", 
            "category": "DPM",
            "speed": "SLOW",
            "quality": "HIGH",
            "steps": {"min": 20, "max": 60, "recommended": 40}
        },
        {
            "name": "DPM++ SDE",
            "category": "DPM",
            "speed": "SLOW", 
            "quality": "HIGH",
            "steps": {"min": 20, "max": 60, "recommended": 40}
        },
        {
            "name": "Euler a",
            "category": "EULER",
            "speed": "FAST",
            "quality": "MEDIUM", 
            "steps": {"min": 10, "max": 40, "recommended": 25}
        },
        {
            "name": "Euler",
            "category": "EULER",
            "speed": "FAST",
            "quality": "MEDIUM",
            "steps": {"min": 10, "max": 40, "recommended": 25}
        },
        {
            "name": "LMS",
            "category": "OTHER",
            "speed": "MEDIUM",
            "quality": "MEDIUM",
            "steps": {"min": 15, "max": 45, "recommended": 30}
        },
        {
            "name": "Heun",
            "category": "OTHER", 
            "speed": "SLOW",
            "quality": "HIGH",
            "steps": {"min": 20, "max": 50, "recommended": 35}
        },
        {
            "name": "DPM2",
            "category": "DPM",
            "speed": "MEDIUM",
            "quality": "MEDIUM",
            "steps": {"min": 15, "max": 40, "recommended": 25}
        },
        {
            "name": "DPM2 a",
            "category": "DPM", 
            "speed": "MEDIUM",
            "quality": "MEDIUM",
            "steps": {"min": 15, "max": 40, "recommended": 25}
        },
        {
            "name": "DPM fast",
            "category": "DPM",
            "speed": "FAST",
            "quality": "LOW",
            "steps": {"min": 5, "max": 20, "recommended": 10}
        },
        {
            "name": "DPM adaptive",
            "category": "DPM",
            "speed": "SLOW",
            "quality": "HIGH", 
            "steps": {"min": 25, "max": 80, "recommended": 50}
        },
        {
            "name": "LMS Karras",
            "category": "KARRAS",
            "speed": "MEDIUM",
            "quality": "HIGH",
            "steps": {"min": 15, "max": 45, "recommended": 30}
        },
        {
            "name": "DPM2 Karras",
            "category": "KARRAS",
            "speed": "MEDIUM", 
            "quality": "HIGH",
            "steps": {"min": 15, "max": 40, "recommended": 25}
        },
        {
            "name": "DPM2 a Karras",
            "category": "KARRAS",
            "speed": "MEDIUM",
            "quality": "HIGH",
            "steps": {"min": 15, "max": 40, "recommended": 25}
        }
    ]
    
    # Convert to the format expected by the frontend
    samplers = []
    for i, sampler_data in enumerate(samplers_data):
        sampler_id = sampler_data["name"].lower().replace('+', 'plus').replace(' ', '-').replace('(', '').replace(')', '')
        samplers.append({
            "id": sampler_id,
            "name": sampler_data["name"],
            "displayName": sampler_data["name"],
            "description": f"{sampler_data['category']} sampler - {sampler_data['quality'].lower()} quality, {sampler_data['speed'].lower()} speed",
            "category": sampler_data["category"],
            "isActive": True,  # All samplers are active by default
            "isDefault": sampler_data["name"] == "DPM++ 2M SDE Karras",  # Set default
            "steps": sampler_data["steps"], 
            "speed": sampler_data["speed"],
            "quality": sampler_data["quality"],
            "usageCount": 0,
            "lastUsed": None,
            "createdAt": "2024-01-01T00:00:00Z",
            "updatedAt": "2024-01-01T00:00:00Z"
        })
    
    return samplers

def scan_embeddings_directory():
    """Scan the embeddings directory for available embeddings"""
    embeddings_dir = "embeddings"
    embeddings = []
    
    if os.path.exists(embeddings_dir):
        for file in os.listdir(embeddings_dir):
            if file.endswith('.safetensors'):
                display_name = file.replace('.safetensors', '')
                # Use the display name as both name and trigger word
                # Common embedding naming patterns
                trigger_word = display_name.replace('_', ' ').replace('-', ' ')
                
                embeddings.append({
                    "filename": file,
                    "display_name": display_name,
                    "name": display_name,
                    "trigger_word": trigger_word,
                    "description": f"Embedding: {display_name}",
                    "path": os.path.join(embeddings_dir, file)
                })
    
    return embeddings

# Startup/shutdown events
@app.on_event("startup")
async def startup():
    """Initialize connections on startup"""
    await queue_manager.connect()
    logger.info("API v2 started - queue-based architecture")

@app.on_event("shutdown") 
async def shutdown():
    """Clean up connections on shutdown"""
    await queue_manager.disconnect()
    logger.info("API v2 shutdown")

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        stats = await queue_manager.get_queue_stats()
        return {
            "status": "healthy",
            "version": "2.0.0",
            "queue_stats": stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

# Asset Management Endpoints
@app.get("/models")
async def get_models():
    """Get available models"""
    try:
        models = scan_models_directory()
        try:
            import main  # type: ignore
            current_model = getattr(main, "current_model", None)
        except Exception:
            current_model = None
        
        return {
            "models": models,
            "current_model": current_model
        }
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_filename}/config")
async def get_model_config(model_filename: str):
    """Get model-specific configuration including prompt defaults"""
    try:
        db_client = await get_db_client()
        
        async with db_client.get_session() as session:
            result = await session.execute(
                select(AiModel).where(AiModel.filename == model_filename)
            )
            model = result.scalar_one_or_none()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Return model configuration with prompt defaults
            config = {
                "filename": model.filename,
                "display_name": model.display_name,
                "default_sampler": model.default_sampler,
                "prompt_defaults": {
                    "positive_prompt": model.default_positive_prompt or "",
                    "negative_prompt": model.default_negative_prompt or "",
                    "usage_notes": model.usage_notes or "",
                    "suggested_prompts": model.suggested_prompts or [],
                    "suggested_tags": model.suggested_tags or [],
                    "suggested_negative_prompts": model.suggested_negative_prompts or [],
                    "suggested_negative_tags": model.suggested_negative_tags or []
                }
            }
            
            return config
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ModelConfigRequest(BaseModel):
    prompt_defaults: Optional[Dict[str, Any]] = None

@app.put("/models/{model_filename}/config")
async def update_model_config(model_filename: str, request: ModelConfigRequest):
    """Update model-specific configuration"""
    try:
        db_client = await get_db_client()
        
        async with db_client.get_session() as session:
            result = await session.execute(
                select(AiModel).where(AiModel.filename == model_filename)
            )
            model = result.scalar_one_or_none()
            
            if not model:
                raise HTTPException(status_code=404, detail="Model not found")
            
            # Update dedicated database fields for prompt defaults
            if request.prompt_defaults is not None:
                model.default_positive_prompt = request.prompt_defaults.get("positive_prompt", "")
                model.default_negative_prompt = request.prompt_defaults.get("negative_prompt", "")
                model.usage_notes = request.prompt_defaults.get("usage_notes", "")
                model.suggested_prompts = request.prompt_defaults.get("suggested_prompts", [])
                model.suggested_tags = request.prompt_defaults.get("suggested_tags", [])
                model.suggested_negative_prompts = request.prompt_defaults.get("suggested_negative_prompts", [])
                model.suggested_negative_tags = request.prompt_defaults.get("suggested_negative_tags", [])
            
            await session.commit()
            
            return {
                "success": True,
                "message": f"Updated configuration for {model.display_name}"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update model config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/loras")
async def get_loras():
    """Get available LoRAs"""
    try:
        loras = scan_loras_directory()
        return {"loras": loras}
    except Exception as e:
        logger.error(f"Failed to get LoRAs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/vaes")
async def get_vaes():
    """Get available VAEs"""
    try:
        vaes = scan_vaes_directory()
        return {"vaes": vaes}
    except Exception as e:
        logger.error(f"Failed to get VAEs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/upscalers")
async def get_upscalers():
    """Get available upscalers"""
    try:
        upscalers = scan_upscalers_directory()
        return {"upscalers": upscalers}
    except Exception as e:
        logger.error(f"Failed to get upscalers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/samplers")
async def get_samplers():
    """Get available samplers"""
    try:
        samplers = get_available_samplers()
        return {"samplers": samplers}
    except Exception as e:
        logger.error(f"Failed to get samplers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/embeddings")
async def get_embeddings():
    """Get available embeddings"""
    try:
        embeddings = scan_embeddings_directory()
        return {"embeddings": embeddings}
    except Exception as e:
        logger.error(f"Failed to get embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Queue statistics
@app.get("/queue/stats")
async def get_queue_stats():
    """Get current queue statistics"""
    try:
        stats = await queue_manager.get_queue_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Generate image (non-blocking)
@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """
    Submit an image generation job to the queue
    Returns immediately with a job ID for tracking
    """
    try:
        # Convert request to job data
        job_data = request.dict()
        
        # Add job to queue
        job_id = await queue_manager.add_job(job_data)
        
        # Get current queue stats for wait time estimation
        stats = await queue_manager.get_queue_stats()
        estimated_wait = stats.get("pending_jobs", 0) * 45  # 45 seconds per job estimate
        
        return GenerationResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Job queued successfully",
            estimated_wait_time=estimated_wait
        )
        
    except Exception as e:
        logger.error(f"Failed to queue generation job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Switch model
@app.post("/switch-model")
async def switch_model(request: ModelSwitchRequest):
    """Switch to a different model checkpoint"""
    try:
        # Lazy import to avoid circular deps
        import main  # type: ignore
        from main import load_model, get_available_models  # type: ignore

        # Prefer cached DB list, fallback to filesystem scan
        models = get_available_models()
        if not models:
            models = scan_models_directory()
        model_info = next((m for m in models if m["filename"] == request.model_name), None)
        if not model_info:
            raise HTTPException(status_code=404, detail="Model not found")

        success = load_model(model_info["path"], model_info.get("type", "sdxl"))
        if success:
            # Update cache so future calls list correct current model
            try:
                import main as _main  # type: ignore
                _main._cached_models = models
            except Exception:
                pass
            return {"success": True, "message": f"Switched to {request.model_name}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to switch model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Get job status
@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the current status of a generation job"""
    try:
        job_data = await queue_manager.get_job_status(job_id)
        
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        # Extract only the fields needed for JobStatusResponse
        status_data = {
            "job_id": job_id,  # Add the job_id which isn't in the Redis data
            "status": job_data.get("status", "pending"),
            "progress": float(job_data.get("progress", 0.0)),
            "eta": int(job_data["eta"]) if job_data.get("eta") else None,
            "result_url": job_data.get("result_url"),
            "error_message": job_data.get("error_message"),
            "created_at": job_data.get("created_at"),
            "started_at": job_data.get("started_at"),
            "completed_at": job_data.get("completed_at")
        }
        
        return JobStatusResponse(**status_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Cancel job
@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending generation job"""
    try:
        success = await queue_manager.cancel_job(job_id)
        
        if success:
            return {"message": "Job cancelled successfully"}
        else:
            return {"message": "Job could not be cancelled (may already be processing)"}
            
    except Exception as e:
        logger.error(f"Failed to cancel job: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket for real-time updates
@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time job progress updates
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for job {job_id}")
    
    last_progress: float = -1.0  # Track last progress to avoid redundant messages
    last_status: str | None = None
    try:
        while True:
            # Get current job status
            job_data = await queue_manager.get_job_status(job_id)
            
            if job_data:
                current_progress = float(job_data.get("progress", 0.0))
                current_status = job_data.get("status", "pending")
                # Only send when progress or status changed
                if current_progress != last_progress or current_status != last_status:
                    status_data = {
                        "job_id": job_id,
                        "status": current_status,
                        "progress": current_progress,
                        "eta": int(job_data["eta"]) if job_data.get("eta") else None,
                        "result_url": job_data.get("result_url"),
                        "error_message": job_data.get("error_message"),
                        "created_at": job_data.get("created_at"),
                        "started_at": job_data.get("started_at"),
                        "completed_at": job_data.get("completed_at")
                    }
                    await websocket.send_text(json.dumps(status_data))
                    last_progress = current_progress
                    last_status = current_status
                # Exit loop when job finished
                if current_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    break
            # Shorter sleep for finer granularity (250 ms)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for job {job_id}")
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}")
        await websocket.close()

# List recent jobs (optional - for debugging)
@app.get("/jobs")
async def list_recent_jobs(limit: int = 10):
    """List recent jobs (for admin/debugging)"""
    try:
        jobs = await queue_manager.get_recent_jobs(limit)
        return {"jobs": jobs}
    except Exception as e:
        logger.error(f"Failed to list recent jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Worker management
@app.get("/workers")
async def list_workers():
    """List active workers"""
    try:
        workers = await queue_manager.get_workers()
        active_workers = sum(1 for w in workers.values() if w.get("status") == "running")
        return {
            "workers": workers,
            "queue_stats": await queue_manager.get_queue_stats(),
            "total_workers": len(workers),
            "active_workers": active_workers
        }
    except Exception as e:
        logger.error(f"Failed to list workers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workers/start")
async def start_worker(worker_type: str = "local", gpu_id: int = 0):
    """Start a new worker"""
    try:
        # This would typically start a worker process
        # For now, return mock response
        return {
            "message": f"Worker started successfully (type: {worker_type}, GPU: {gpu_id})",
            "worker_id": f"worker_{worker_type}_{gpu_id}"
        }
    except Exception as e:
        logger.error(f"Failed to start worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/workers/{worker_id}")
async def stop_worker(worker_id: str):
    """Stop a worker"""
    try:
        # This would typically stop a worker process
        # For now, return mock response
        return {"message": f"Worker {worker_id} stopped successfully"}
    except Exception as e:
        logger.error(f"Failed to stop worker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoint for compatibility
@app.post("/generate-legacy")
async def generate_legacy():
    """Legacy endpoint - redirects to queue-based generation"""
    return {
        "status": "deprecated",
        "message": "Use /generate endpoint with queue-based processing"
    }

# ===== FAVORITE PROMPT ENDPOINTS =====

class FavoritePromptCreate(BaseModel):
    """Request schema for saving a generated prompt and its image"""
    title: str
    prompt: str
    negative_prompt: Optional[str] = ""
    description: Optional[str] = None
    # Generation parameters
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    sampler: str
    clip_skip: Optional[int] = None
    loras_used: Optional[List[Dict[str, Any]]] = None
    model_name: Optional[str] = None
    # Image data
    image_base64: Optional[str] = None  # PNG/JPEG encoded as base64 string
    image_url: Optional[str] = None  # Alternative: URL of existing image
    is_public: bool = False

class FavoritePromptSummary(BaseModel):
    """Lightweight response for list views"""
    id: str
    title: str
    prompt: str
    negative_prompt: Optional[str]
    image_url: str
    image_r2_url: Optional[str] = None  # Add R2 URL as fallback
    thumbnail_url: Optional[str] = None
    created_at: str

class FavoritePromptDetail(FavoritePromptSummary):
    """Detailed prompt info"""
    description: Optional[str]
    parameters: Dict[str, Any]
    loras_used: Optional[List[Dict[str, Any]]]
    model_name: Optional[str]
    tags: List[str]
    is_public: bool


@app.post("/favorite-prompts", response_model=FavoritePromptDetail)
async def save_prompt(payload: FavoritePromptCreate):
    """Save a generated prompt along with its image to R2 & DB"""
    try:
        import base64, re, aiohttp

        image_bytes: bytes
        if payload.image_base64:
            # Clean header if present and decode
            base64_data = re.sub('^data:image/\\w+;base64,', '', payload.image_base64)
            image_bytes = base64.b64decode(base64_data)
        elif payload.image_url:
            # Fetch image from URL
            async with aiohttp.ClientSession() as session_http:
                async with session_http.get(payload.image_url) as resp:
                    if resp.status != 200:
                        raise HTTPException(status_code=400, detail="Failed to fetch image from URL")
                    image_bytes = await resp.read()
        else:
            raise HTTPException(status_code=400, detail="Either image_base64 or image_url must be provided")

        # Upload to R2 with a human-friendly file name: <timestamp>_<slug>_<8-char uuid>.png
        from storage.r2_client import R2StorageClient  # Lazy import to avoid circular deps
        from datetime import datetime

        # Create slug from title (letters, numbers, dashes/underscores only)
        slug_raw = payload.title.strip().lower()
        slug = re.sub(r"[^a-z0-9_-]+", "-", slug_raw)[:60].strip("-") or "prompt"

        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        custom_filename = f"{timestamp}_{slug}_{short_uuid}.png"

        r2_client = R2StorageClient()
        upload_info = r2_client.upload_image(image_bytes, folder="prompts", filename=custom_filename)

        # Insert into DB
        db_client = await get_db_client()
        async with db_client.get_session() as session:
            from database.sqlalchemy_models import FavoritePrompt
            new_prompt = FavoritePrompt(
                id=str(uuid.uuid4()),
                title=payload.title,
                prompt=payload.prompt,
                negative_prompt=payload.negative_prompt,
                description=payload.description,
                width=payload.width,
                height=payload.height,
                steps=payload.steps,
                guidance_scale=payload.guidance_scale,
                seed=payload.seed,
                sampler=payload.sampler,
                clip_skip=payload.clip_skip,
                loras_used=payload.loras_used,
                image_r2_key=upload_info["r2_key"],
                image_r2_url=upload_info["r2_url"],
                image_cdn_url=upload_info.get("cdn_url"),
                image_width=upload_info["width"],
                image_height=upload_info["height"],
                image_size_bytes=upload_info["size_bytes"],
                image_format=upload_info["content_type"].split("/")[-1],
                thumbnail_r2_key=upload_info.get("thumbnail", {}).get("r2_key") if upload_info.get("thumbnail") else None,
                thumbnail_cdn_url=upload_info.get("thumbnail", {}).get("cdn_url") if upload_info.get("thumbnail") else None,
                tags=[],
                category=None,
                is_public=payload.is_public,
                is_featured=False,
                model_id=None  # To be resolved later
            )
            session.add(new_prompt)
            await session.commit()
            await session.refresh(new_prompt)

            result = FavoritePromptDetail(
                id=new_prompt.id,
                title=new_prompt.title,
                prompt=new_prompt.prompt,
                negative_prompt=new_prompt.negative_prompt,
                description=new_prompt.description,
                image_url=new_prompt.image_cdn_url or new_prompt.image_r2_url,
                thumbnail_url=new_prompt.thumbnail_cdn_url,
                created_at=new_prompt.created_at.isoformat(),
                parameters={
                    "width": new_prompt.width,
                    "height": new_prompt.height,
                    "steps": new_prompt.steps,
                    "guidance_scale": new_prompt.guidance_scale,
                    "seed": new_prompt.seed,
                    "sampler": new_prompt.sampler,
                    "clip_skip": new_prompt.clip_skip,
                },
                loras_used=new_prompt.loras_used,
                model_name=payload.model_name,
                tags=new_prompt.tags,
                is_public=new_prompt.is_public,
            )
            return result
    except Exception as e:
        logger.error(f"Failed to save prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def get_cdn_url_from_r2(r2_url: str, thumbnail_cdn_url: Optional[str] = None) -> str:
    """Convert R2 URL to CDN URL if possible"""
    if thumbnail_cdn_url and "foto.mylinkbuddy.com" in thumbnail_cdn_url:
        # Extract the path from R2 URL and use CDN domain
        import re
        # Extract everything after the bucket name
        match = re.search(r'/prompts/(.+)$', r2_url)
        if match:
            path = match.group(1)
            return f"https://foto.mylinkbuddy.com/prompts/{path}"
    return r2_url

@app.get("/favorite-prompts")
async def list_prompts(limit: int = 50, offset: int = 0):
    """List saved prompts with thumbnails"""
    try:
        db_client = await get_db_client()
        async with db_client.get_session() as session:
            from database.sqlalchemy_models import FavoritePrompt
            result = await session.execute(
                select(FavoritePrompt).order_by(FavoritePrompt.created_at.desc()).offset(offset).limit(limit)
            )
            prompts = result.scalars().all()
            prompts_list = [
                FavoritePromptSummary(
                    id=p.id,
                    title=p.title,
                    prompt=p.prompt,
                    negative_prompt=p.negative_prompt,
                    image_url=p.image_cdn_url or get_cdn_url_from_r2(p.image_r2_url, p.thumbnail_cdn_url),
                    image_r2_url=p.image_r2_url,  # Include R2 URL as fallback
                    thumbnail_url=p.thumbnail_cdn_url,
                    created_at=p.created_at.isoformat(),
                )
                for p in prompts
            ]
            
            # Return wrapped response consistent with other endpoints
            return {
                "prompts": prompts_list,
                "total": len(prompts_list),
                "limit": limit,
                "offset": offset
            }
    except Exception as e:
        logger.error(f"Failed to list prompts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/favorite-prompts/{prompt_id}", response_model=FavoritePromptDetail)
async def get_prompt(prompt_id: str):
    """Get detailed prompt info"""
    try:
        db_client = await get_db_client()
        async with db_client.get_session() as session:
            from database.sqlalchemy_models import FavoritePrompt  # type: ignore
            result = await session.execute(select(FavoritePrompt).where(FavoritePrompt.id == prompt_id))
            prompt_obj = result.scalar_one_or_none()
            if not prompt_obj:
                raise HTTPException(status_code=404, detail="Prompt not found")
            
            # Access all attributes while still in session context and store them
            prompt_id_val = prompt_obj.id
            title_val = prompt_obj.title
            prompt_val = prompt_obj.prompt
            negative_prompt_val = prompt_obj.negative_prompt
            description_val = prompt_obj.description
            image_cdn_url_val = prompt_obj.image_cdn_url
            image_r2_url_val = prompt_obj.image_r2_url
            thumbnail_cdn_url_val = prompt_obj.thumbnail_cdn_url
            created_at_val = prompt_obj.created_at
            width_val = prompt_obj.width
            height_val = prompt_obj.height
            steps_val = prompt_obj.steps
            guidance_scale_val = prompt_obj.guidance_scale
            seed_val = prompt_obj.seed
            sampler_val = prompt_obj.sampler
            clip_skip_val = prompt_obj.clip_skip
            loras_used_val = prompt_obj.loras_used
            model_id_val = prompt_obj.model_id
            tags_val = prompt_obj.tags
            is_public_val = prompt_obj.is_public
            
        # Build the response using stored values (outside session context)
        prompt_data = FavoritePromptDetail(
            id=prompt_id_val,
            title=title_val,
            prompt=prompt_val,
            negative_prompt=negative_prompt_val,
            description=description_val,
            image_url=image_cdn_url_val or get_cdn_url_from_r2(image_r2_url_val, thumbnail_cdn_url_val),
            image_r2_url=image_r2_url_val,  # Include R2 URL as fallback
            thumbnail_url=thumbnail_cdn_url_val,
            created_at=created_at_val.isoformat() if created_at_val else "",
            parameters={
                "width": width_val,
                "height": height_val,
                "steps": steps_val,
                "guidance_scale": guidance_scale_val,
                "seed": seed_val,
                "sampler": sampler_val,
                "clip_skip": clip_skip_val,
            },
            loras_used=loras_used_val or [],
            model_name=model_id_val,
            tags=tags_val or [],
            is_public=is_public_val,
        )
        
        return prompt_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== SYSTEM CONFIG CRUD ENDPOINTS =====

@app.get("/config")
async def list_config():
    """List all system configuration"""
    try:
        db_client = await get_db_client()
        
        async with db_client.get_session() as session:
            result = await session.execute(select(SystemConfig))
            configs = result.scalars().all()
        
            return {
                "configs": [
                    {
                        "id": config.id,
                        "key": config.key,
                        "value": config.value if not config.is_secret else "[HIDDEN]",
                        "description": config.description,
                        "category": config.category.upper() if config.category else "OTHER",
                        "dataType": config.type.upper() if config.type else "STRING",
                        "isDefault": False,  # Default value
                        "isEditable": not config.is_secret,  # Editable if not secret
                        "validation": None,
                        "lastModified": config.updated_at.isoformat() if config.updated_at else None,
                        "modifiedBy": None,  # Not tracked in current schema
                        "createdAt": config.created_at.isoformat() if config.created_at else "",
                        "updatedAt": config.updated_at.isoformat() if config.updated_at else ""
                    }
                    for config in configs
                ],
                "total_count": len(configs)
            }
        
    except Exception as e:
        logger.error(f"Error listing config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/config")
async def create_config(config_data: dict):
    """Create new system configuration"""
    try:
        db_client = await get_db_client()
        
        # Validate required fields
        required_fields = ["key", "value"]
        for field in required_fields:
            if field not in config_data:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        async with db_client.get_session() as session:
            # Check if config with key already exists
            existing_result = await session.execute(
                select(SystemConfig).where(SystemConfig.key == config_data["key"])
            )
            if existing_result.scalar_one_or_none():
                raise HTTPException(status_code=400, detail="Config with this key already exists")
            
            # Create Config
            config = SystemConfig(
                id=str(uuid.uuid4()),
                key=config_data["key"],
                value=config_data["value"],
                type=config_data.get("type", "string"),
                description=config_data.get("description"),
                category=config_data.get("category"),
                is_secret=config_data.get("is_secret", False)
            )
            
            session.add(config)
            await session.commit()
            await session.refresh(config)
            
            return {
                "status": "success",
                "message": f"Config '{config_data['key']}' created successfully",
                "config": {
                    "key": config.key,
                    "value": config.value if not config.is_secret else "[HIDDEN]",
                    "type": config.type,
                    "description": config.description,
                    "category": config.category,
                    "is_secret": config.is_secret
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.put("/config/{config_key}")
async def update_config(config_key: str, config_data: dict):
    """Update existing system configuration"""
    try:
        db_client = await get_db_client()
        
        async with db_client.get_session() as session:
            # Get Config
            result = await session.execute(
                select(SystemConfig).where(SystemConfig.key == config_key)
            )
            config = result.scalar_one_or_none()
            if not config:
                raise HTTPException(status_code=404, detail="Config not found")
            
            # Update fields
            updateable_fields = ["value", "type", "description", "category", "is_secret"]
            
            for field in updateable_fields:
                if field in config_data:
                    setattr(config, field, config_data[field])
            
            await session.commit()
            await session.refresh(config)
            
            return {
                "status": "success", 
                "message": f"Config '{config_key}' updated successfully",
                "config": {
                    "key": config.key,
                    "value": config.value if not config.is_secret else "[HIDDEN]",
                    "type": config.type,
                    "description": config.description,
                    "category": config.category,
                    "is_secret": config.is_secret
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/config/{config_key}")
async def delete_config(config_key: str):
    """Delete system configuration"""
    try:
        db_client = await get_db_client()
        
        async with db_client.get_session() as session:
            # Get Config
            result = await session.execute(
                select(SystemConfig).where(SystemConfig.key == config_key)
            )
            config = result.scalar_one_or_none()
            if not config:
                raise HTTPException(status_code=404, detail="Config not found")
            
            # Delete Config
            await session.delete(config)
            await session.commit()
            
            return {
                "status": "success",
                "message": f"Config '{config_key}' deleted successfully"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting config: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    args = parser.parse_args()
    
    logger.info(f"🚀 Starting Image Generation API v2.0.0 on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port) 