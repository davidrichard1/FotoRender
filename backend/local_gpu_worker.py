#!/usr/bin/env python3
"""
Local GPU Worker - EXACT integration with main.py functions
Mirrors main.py lifespan initialization and generation logic
"""

import asyncio
import logging
import os
import sys
import signal
import torch
import base64
import io
from pathlib import Path
from typing import TYPE_CHECKING
from queue_system import QueueManager, JobStatus
import time
from datetime import datetime
import contextlib

# Import your existing functions from main.py - EXACT same imports as main.py uses
try:
    # Core generation functions
    from main import (
        load_model, 
        load_multiple_loras_to_pipeline,
        unload_loras_from_pipeline,
        get_scheduler,
        encode_long_prompt_with_embeddings,
        get_available_models,
        get_available_loras,
        get_available_loras_async,  # Fix: Import the async version
        load_embeddings_to_pipeline_async,
        # Database managers to populate cache - EXACT same as main.py lifespan
        model_manager,
        lora_manager,
        vae_manager,
        upscaler_manager,
        embedding_manager,
        sampler_manager
    )
    
    # Import main module to access global variables
    if TYPE_CHECKING:
        import main
    else:
        import main  # type: ignore
    
except ImportError as e:
    logging.error(f"Could not import from main.py: {e}")
    logging.error("Make sure main.py is in the same directory")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LocalGPUWorker:
    def __init__(self, worker_id: str = "local-gpu-1"):
        self.worker_id = worker_id
        self.queue_manager = QueueManager()
        self.is_running = False
        self.current_job_id = None
        
        # These will be set to reference main.py's globals
        self.pipeline = None
        self.current_model = None
        
    async def start(self):
        """Start the local GPU worker with EXACT main.py initialization"""
        logger.info(f"Starting local GPU worker {self.worker_id}")
        
        # Check GPU availability
        if not torch.cuda.is_available():
            logger.error("CUDA not available! Local GPU worker requires GPU.")
            sys.exit(1)
            
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        
        # Connect to queue
        await self.queue_manager.connect()
        
        # CRITICAL: Initialize EXACTLY like main.py lifespan function
        await self._initialize_exactly_like_main_py()
        
        self.is_running = True
        
        # Register worker info and start heartbeat task
        await self.queue_manager.update_worker_info(self.worker_id, {
            "id": self.worker_id,
            "type": "local",
            "status": "running",
            "gpu_id": 0,
            "jobs_processed": 0,
            "started_at": datetime.utcnow().isoformat()
        })
        self.heartbeat_task = asyncio.create_task(self._heartbeat())
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start processing jobs
        await self._process_jobs()
        
    async def _initialize_exactly_like_main_py(self):
        """
        EXACT COPY of main.py lifespan initialization
        This mirrors the FastAPI lifespan function 1:1
        """
        logger.info("🗄️ Starting up - PURE DATABASE-DRIVEN MODE (no hardcoded NSFW references)")
        
        # 🗄️ Pre-populate GLOBAL caches for pure database-driven sync functions
        # EXACT same order and approach as main.py lifespan
        try:
            logger.info("📊 Pre-loading samplers cache...")
            await sampler_manager.get_available_samplers()
            logger.info("📊 Samplers cache populated successfully")
        except Exception as e:
            logger.warning(f"Could not populate samplers cache: {e}")
        
        try:
            logger.info("🗄️ Pre-loading MODELS cache...")
            main._cached_models = await model_manager.get_available_models()
            logger.info(f"🗄️ Models cache populated: {len(main._cached_models)} models")
        except Exception as e:
            logger.warning(f"Could not populate models cache: {e}")
            main._cached_models = []
        
        try:
            logger.info("🎨 Pre-loading LORAS cache...")
            main._cached_loras = await lora_manager.get_available_loras()
            logger.info(f"🎨 LoRAs cache populated: {len(main._cached_loras)} LoRAs")
        except Exception as e:
            logger.warning(f"Could not populate LoRAs cache: {e}")
            main._cached_loras = []
        
        try:
            logger.info("🎭 Pre-loading VAES cache...")
            main._cached_vaes = await vae_manager.get_available_vaes()
            logger.info(f"🎭 VAEs cache populated: {len(main._cached_vaes)} VAEs")
        except Exception as e:
            logger.warning(f"Could not populate VAEs cache: {e}")
            main._cached_vaes = []
        
        try:
            logger.info("🔍 Pre-loading UPSCALERS cache...")
            main._cached_upscalers = await upscaler_manager.get_available_upscalers()
            logger.info(f"🔍 Upscalers cache populated: {len(main._cached_upscalers)} upscalers")
        except Exception as e:
            logger.warning(f"Could not populate upscalers cache: {e}")
            main._cached_upscalers = []
        
        try:
            logger.info("📚 Pre-loading EMBEDDINGS cache...")
            main._cached_embeddings = await embedding_manager.get_available_embeddings()
            logger.info(f"📚 Embeddings cache populated: {len(main._cached_embeddings)} embeddings")
        except Exception as e:
            logger.warning(f"Could not populate embeddings cache: {e}")
            main._cached_embeddings = []
        
        # Load default model EXACTLY like main.py lifespan
        models = main._cached_models
        if models:
            # 🗄️ DATABASE-DRIVEN: Load first available model (no hardcoded NSFW preferences)
            default_model = models[0]["path"]
            default_model_type = models[0].get("type", "sdxl")
            
            logger.info(f"🗄️ Loading default model: {models[0].get('display_name', 'Unknown')}")
            success = load_model(default_model, default_model_type)
            
            # CRITICAL: Load embeddings after model is loaded - EXACT same as main.py
            if success and main.pipeline is not None:
                logger.info("📚 Loading embeddings at startup...")
                embedding_result = await load_embeddings_to_pipeline_async(main.pipeline)
                if embedding_result["loaded"] > 0:
                    logger.info(f"📚 Startup: Loaded {embedding_result['loaded']} embeddings")
                else:
                    logger.info("📚 Startup: No embeddings found to load")
                    
                # Set our references to main.py globals
                self.pipeline = main.pipeline
                self.current_model = main.current_model
                
                logger.info(f"✅ Worker initialization complete - model: {self.current_model}")
            else:
                raise Exception("Failed to load default model")
        else:
            raise Exception("No models found in models directory")
            
    async def stop(self):
        """Stop the worker gracefully"""
        logger.info(f"Stopping local GPU worker {self.worker_id}")
        self.is_running = False
        
        if self.current_job_id:
            await self.queue_manager.update_job_status(
                self.current_job_id, 
                JobStatus.CANCELLED,
                error_message="Worker shutdown"
            )
            
        await self.queue_manager.disconnect()
        
        if hasattr(self, "heartbeat_task"):
            self.heartbeat_task.cancel()
            with contextlib.suppress(Exception):
                await self.heartbeat_task
        await self.queue_manager.update_worker_info(self.worker_id, {
            "status": "stopped"
        })
        
        # Cleanup GPU memory using main.py approach
        if main.pipeline is not None:
            del main.pipeline
            torch.cuda.empty_cache()
            
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop())
        
    async def _process_jobs(self):
        """Main job processing loop"""
        logger.info(f"Local GPU worker {self.worker_id} ready for jobs")
        
        while self.is_running:
            try:
                # Get next job from queue
                job = await self.queue_manager.get_next_job()
                
                if not job:
                    continue
                    
                logger.info(f"Processing job {job.id} on local GPU")
                self.current_job_id = job.id
                
                # Update status to processing
                await self.queue_manager.update_job_status(
                    job.id, 
                    JobStatus.PROCESSING
                )
                
                # Process the job using EXACT main.py logic
                await self._process_single_job(job)
                
                self.current_job_id = None
                
            except Exception as e:
                logger.error(f"Error processing job: {e}")
                if self.current_job_id:
                    await self.queue_manager.update_job_status(
                        self.current_job_id,
                        JobStatus.FAILED,
                        error_message=str(e)
                    )
                    
    async def _process_single_job(self, job):
        """
        Process a single job using EXACT main.py generation logic
        Mirrors the /generate endpoint logic 1:1
        """
        try:
            # Check if pipeline is loaded (use main.py global)
            if main.pipeline is None:
                raise Exception("No model loaded")
                
            # WINDOWS FIX: Set Windows-specific event loop policy
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
                logger.info("🪟 Applied Windows asyncio fix (WindowsSelectorEventLoopPolicy)")
            
            # REAL SOLUTION: Pipeline callbacks in thread using run_in_executor()!
            logger.info("🚀 Using REAL pipeline callbacks with run_in_executor() - WebSocket docs approved!")
            
            # Capture the currently running event loop to use inside threaded callbacks
            parent_loop = asyncio.get_running_loop()

            def create_threaded_callback():
                """Create callback that safely updates progress from a background thread"""
                def callback_wrapper(step: int, timestep: int, latents):
                    try:
                        progress = (step / job.num_inference_steps) * 100
                        eta = (job.num_inference_steps - step)
                        # Schedule the coroutine on the original asyncio loop
                        asyncio.run_coroutine_threadsafe(
                            self.queue_manager.update_job_progress(job.id, progress, eta),
                            parent_loop
                        )
                    except Exception as e:
                        logger.error(f"Progress callback error: {e}")
                return callback_wrapper
            
            progress_callback = create_threaded_callback()
            
            # EXACT main.py scheduler logic
            if job.sampler != "DPM++ 2M SDE Karras":  # Only change if different from default
                scheduler = get_scheduler(job.sampler, main.pipeline.scheduler.config)
                main.pipeline.scheduler = scheduler
                logger.info(f"Using sampler: {job.sampler}")
            
            # EXACT main.py seed generation
            generator = None
            actual_seed = job.seed
            if job.seed == -1:
                actual_seed = int(torch.randint(0, 2**32, (1,)).item())
            
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(int(actual_seed))
            
            logger.info(f"🎨 Generating image with prompt: {job.prompt[:50]}...")
            logger.info(f"Using sampler: {job.sampler}, Steps: {job.num_inference_steps}, Guidance: {job.guidance_scale}, CLIP Skip: {job.clip_skip or 'Default'}")
            
            # EXACT main.py LoRA handling
            loaded_loras = []
            lora_processing_info = "No LoRAs used"
            
            if job.loras and len(job.loras) > 0:
                logger.info(f"🎨 Processing {len(job.loras)} LoRA(s) for this generation")
                
                # Unload any existing LoRAs first for clean state
                unload_loras_from_pipeline(main.pipeline)
                
                # Prepare LoRAs for batch loading (EXACT main.py logic)
                available_loras = await get_available_loras_async()
                lora_dict = {lora["filename"]: lora for lora in available_loras}
                
                loras_to_load = []
                
                for lora_request in job.loras:
                    lora_filename = lora_request.get("filename", "")
                    lora_scale = lora_request.get("scale", 1.0)
                    
                    if lora_filename in lora_dict:
                        lora_info = lora_dict[lora_filename]
                        loras_to_load.append({
                            "path": lora_info["path"],
                            "scale": lora_scale,
                            "name": lora_info["display_name"]
                        })
                        loaded_loras.append({
                            "name": lora_info["display_name"], 
                            "scale": lora_scale,
                            "filename": lora_filename
                        })
                    else:
                        logger.warning(f"   ⚠️ LoRA not found: {lora_filename}")
                
                # Load all LoRAs together with proper multi-LoRA support
                if loras_to_load:
                    success = load_multiple_loras_to_pipeline(main.pipeline, loras_to_load)
                    if success:
                        logger.info(f"   ✅ Successfully loaded {len(loras_to_load)} LoRA(s) with proper scaling")
                    else:
                        logger.warning(f"   ⚠️ Failed to load some LoRAs")
                        loaded_loras = []  # Clear if loading failed
                
                if loaded_loras:
                    lora_names = [f"{lora['name']}({lora['scale']})" for lora in loaded_loras]
                    lora_processing_info = f"LoRAs applied: {', '.join(lora_names)}"
                    logger.info(f"🎨 {len(loaded_loras)} LoRA(s) successfully loaded for generation")
            
            # EXACT main.py token count logic
            try:
                prompt_tokens = len(main.pipeline.tokenizer.encode(job.prompt))
            except Exception as e:
                logger.warning(f"⚠️ Could not get token count: {e}, estimating instead")
                prompt_tokens = len(job.prompt) // 4
                
            logger.info(f"🚀 Processing ~{prompt_tokens} tokens (FULL PROMPT SUPPORT - NO LIMITS!)")
            
            # EXACT main.py chunking decision
            CHUNK_THRESHOLD = 75  # Leave room for BOS/EOS tokens
            should_chunk = prompt_tokens > CHUNK_THRESHOLD or False  # No force_chunking in job
            
            if prompt_tokens > CHUNK_THRESHOLD:
                logger.info(f"🔥 Long prompt detected ({prompt_tokens} tokens) - using chunked processing with {((prompt_tokens + 74) // 75)} chunks")
            
            logger.info(f"🎯 SDXL CHUNK DECISION:")
            logger.info(f"   📊 Tokens: {prompt_tokens}")
            logger.info(f"   📏 Threshold: {CHUNK_THRESHOLD}")
            logger.info(f"   ✅ Will use chunking: {should_chunk}")
            
            # Handle SDXL generation with EXACT main.py logic
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            if should_chunk:
                logger.info(f"🔥 CHUNKED MODE ACTIVATED! Using advanced long prompt encoding for {prompt_tokens} tokens")
                
                try:
                    logger.info(f"🔧 Attempting chunked processing for prompt with {prompt_tokens} tokens...")
                    
                    # Process prompts with TRUE embeddings approach - EXACT main.py implementation
                    prompt_embeds_1, prompt_embeds_2, pooled_prompt_embeds, pos_chunk_count = encode_long_prompt_with_embeddings(
                        main.pipeline, job.prompt, device=device, clip_skip=job.clip_skip
                    )
                    logger.info(f"   ✅ Positive prompt processed with unlimited tokens!")
                    
                    negative_prompt_embeds_1, negative_prompt_embeds_2, negative_pooled_prompt_embeds, neg_chunk_count = encode_long_prompt_with_embeddings(
                        main.pipeline, job.negative_prompt, device=device, clip_skip=job.clip_skip
                    )
                    logger.info(f"   ✅ Negative prompt processed with unlimited tokens!")
                    
                    # CRITICAL: SDXL requires BOTH text encoders concatenated along CHANNEL dimension - EXACT main.py
                    logger.info(f"   🔗 Concatenating SDXL dual-encoder embeddings along CHANNEL dimension:")
                    logger.info(f"      prompt_embeds_1: {prompt_embeds_1.shape} (CLIP ViT-L)")
                    logger.info(f"      prompt_embeds_2: {prompt_embeds_2.shape} (OpenCLIP ViT-bigG)")
                    
                    # SDXL: Concatenate along CHANNEL dimension (dim=2) - EXACT main.py
                    prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=2)
                    negative_prompt_embeds = torch.cat([negative_prompt_embeds_1, negative_prompt_embeds_2], dim=2)
                    
                    logger.info(f"   ✅ SDXL channel concatenation complete:")
                    logger.info(f"      Final prompt_embeds: {prompt_embeds.shape} (should be [1,77,2048])")
                    logger.info(f"      Final negative_prompt_embeds: {negative_prompt_embeds.shape}")
                    
                    # CRITICAL: Ensure both embeddings have the same sequence length - EXACT main.py
                    if prompt_embeds.shape[1] != negative_prompt_embeds.shape[1]:
                        logger.warning(f"Sequence length mismatch: positive={prompt_embeds.shape[1]}, negative={negative_prompt_embeds.shape[1]}")
                        max_len = max(prompt_embeds.shape[1], negative_prompt_embeds.shape[1])
                        
                        if prompt_embeds.shape[1] < max_len:
                            pad_size = max_len - prompt_embeds.shape[1]
                            padding = torch.zeros(1, pad_size, prompt_embeds.shape[2], device=prompt_embeds.device, dtype=prompt_embeds.dtype)
                            prompt_embeds = torch.cat([prompt_embeds, padding], dim=1)
                            logger.info(f"   🔧 Padded positive embeddings to {prompt_embeds.shape}")
                        
                        if negative_prompt_embeds.shape[1] < max_len:
                            pad_size = max_len - negative_prompt_embeds.shape[1]
                            padding = torch.zeros(1, pad_size, negative_prompt_embeds.shape[2], device=negative_prompt_embeds.device, dtype=negative_prompt_embeds.dtype)
                            negative_prompt_embeds = torch.cat([negative_prompt_embeds, padding], dim=1)
                            logger.info(f"   🔧 Padded negative embeddings to {negative_prompt_embeds.shape}")
                    
                    # CRITICAL: SDXL requires pooled embeddings - EXACT main.py validation
                    if pooled_prompt_embeds is None:
                        raise RuntimeError("SDXL requires pooled_prompt_embeds when prompt_embeds are provided")
                    
                    pooled_prompt_embeds = pooled_prompt_embeds.to(device)
                    if len(pooled_prompt_embeds.shape) != 2 or pooled_prompt_embeds.shape[0] != 1:
                        logger.warning(f"Fixing pooled_prompt_embeds shape: {pooled_prompt_embeds.shape}")
                        if len(pooled_prompt_embeds.shape) == 1:
                            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)
                        elif pooled_prompt_embeds.shape[0] != 1:
                            pooled_prompt_embeds = pooled_prompt_embeds[:1]
                    
                    if negative_pooled_prompt_embeds is None:
                        raise RuntimeError("SDXL requires negative_pooled_prompt_embeds when negative_prompt_embeds are provided")
                    
                    negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.to(device)
                    if len(negative_pooled_prompt_embeds.shape) != 2 or negative_pooled_prompt_embeds.shape[0] != 1:
                        logger.warning(f"Fixing negative_pooled_prompt_embeds shape: {negative_pooled_prompt_embeds.shape}")
                        if len(negative_pooled_prompt_embeds.shape) == 1:
                            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.unsqueeze(0)
                        elif negative_pooled_prompt_embeds.shape[0] != 1:
                            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds[:1]
                    
                    # HYBRID APPROACH: Real generation + simulated progress (for RTX 5090 speed!)
                    with torch.inference_mode():
                        logger.info(f"   🚀 Starting SDXL generation with REAL callbacks in thread...")
                        
                        # REAL PIPELINE GENERATION with REAL CALLBACKS in thread!
                        import concurrent.futures
                        
                        def run_real_generation():
                            """Run REAL generation with REAL callbacks in separate thread"""
                            print("🔥 REAL generation with REAL callbacks started in thread!")
                            return main.pipeline(
                                prompt_embeds=prompt_embeds,  # Single concatenated embeddings - EXACT main.py
                                negative_prompt_embeds=negative_prompt_embeds,  # Single concatenated negative embeddings
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                width=job.width,
                                height=job.height,
                                num_inference_steps=job.num_inference_steps,
                                guidance_scale=job.guidance_scale,
                                generator=generator,
                                callback=progress_callback,  # REAL CALLBACK!
                                callback_steps=1,
                                cross_attention_kwargs={"scale": 1.0},  # PEFT integration compatibility
                                output_type="pil"
                            )
                        
                        # Run generation with real callbacks in thread (WebSocket docs approved!)
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            result = await loop.run_in_executor(executor, run_real_generation)
                        
                        print("🔥 REAL generation with REAL callbacks completed!")
                        await self.queue_manager.update_job_progress(job.id, 100.0, 0)
                    
                    logger.info(f"   ✅ SDXL generation completed successfully!")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Chunked prompt processing failed: {type(e).__name__}: {str(e)[:100]}...")
                    logger.info("🔄 Falling back to standard SDXL generation (prompt will be truncated to 77 tokens)")
                    
                    # Fallback to standard generation if long prompt fails - EXACT main.py
                    with torch.inference_mode():
                        def run_fallback_generation():
                            """Run fallback generation with REAL callbacks in thread"""
                            print("🔥 FALLBACK generation with REAL callbacks in thread!")
                            return main.pipeline(
                                prompt=job.prompt,  # Use full prompt, let SDXL handle truncation
                                negative_prompt=job.negative_prompt,
                                width=job.width,
                                height=job.height,
                                num_inference_steps=job.num_inference_steps,
                                guidance_scale=job.guidance_scale,
                                generator=generator,
                                callback=progress_callback,  # REAL CALLBACK!
                                callback_steps=1,
                                cross_attention_kwargs={"scale": 1.0},  # PEFT integration compatibility
                                output_type="pil"
                            )
                        
                        # Run fallback with real callbacks in thread
                        loop = asyncio.get_event_loop()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            result = await loop.run_in_executor(executor, run_fallback_generation)
                        
                        await self.queue_manager.update_job_progress(job.id, 100.0, 0)
                    logger.info("✅ Fallback generation completed successfully")
            else:
                logger.info(f"✅ Using standard SDXL generation for {prompt_tokens} tokens")
                
                # Use standard generation for short prompts - EXACT main.py
                import concurrent.futures
                with torch.inference_mode():
                    def run_standard_generation():
                        """Run standard generation with REAL callbacks in thread"""
                        print("🔥 STANDARD generation with REAL callbacks in thread!")
                        return main.pipeline(
                            prompt=job.prompt,
                            negative_prompt=job.negative_prompt,
                            width=job.width,
                            height=job.height,
                            num_inference_steps=job.num_inference_steps,
                            guidance_scale=job.guidance_scale,
                            generator=generator,
                            callback=progress_callback,  # REAL CALLBACK!
                            callback_steps=1,
                            cross_attention_kwargs={"scale": 1.0},  # PEFT integration compatibility
                            output_type="pil"
                        )
                    
                    # Run standard with real callbacks in thread
                    loop = asyncio.get_event_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        result = await loop.run_in_executor(executor, run_standard_generation)
                    
                    await self.queue_manager.update_job_progress(job.id, 100.0, 0)
            
                         # EXACT main.py image extraction logic
            image = None
            
            try:
                # Handle different pipeline return types robustly - EXACT main.py
                if isinstance(result, dict):
                    if 'images' in result and isinstance(result['images'], (list, tuple)) and len(result['images']) > 0:
                        image = result['images'][0]
                    elif 'samples' in result and isinstance(result['samples'], (list, tuple)) and len(result['samples']) > 0:
                        image = result['samples'][0]
                elif hasattr(result, 'images'):
                    images_attr = getattr(result, 'images', None)
                    if images_attr is not None and isinstance(images_attr, (list, tuple)) and len(images_attr) > 0:
                        image = images_attr[0]
                elif isinstance(result, (tuple, list)) and len(result) > 0:
                    first_item = result[0]
                    if hasattr(first_item, 'images'):
                        images_attr = getattr(first_item, 'images', None)
                        if images_attr is not None and isinstance(images_attr, (list, tuple)) and len(images_attr) > 0:
                            image = images_attr[0]
                    elif isinstance(first_item, dict) and 'images' in first_item:
                        if isinstance(first_item['images'], (list, tuple)) and len(first_item['images']) > 0:
                            image = first_item['images'][0]
                    
            except Exception as e:
                logger.error(f"Error extracting image from result: {e}")
                logger.error(f"Result type: {type(result)}")
                
            if image is None:
                raise RuntimeError(f"Could not extract image from pipeline result. Result type: {type(result)}")
            
            # Convert to base64 - EXACT main.py logic
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            # For queue system, create data URL for now
            image_url = f"data:image/png;base64,{img_str}"
            
            # Clean up LoRAs - EXACT main.py logic
            if loaded_loras:
                logger.info("🧹 Auto-unloading LoRAs after generation")
                unload_loras_from_pipeline(main.pipeline)
            
            logger.info("✅ Simulated progress system completed successfully!")
            
            # Update job as completed
            await self.queue_manager.update_job_status(
                job.id,
                JobStatus.COMPLETED,
                result_url=image_url,
                seed=actual_seed
            )
            
            logger.info(f"✅ Completed job {job.id} on local GPU - seed: {actual_seed}")
            
        except Exception as e:
            logger.error(f"Local GPU job {job.id} failed: {e}")
            
            logger.info("✅ Generation failed, no cleanup needed with simulated progress")
                
            await self.queue_manager.update_job_status(
                job.id,
                JobStatus.FAILED,
                error_message=str(e)
            )

    async def _heartbeat(self):
        """Periodically update worker heartbeat and stats"""
        while self.is_running:
            try:
                await self.queue_manager.update_worker_info(self.worker_id, {
                    "status": "running",
                    "last_seen": datetime.utcnow().isoformat(),
                    "current_job": self.current_job_id or ""
                })
            except Exception as e:
                logger.warning(f"Worker heartbeat failed: {e}")
            await asyncio.sleep(5)

# Entry point
async def run_worker():
    worker_id = os.getenv("WORKER_ID", f"local-gpu-{os.getpid()}")
    worker = LocalGPUWorker(worker_id)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await worker.stop()

if __name__ == "__main__":
    asyncio.run(run_worker()) 