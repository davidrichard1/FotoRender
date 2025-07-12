#!/usr/bin/env python3
"""
GPU Worker Process - Handles actual image generation
Runs separately from the API server for scalability
"""

import asyncio
import logging
import os
import sys
import signal
import torch
from pathlib import Path
from queue_system import QueueManager, JobStatus
from generation_service import GenerationService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImageGenerationWorker:
    def __init__(self, worker_id: str = "worker-1"):
        self.worker_id = worker_id
        self.queue_manager = QueueManager()
        self.generation_service = GenerationService()
        self.is_running = False
        self.current_job_id = None
        
    async def start(self):
        """Start the worker process"""
        logger.info(f"Starting worker {self.worker_id}")
        
        # Connect to queue
        await self.queue_manager.connect()
        
        # Initialize generation service (load models, etc.)
        await self.generation_service.initialize()
        
        self.is_running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Start processing jobs
        await self._process_jobs()
        
    async def stop(self):
        """Stop the worker gracefully"""
        logger.info(f"Stopping worker {self.worker_id}")
        self.is_running = False
        
        # Cancel current job if any
        if self.current_job_id:
            await self.queue_manager.update_job_status(
                self.current_job_id, 
                JobStatus.CANCELLED,
                error_message="Worker shutdown"
            )
            
        await self.queue_manager.disconnect()
        await self.generation_service.cleanup()
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop())
        
    async def _process_jobs(self):
        """Main job processing loop"""
        logger.info(f"Worker {self.worker_id} ready for jobs")
        
        while self.is_running:
            try:
                # Get next job from queue (blocking with timeout)
                job = await self.queue_manager.get_next_job()
                
                if not job:
                    continue  # Timeout, try again
                    
                logger.info(f"Processing job {job.id}")
                self.current_job_id = job.id
                
                # Update job status to processing
                await self.queue_manager.update_job_status(
                    job.id, 
                    JobStatus.PROCESSING
                )
                
                # Process the job
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
        """Process a single generation job"""
        try:
            # Prepare generation parameters
            params = {
                "prompt": job.prompt,
                "negative_prompt": job.negative_prompt,
                "width": job.width,
                "height": job.height,
                "num_inference_steps": job.num_inference_steps,
                "guidance_scale": job.guidance_scale,
                "seed": job.seed,
                "sampler": job.sampler,
                "loras": job.loras,
                "clip_skip": job.clip_skip
            }
            
            # Progress callback
            async def progress_callback(step: int, total_steps: int):
                progress = (step / total_steps) * 100
                eta = ((total_steps - step) * 2)  # Rough estimate: 2 seconds per step
                
                await self.queue_manager.update_job_progress(
                    job.id, 
                    progress, 
                    eta
                )
                
            # Generate image
            result = await self.generation_service.generate_image(
                params, 
                progress_callback=progress_callback
            )
            
            # Update job with result
            await self.queue_manager.update_job_status(
                job.id,
                JobStatus.COMPLETED,
                result_url=result["image_url"],
                seed=result["seed"]
            )
            
            logger.info(f"Completed job {job.id}")
            
        except Exception as e:
            logger.error(f"Job {job.id} failed: {e}")
            await self.queue_manager.update_job_status(
                job.id,
                JobStatus.FAILED,
                error_message=str(e)
            )

# Entry point for running worker
async def main():
    worker_id = os.getenv("WORKER_ID", f"worker-{os.getpid()}")
    worker = ImageGenerationWorker(worker_id)
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await worker.stop()

if __name__ == "__main__":
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available! Worker requires GPU.")
        sys.exit(1)
        
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    
    # Run worker
    asyncio.run(main()) 