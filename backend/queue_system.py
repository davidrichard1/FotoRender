import asyncio
import json
import uuid
from datetime import datetime
from typing import Dict, Optional, Any, Union
try:
    import redis.asyncio as redis  # type: ignore
    REDIS_ASYNCIO_AVAILABLE = True
except ImportError:
    import redis  # type: ignore
    REDIS_ASYNCIO_AVAILABLE = False
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class GenerationJob(BaseModel):
    id: str
    user_id: Optional[str] = None
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30
    guidance_scale: float = 7.0
    seed: int = -1
    sampler: str = "DPM++ 2M SDE Karras"
    loras: list = []
    clip_skip: Optional[int] = None
    status: str = JobStatus.PENDING
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    result_url: Optional[str] = None
    error_message: Optional[str] = None
    estimated_time: Optional[int] = None  # seconds

class QueueManager:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Any = None
        self.job_queue_key = "generation_jobs"
        self.job_data_key = "job_data"
        self.progress_key = "job_progress"
        self.worker_info_prefix = "worker_info"
        
    async def connect(self):
        """Connect to Redis"""
        if REDIS_ASYNCIO_AVAILABLE:
            self.redis_client = redis.from_url(self.redis_url)
        else:
            # Fallback to sync Redis (this won't work well with async, but prevents import errors)
            self.redis_client = redis.from_url(self.redis_url)
        
        if self.redis_client:
            await self.redis_client.ping()  # type: ignore
            logger.info("Connected to Redis queue")
        else:
            raise RuntimeError("Failed to create Redis client")
        
    async def disconnect(self):
        """Disconnect from Redis"""
        if self.redis_client:
            await self.redis_client.close()  # type: ignore
            
    async def add_job(self, job_data: Dict[str, Any]) -> str:
        """Add a new generation job to the queue"""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
            
        job_id = str(uuid.uuid4())
        
        job = GenerationJob(
            id=job_id,
            created_at=datetime.utcnow(),
            **job_data
        )
        
        # Convert job to dict and handle serialization
        job_dict = job.dict()
        job_dict['created_at'] = job_dict['created_at'].isoformat()
        job_dict['loras'] = json.dumps(job_dict['loras'])
        
        # Filter out None values (Redis doesn't accept None)
        filtered_job_dict = {k: v for k, v in job_dict.items() if v is not None}
        
        # Store job data
        await self.redis_client.hset(  # type: ignore
            f"{self.job_data_key}:{job_id}",
            mapping=filtered_job_dict
        )
        
        # Add to queue
        await self.redis_client.lpush(self.job_queue_key, job_id)  # type: ignore
        
        logger.info(f"Added job {job_id} to queue")
        return job_id
        
    async def get_next_job(self) -> Optional[GenerationJob]:
        """Get the next job from the queue (blocking)"""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
            
        try:
            # Block for up to 60 seconds waiting for a job
            result = await self.redis_client.brpop([self.job_queue_key], timeout=60)  # type: ignore
            if not result:
                return None
                
            job_id = result[1].decode() if isinstance(result[1], bytes) else result[1]
            job_data = await self.redis_client.hgetall(f"{self.job_data_key}:{job_id}")  # type: ignore
            
            if not job_data:
                logger.warning(f"Job {job_id} data not found")
                return None
                
            # Convert bytes to strings and parse
            job_dict = {}
            for k, v in job_data.items():
                key = k.decode() if isinstance(k, bytes) else k
                value = v.decode() if isinstance(v, bytes) else v
                job_dict[key] = value
            
            # Parse JSON fields
            if 'loras' in job_dict:
                job_dict['loras'] = json.loads(job_dict['loras'])
            if 'created_at' in job_dict:
                job_dict['created_at'] = datetime.fromisoformat(job_dict['created_at'])
                
            return GenerationJob(**job_dict)
            
        except Exception as e:
            logger.error(f"Error getting next job: {e}")
            return None
            
    async def update_job_status(self, job_id: str, status: str, **kwargs):
        """Update job status and other fields"""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
            
        update_data = {"status": status}
        
        if status == JobStatus.PROCESSING:
            update_data["started_at"] = datetime.utcnow().isoformat()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            update_data["completed_at"] = datetime.utcnow().isoformat()
            
        update_data.update(kwargs)
        
        # Filter out None values before storing in Redis
        filtered_update_data = {k: v for k, v in update_data.items() if v is not None}
        
        await self.redis_client.hmset(f"{self.job_data_key}:{job_id}", filtered_update_data)  # type: ignore
        
    async def update_job_progress(self, job_id: str, progress: float, eta: Optional[int] = None):
        """Update job progress"""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
            
        progress_data = {"progress": str(progress)}
        if eta:
            progress_data["eta"] = str(eta)
            
        # Filter out None values before storing in Redis
        filtered_progress_data = {k: v for k, v in progress_data.items() if v is not None}
        
        await self.redis_client.hmset(f"{self.progress_key}:{job_id}", filtered_progress_data)  # type: ignore
        await self.redis_client.expire(f"{self.progress_key}:{job_id}", 3600)  # type: ignore
        
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get current job status and progress"""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
            
        # Get job data
        job_data = await self.redis_client.hgetall(f"{self.job_data_key}:{job_id}")  # type: ignore
        if not job_data:
            return None
            
        # Get progress data
        progress_data = await self.redis_client.hgetall(f"{self.progress_key}:{job_id}")  # type: ignore
        
        result = {}
        for k, v in job_data.items():
            key = k.decode() if isinstance(k, bytes) else k
            value = v.decode() if isinstance(v, bytes) else v
            result[key] = value
            
        if progress_data:
            for k, v in progress_data.items():
                key = k.decode() if isinstance(k, bytes) else k
                value = v.decode() if isinstance(v, bytes) else v
                result[key] = value
            
        return result
        
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending job"""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
            
        # Try to remove from queue first
        removed = await self.redis_client.lrem(self.job_queue_key, 0, job_id)  # type: ignore
        
        # Update status regardless
        await self.update_job_status(job_id, JobStatus.CANCELLED)
        
        return bool(removed > 0)
        
    async def get_queue_stats(self) -> Dict:
        """Get queue statistics"""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
            
        queue_length = await self.redis_client.llen(self.job_queue_key)  # type: ignore
        
        return {
            "pending_jobs": int(queue_length),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    async def get_recent_jobs(self, limit: int = 10) -> list:
        """Get recent jobs (completed, failed, or cancelled)"""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
            
        try:
            # Get all job keys from Redis
            job_keys = await self.redis_client.keys(f"{self.job_data_key}:*")  # type: ignore
            jobs = []
            
            for key in job_keys:
                job_data = await self.redis_client.hgetall(key)  # type: ignore
                if job_data:
                    job_dict = {}
                    for k, v in job_data.items():
                        key_str = k.decode() if isinstance(k, bytes) else k
                        value_str = v.decode() if isinstance(v, bytes) else v
                        job_dict[key_str] = value_str
                    
                    # Parse JSON fields if they exist
                    if 'loras' in job_dict:
                        try:
                            job_dict['loras'] = json.loads(job_dict['loras'])
                        except:
                            job_dict['loras'] = []
                    
                    jobs.append(job_dict)
            
            # Sort by created_at (most recent first) and limit
            jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return jobs[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent jobs: {e}")
            return []
        
    async def cleanup_completed_jobs(self, max_age_hours: int = 24):
        """Clean up old completed jobs"""
        # This would implement cleanup logic for old job data
        # For now, just a placeholder
        pass 

    async def update_worker_info(self, worker_id: str, info: Dict[str, Any]):
        """Update or create worker info in Redis with a TTL (acts as heartbeat)."""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        key = f"{self.worker_info_prefix}:{worker_id}"
        # Convert non-str values to str for Redis
        mapping = {k: (json.dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in info.items() if v is not None}
        await self.redis_client.hmset(key, mapping)  # type: ignore
        # Set TTL so stale workers disappear automatically (30s)
        await self.redis_client.expire(key, 30)  # type: ignore

    async def get_workers(self) -> Dict[str, Dict[str, str]]:
        """Return dict of all workers info from Redis."""
        if not self.redis_client:
            raise RuntimeError("Redis client not connected")
        workers: Dict[str, Dict[str, str]] = {}
        async for key in self.redis_client.scan_iter(f"{self.worker_info_prefix}:*"):  # type: ignore
            wid = key.decode().split(":", 1)[1] if isinstance(key, bytes) else key.split(":", 1)[1]
            data = await self.redis_client.hgetall(key)  # type: ignore
            workers[wid] = { (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k,v in data.items() }
        return workers 