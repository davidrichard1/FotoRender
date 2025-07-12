"""
Worker Configuration System
Manages hybrid local/cloud worker deployments
"""

import os
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class WorkerType(Enum):
    LOCAL_GPU = "local_gpu"
    CLOUD_REPLICATE = "cloud_replicate"
    CLOUD_RUNPOD = "cloud_runpod"
    CLOUD_LAMBDA = "cloud_lambda"

@dataclass
class WorkerConfig:
    worker_id: str
    worker_type: WorkerType
    enabled: bool = True
    priority: int = 1  # Lower = higher priority
    cost_per_generation: float = 0.0
    max_concurrent_jobs: int = 1
    gpu_memory_gb: Optional[int] = None
    models_supported: List[str] = None
    
    # Local GPU specific
    cuda_device: Optional[int] = None
    
    # Cloud specific
    api_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    region: Optional[str] = None

class WorkerManager:
    def __init__(self, config_file: str = "worker_config.json"):
        self.config_file = config_file
        self.workers: Dict[str, WorkerConfig] = {}
        self.load_config()
        
    def load_config(self):
        """Load worker configuration from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    
                for worker_data in data.get('workers', []):
                    worker_config = WorkerConfig(
                        worker_id=worker_data['worker_id'],
                        worker_type=WorkerType(worker_data['worker_type']),
                        enabled=worker_data.get('enabled', True),
                        priority=worker_data.get('priority', 1),
                        cost_per_generation=worker_data.get('cost_per_generation', 0.0),
                        max_concurrent_jobs=worker_data.get('max_concurrent_jobs', 1),
                        gpu_memory_gb=worker_data.get('gpu_memory_gb'),
                        models_supported=worker_data.get('models_supported', []),
                        cuda_device=worker_data.get('cuda_device'),
                        api_key=worker_data.get('api_key'),
                        endpoint_url=worker_data.get('endpoint_url'),
                        region=worker_data.get('region')
                    )
                    self.workers[worker_config.worker_id] = worker_config
                    
            except Exception as e:
                print(f"Error loading worker config: {e}")
                self._create_default_config()
        else:
            self._create_default_config()
            
    def _create_default_config(self):
        """Create default configuration with local GPU worker"""
        print("Creating default worker configuration...")
        
        # Default local GPU worker
        local_worker = WorkerConfig(
            worker_id="local-gpu-1",
            worker_type=WorkerType.LOCAL_GPU,
            enabled=True,
            priority=1,
            cost_per_generation=0.0,  # Free local GPU
            max_concurrent_jobs=1,
            gpu_memory_gb=24,  # Adjust based on your GPU
            cuda_device=0
        )
        
        self.workers[local_worker.worker_id] = local_worker
        self.save_config()
        
    def save_config(self):
        """Save current configuration to file"""
        data = {
            'workers': []
        }
        
        for worker in self.workers.values():
            worker_data = {
                'worker_id': worker.worker_id,
                'worker_type': worker.worker_type.value,
                'enabled': worker.enabled,
                'priority': worker.priority,
                'cost_per_generation': worker.cost_per_generation,
                'max_concurrent_jobs': worker.max_concurrent_jobs,
                'gpu_memory_gb': worker.gpu_memory_gb,
                'models_supported': worker.models_supported,
                'cuda_device': worker.cuda_device,
                'api_key': worker.api_key,
                'endpoint_url': worker.endpoint_url,
                'region': worker.region
            }
            # Remove None values
            worker_data = {k: v for k, v in worker_data.items() if v is not None}
            data['workers'].append(worker_data)
            
        with open(self.config_file, 'w') as f:
            json.dump(data, f, indent=2)
            
    def add_worker(self, worker_config: WorkerConfig):
        """Add a new worker configuration"""
        self.workers[worker_config.worker_id] = worker_config
        self.save_config()
        
    def remove_worker(self, worker_id: str):
        """Remove a worker configuration"""
        if worker_id in self.workers:
            del self.workers[worker_id]
            self.save_config()
            
    def get_enabled_workers(self) -> List[WorkerConfig]:
        """Get all enabled workers sorted by priority"""
        enabled = [w for w in self.workers.values() if w.enabled]
        return sorted(enabled, key=lambda x: x.priority)
        
    def get_local_workers(self) -> List[WorkerConfig]:
        """Get all local GPU workers"""
        return [w for w in self.workers.values() 
                if w.worker_type == WorkerType.LOCAL_GPU and w.enabled]
                
    def get_cloud_workers(self) -> List[WorkerConfig]:
        """Get all cloud workers"""
        return [w for w in self.workers.values() 
                if w.worker_type != WorkerType.LOCAL_GPU and w.enabled]
                
    def estimate_cost(self, num_generations: int) -> Dict[str, float]:
        """Estimate cost breakdown by worker type"""
        costs = {}
        workers = self.get_enabled_workers()
        
        for worker in workers:
            worker_type = worker.worker_type.value
            if worker_type not in costs:
                costs[worker_type] = 0.0
            costs[worker_type] += worker.cost_per_generation * num_generations
            
        return costs
        
    def add_replicate_worker(self, worker_id: str, api_key: str, cost_per_gen: float = 0.75):
        """Helper to add Replicate worker"""
        worker = WorkerConfig(
            worker_id=worker_id,
            worker_type=WorkerType.CLOUD_REPLICATE,
            enabled=True,
            priority=2,  # Lower priority than local GPU
            cost_per_generation=cost_per_gen,
            max_concurrent_jobs=5,
            api_key=api_key
        )
        self.add_worker(worker)
        
    def add_runpod_worker(self, worker_id: str, api_key: str, endpoint_url: str, cost_per_gen: float = 0.50):
        """Helper to add RunPod worker"""
        worker = WorkerConfig(
            worker_id=worker_id,
            worker_type=WorkerType.CLOUD_RUNPOD,
            enabled=True,
            priority=3,
            cost_per_generation=cost_per_gen,
            max_concurrent_jobs=10,
            api_key=api_key,
            endpoint_url=endpoint_url
        )
        self.add_worker(worker)
        
    def get_config_summary(self) -> str:
        """Get a human-readable summary of the configuration"""
        lines = ["Worker Configuration Summary:"]
        lines.append("=" * 40)
        
        for worker in self.get_enabled_workers():
            status = "✅" if worker.enabled else "❌"
            cost = f"${worker.cost_per_generation:.2f}/gen" if worker.cost_per_generation > 0 else "Free"
            
            lines.append(f"{status} {worker.worker_id}")
            lines.append(f"   Type: {worker.worker_type.value}")
            lines.append(f"   Priority: {worker.priority}")
            lines.append(f"   Cost: {cost}")
            lines.append(f"   Concurrent Jobs: {worker.max_concurrent_jobs}")
            lines.append("")
            
        return "\n".join(lines)

# Global instance
worker_manager = WorkerManager()

# CLI for managing workers
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python worker_config.py <command>")
        print("Commands:")
        print("  show                 - Show current configuration")
        print("  add-replicate <id>   - Add Replicate worker")
        print("  add-runpod <id>      - Add RunPod worker")
        print("  disable <id>         - Disable worker")
        print("  enable <id>          - Enable worker")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "show":
        print(worker_manager.get_config_summary())
        
    elif command == "add-replicate":
        if len(sys.argv) < 3:
            print("Usage: python worker_config.py add-replicate <worker_id>")
            sys.exit(1)
            
        worker_id = sys.argv[2]
        api_key = input("Enter Replicate API key: ")
        worker_manager.add_replicate_worker(worker_id, api_key)
        print(f"Added Replicate worker: {worker_id}")
        
    elif command == "add-runpod":
        if len(sys.argv) < 3:
            print("Usage: python worker_config.py add-runpod <worker_id>")
            sys.exit(1)
            
        worker_id = sys.argv[2]
        api_key = input("Enter RunPod API key: ")
        endpoint = input("Enter RunPod endpoint URL: ")
        worker_manager.add_runpod_worker(worker_id, api_key, endpoint)
        print(f"Added RunPod worker: {worker_id}")
        
    elif command == "disable":
        if len(sys.argv) < 3:
            print("Usage: python worker_config.py disable <worker_id>")
            sys.exit(1)
            
        worker_id = sys.argv[2]
        if worker_id in worker_manager.workers:
            worker_manager.workers[worker_id].enabled = False
            worker_manager.save_config()
            print(f"Disabled worker: {worker_id}")
        else:
            print(f"Worker not found: {worker_id}")
            
    elif command == "enable":
        if len(sys.argv) < 3:
            print("Usage: python worker_config.py enable <worker_id>")
            sys.exit(1)
            
        worker_id = sys.argv[2]
        if worker_id in worker_manager.workers:
            worker_manager.workers[worker_id].enabled = True
            worker_manager.save_config()
            print(f"Enabled worker: {worker_id}")
        else:
            print(f"Worker not found: {worker_id}")
            
    else:
        print(f"Unknown command: {command}") 