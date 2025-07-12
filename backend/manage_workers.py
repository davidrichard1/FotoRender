#!/usr/bin/env python3
"""
Worker Management CLI
Easy commands to manage local and cloud GPU workers
"""

import asyncio
import argparse
import json
import subprocess
import sys
import os
import signal
import time
from pathlib import Path
from typing import List, Dict, Optional
import psutil
import redis
from queue_system import QueueManager

class WorkerManager:
    def __init__(self):
        self.queue_manager = QueueManager()
        self.worker_processes: Dict[str, subprocess.Popen] = {}
        self.config_file = Path("worker_config.json")
        
    async def start_local_worker(self, worker_id: Optional[str] = None, gpu_id: int = 0) -> str:
        """Start a local GPU worker"""
        if not worker_id:
            worker_id = f"local-gpu-{gpu_id}-{int(time.time())}"
            
        print(f"🚀 Starting local GPU worker: {worker_id}")
        
        # Set environment variables
        env = os.environ.copy()
        env["WORKER_ID"] = worker_id
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        # Start worker process
        process = subprocess.Popen(
            [sys.executable, "local_gpu_worker.py"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        self.worker_processes[worker_id] = process
        print(f"✅ Started worker {worker_id} (PID: {process.pid})")
        
        # Save to config
        await self._save_worker_config(worker_id, "local", {"gpu_id": gpu_id, "pid": process.pid})
        
        return worker_id
    
    async def stop_worker(self, worker_id: str) -> bool:
        """Stop a specific worker"""
        print(f"🛑 Stopping worker: {worker_id}")
        
        # Check if it's a local process we're managing
        if worker_id in self.worker_processes:
            process = self.worker_processes[worker_id]
            process.terminate()
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                
            del self.worker_processes[worker_id]
            print(f"✅ Stopped local worker {worker_id}")
            
        else:
            # Try to find by PID from config
            config = await self._load_worker_config()
            if worker_id in config:
                worker_info = config[worker_id]
                if "pid" in worker_info:
                    try:
                        os.kill(worker_info["pid"], signal.SIGTERM)
                        print(f"✅ Sent stop signal to worker {worker_id}")
                    except ProcessLookupError:
                        print(f"⚠️ Worker {worker_id} already stopped")
                        
        # Remove from config
        await self._remove_worker_config(worker_id)
        return True
    
    async def list_workers(self) -> Dict:
        """List all workers and their status"""
        print("📊 Worker Status Dashboard")
        print("=" * 50)
        
        # Get queue stats
        await self.queue_manager.connect()
        queue_stats = await self.queue_manager.get_queue_stats()
        
        # Get worker info from config
        config = await self._load_worker_config()
        
        workers = {}
        
        for worker_id, worker_info in config.items():
            status = "unknown"
            
            if worker_info["type"] == "local":
                # Check if process is still running
                if "pid" in worker_info:
                    try:
                        process = psutil.Process(worker_info["pid"])
                        if process.is_running():
                            status = "running"
                        else:
                            status = "stopped"
                    except psutil.NoSuchProcess:
                        status = "stopped"
                        
            elif worker_info["type"] == "cloud":
                # For cloud workers, we'd check their API status
                status = "cloud_unknown"  # Implement cloud status checking
                
            workers[worker_id] = {
                **worker_info,
                "status": status
            }
            
            # Print status
            status_emoji = "🟢" if status == "running" else "🔴" if status == "stopped" else "🟡"
            print(f"{status_emoji} {worker_id}")
            print(f"   Type: {worker_info['type']}")
            print(f"   Status: {status}")
            if "gpu_id" in worker_info:
                print(f"   GPU: {worker_info['gpu_id']}")
            if "pid" in worker_info:
                print(f"   PID: {worker_info['pid']}")
            print()
            
        # Print queue stats
        print("🔄 Queue Statistics:")
        print(f"   Pending: {queue_stats.get('pending_jobs', 0)}")
        print(f"   Processing: {queue_stats.get('processing_jobs', 0)}")
        print(f"   Active Workers: {queue_stats.get('workers_active', 0)}")
        print(f"   Completed: {queue_stats.get('completed_jobs', 0)}")
        print(f"   Failed: {queue_stats.get('failed_jobs', 0)}")
        
        await self.queue_manager.disconnect()
        return workers
    
    async def stop_all_workers(self):
        """Stop all workers"""
        print("🛑 Stopping all workers...")
        
        config = await self._load_worker_config()
        for worker_id in list(config.keys()):
            await self.stop_worker(worker_id)
            
        print("✅ All workers stopped")
    
    async def scale_workers(self, target_count: int):
        """Scale local workers to target count"""
        current_workers = await self._count_running_local_workers()
        
        if target_count > current_workers:
            # Start more workers
            for i in range(target_count - current_workers):
                await self.start_local_worker(gpu_id=0)  # Adjust GPU assignment logic
                
        elif target_count < current_workers:
            # Stop excess workers
            config = await self._load_worker_config()
            local_workers = [w for w, info in config.items() if info["type"] == "local"]
            
            for i in range(current_workers - target_count):
                if local_workers:
                    worker_to_stop = local_workers.pop()
                    await self.stop_worker(worker_to_stop)
                    
        print(f"✅ Scaled to {target_count} workers")
    
    async def _count_running_local_workers(self) -> int:
        """Count currently running local workers"""
        config = await self._load_worker_config()
        count = 0
        
        for worker_id, worker_info in config.items():
            if worker_info["type"] == "local" and "pid" in worker_info:
                try:
                    process = psutil.Process(worker_info["pid"])
                    if process.is_running():
                        count += 1
                except psutil.NoSuchProcess:
                    pass
                    
        return count
    
    async def _load_worker_config(self) -> Dict:
        """Load worker configuration"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    async def _save_worker_config(self, worker_id: str, worker_type: str, info: Dict):
        """Save worker configuration"""
        config = await self._load_worker_config()
        config[worker_id] = {
            "type": worker_type,
            "created_at": time.time(),
            **info
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    async def _remove_worker_config(self, worker_id: str):
        """Remove worker from configuration"""
        config = await self._load_worker_config()
        if worker_id in config:
            del config[worker_id]
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)

async def main():
    parser = argparse.ArgumentParser(description="Worker Management CLI")
    parser.add_argument("command", choices=[
        "start", "stop", "list", "stop-all", "scale", "monitor"
    ])
    parser.add_argument("--worker-id", help="Specific worker ID")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID for local workers")
    parser.add_argument("--count", type=int, help="Target worker count for scaling")
    parser.add_argument("--watch", action="store_true", help="Watch mode (auto-refresh)")
    
    args = parser.parse_args()
    
    manager = WorkerManager()
    
    if args.command == "start":
        worker_id = await manager.start_local_worker(args.worker_id, args.gpu_id)
        print(f"🎉 Worker started: {worker_id}")
        
    elif args.command == "stop":
        if not args.worker_id:
            print("❌ --worker-id required for stop command")
            sys.exit(1)
        await manager.stop_worker(args.worker_id)
        
    elif args.command == "list":
        if args.watch:
            try:
                while True:
                    os.system('clear' if os.name == 'posix' else 'cls')
                    await manager.list_workers()
                    print("\n🔄 Refreshing in 5 seconds... (Ctrl+C to stop)")
                    time.sleep(5)
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
        else:
            await manager.list_workers()
            
    elif args.command == "stop-all":
        await manager.stop_all_workers()
        
    elif args.command == "scale":
        if not args.count:
            print("❌ --count required for scale command")
            sys.exit(1)
        await manager.scale_workers(args.count)
        
    elif args.command == "monitor":
        # Continuous monitoring
        try:
            while True:
                os.system('clear' if os.name == 'posix' else 'cls')
                await manager.list_workers()
                print("\n🔄 Auto-monitoring... (Ctrl+C to stop)")
                time.sleep(3)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped!")

if __name__ == "__main__":
    asyncio.run(main()) 