#!/usr/bin/env python3
"""
Quick Worker Starter
Simple script to start a GPU worker quickly
"""

import sys
import subprocess
import os

def start_worker(gpu_id=0):
    """Start a local GPU worker on specified GPU"""
    try:
        print(f"🚀 Starting local GPU worker on GPU {gpu_id}...")
    except UnicodeEncodeError:
        print(f"[STARTUP] Starting local GPU worker on GPU {gpu_id}...")
    
    # Set environment variables
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["WORKER_ID"] = f"local-gpu-{gpu_id}"
    
    # Start the worker
    try:
        subprocess.run([sys.executable, "local_gpu_worker.py"], env=env)
    except KeyboardInterrupt:
        try:
            print("\n👋 Worker stopped by user")
        except UnicodeEncodeError:
            print("\n[INFO] Worker stopped by user")
    except Exception as e:
        try:
            print(f"❌ Error starting worker: {e}")
        except UnicodeEncodeError:
            print(f"[ERROR] Error starting worker: {e}")

if __name__ == "__main__":
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    start_worker(gpu_id) 