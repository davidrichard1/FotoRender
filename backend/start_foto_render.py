#!/usr/bin/env python3
"""
Foto Render API Launcher
Choose between monolithic (main.py) or queue-based (api_v2.py) API
"""

import os
import sys
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description="Start Foto Render API")
    parser.add_argument("--mode", choices=["mono", "queue"], default="mono",
                       help="API mode: 'mono' for monolithic (main.py), 'queue' for queue-based (api_v2.py)")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    
    args = parser.parse_args()
    
    print("🎨 Foto Render API Launcher")
    print("=" * 40)
    
    if args.mode == "mono":
        print("🏠 Starting MONOLITHIC API (main.py)")
        print("   ✅ Steady power draw (no GPU spikes)")
        print("   ✅ Simple, reliable")
        print("   ❌ Single user only")
        print()
        
        # Start main.py
        cmd = [sys.executable, "main.py", "--port", str(args.port)]
        
    else:  # queue mode
        print("⚡ Starting QUEUE-BASED API (api_v2.py)")
        print("   ✅ Multiple users")
        print("   ✅ Non-blocking generation")
        print("   ❌ Potential power spikes")
        print("   ℹ️  Requires Redis running")
        print()
        
        # Start api_v2.py
        cmd = [sys.executable, "api_v2.py", "--port", str(args.port)]
    
    print(f"🚀 Starting on port {args.port}...")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    except subprocess.CalledProcessError as e:
        print(f"❌ API failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 