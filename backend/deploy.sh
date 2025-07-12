#!/bin/bash

# Deployment script for scalable image generation architecture
# Run this to start the new queue-based system

set -e

echo "🚀 Starting Image Generation Service v2.0 (Scalable Architecture)"

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Please start Redis first:"
    echo "   Docker: docker run -d -p 6379:6379 redis:7-alpine"
    echo "   Local:  redis-server"
    exit 1
fi

echo "✅ Redis is running"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Start services in the background
echo "🔧 Starting services..."

# 1. Start API Server (stateless, handles requests)
echo "Starting API server on port 8000..."
uvicorn api_v2:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# 2. Start Worker Process (GPU processing)
echo "Starting GPU worker..."
python worker.py &
WORKER_PID=$!

# Store PIDs for cleanup
echo $API_PID > api.pid
echo $WORKER_PID > worker.pid

echo "🎉 Services started successfully!"
echo ""
echo "📡 API Server: http://localhost:8000"
echo "📊 API Docs:   http://localhost:8000/docs"
echo "💾 Queue Stats: http://localhost:8000/queue/stats"
echo ""
echo "🔄 To scale up, run additional workers:"
echo "   WORKER_ID=worker-2 python worker.py &"
echo "   WORKER_ID=worker-3 python worker.py &"
echo ""
echo "🛑 To stop services:"
echo "   kill \$(cat api.pid worker.pid)"
echo "   rm *.pid"
echo ""

# Wait for interrupt
trap 'echo "🛑 Shutting down services..."; kill $API_PID $WORKER_PID; rm -f api.pid worker.pid; exit' INT

# Keep script running
wait 