#!/bin/bash

# Local GPU Deployment Script
# Uses your existing models and GPU with the new queue architecture

set -e

echo "🏠 Starting Image Generation Service v2.0 (Local GPU Mode)"

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo "❌ Redis is not running. Starting Redis..."
    echo "📦 Using Docker to start Redis:"
    docker run -d -p 6379:6379 --name foto-render-redis redis:7-alpine
    sleep 2
fi

echo "✅ Redis is running"

# Check GPU availability
if ! python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('GPU Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"; then
    echo "❌ Python/PyTorch not properly configured"
    exit 1
fi

# Check if models directory exists
if [ ! -d "models" ]; then
    echo "⚠️  Models directory not found. Make sure your models are in ./models/"
    echo "   Current directory contents:"
    ls -la
fi

# Install dependencies
echo "📦 Installing queue dependencies..."
pip install redis>=5.0.0 websockets>=12.0

# Start services
echo "🔧 Starting services..."

# 1. Start stateless API server (no GPU needed)
echo "🌐 Starting API server on port 8000..."
uvicorn api_v2:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# 2. Start local GPU worker (uses your existing models/functions)
echo "🖥️  Starting local GPU worker..."
python local_gpu_worker.py &
WORKER_PID=$!

# Store PIDs for cleanup
echo $API_PID > api.pid
echo $WORKER_PID > worker.pid

echo ""
echo "🎉 Local GPU services started successfully!"
echo ""
echo "📊 Dashboard:"
echo "   🌐 API Server:    http://localhost:8000"
echo "   📖 API Docs:      http://localhost:8000/docs"
echo "   📈 Queue Stats:   http://localhost:8000/queue/stats"
echo "   🔍 Health Check:  http://localhost:8000/health"
echo ""
echo "🧪 Test your setup:"
echo "   curl -X POST http://localhost:8000/generate \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"prompt\": \"a beautiful sunset\", \"width\": 1024, \"height\": 1024}'"
echo ""
echo "⚡ Performance Tips:"
echo "   - Your local GPU will handle all generations"
echo "   - Models load once and stay in GPU memory"
echo "   - Multiple requests queue automatically"
echo "   - Real-time progress via WebSocket"
echo ""
echo "🔄 To add more local workers (if you have multiple GPUs):"
echo "   CUDA_VISIBLE_DEVICES=1 WORKER_ID=local-gpu-2 python local_gpu_worker.py &"
echo ""
echo "☁️  To add cloud workers later:"
echo "   REDIS_URL=redis://localhost:6379 python worker.py &"
echo ""
echo "🛑 To stop services:"
echo "   kill \$(cat api.pid worker.pid)"
echo "   rm *.pid"
echo "   docker stop foto-render-redis"
echo ""

# Function to show real-time queue stats
show_stats() {
    while true; do
        clear
        echo "📊 Real-time Queue Statistics:"
        echo "=============================="
        curl -s http://localhost:8000/queue/stats | python -m json.tool 2>/dev/null || echo "API not ready yet..."
        echo ""
        echo "Press Ctrl+C to stop monitoring"
        sleep 5
    done
}

echo "📊 Monitor queue in real-time? (y/N)"
read -t 5 -n 1 response
if [[ $response =~ ^[Yy]$ ]]; then
    show_stats
fi

# Wait for interrupt
trap 'echo ""; echo "🛑 Shutting down services..."; kill $API_PID $WORKER_PID 2>/dev/null; rm -f api.pid worker.pid; docker stop foto-render-redis 2>/dev/null; exit' INT

echo "✨ Services running. Press Ctrl+C to stop."
wait 