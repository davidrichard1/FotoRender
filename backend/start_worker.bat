@echo off
echo 🚀 Starting Foto Render GPU Worker...
echo.

REM Check if Redis is running
echo Checking Redis connection...
python -c "import redis; r=redis.Redis(); r.ping(); print('✅ Redis connected')" 2>nul
if %errorlevel% neq 0 (
    echo ❌ Redis not connected. Please start Redis first:
    echo    docker run -d -p 6379:6379 --name foto-render-redis redis:7-alpine
    pause
    exit /b 1
)

REM Set worker environment
set WORKER_ID=local-gpu-0
set CUDA_VISIBLE_DEVICES=0

echo Starting worker: %WORKER_ID%
echo GPU: %CUDA_VISIBLE_DEVICES%
echo.

REM Start the worker
python local_gpu_worker.py

echo.
echo Worker stopped. Press any key to exit...
pause >nul 