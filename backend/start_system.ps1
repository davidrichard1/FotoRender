# Simple start system script for Foto Render
param(
    [int]$Port = 8000,
    [int]$WorkerGpuId = 0,
    [switch]$SkipCleanup,
    [switch]$Verbose
)

Write-Host "FOTO RENDER - STARTING SYSTEM" -ForegroundColor Cyan
Write-Host "API Port: $Port | Worker GPU: $WorkerGpuId" -ForegroundColor Gray
Write-Host ""

# Step 1: Cleanup (unless skipped)
if (-not $SkipCleanup) {
    Write-Host "STEP 1: CLEANING UP..." -ForegroundColor Magenta
    .\cleanup.ps1
    Write-Host "Waiting 3 seconds for cleanup to settle..." -ForegroundColor Yellow
    Start-Sleep -Seconds 3
    Write-Host ""
}

# Step 2: Start Redis
Write-Host "STEP 2: STARTING REDIS..." -ForegroundColor Magenta
$redisCheck = docker ps --filter "name=foto-render-redis" --format "{{.Names}}" 2>$null
if (-not ($redisCheck -match "foto-render-redis")) {
    docker rm -f foto-render-redis 2>$null | Out-Null
    docker run -d -p 6379:6379 --name foto-render-redis redis:7-alpine 2>$null | Out-Null
    Write-Host "Redis container started" -ForegroundColor Green
} else {
    Write-Host "Redis already running" -ForegroundColor Green
}
Write-Host ""

# Step 3: Start API
Write-Host "STEP 3: STARTING API SERVER..." -ForegroundColor Magenta
Write-Host "Starting API v2 on port $Port..." -ForegroundColor Yellow

# Start API using simple batch file
Start-Process -FilePath "cmd" -ArgumentList "/c", "start_api.bat" -WindowStyle Hidden

# Wait a moment for startup
Start-Sleep -Seconds 3

# Check if API is responding
$maxAttempts = 10
$attempt = 0
$apiReady = $false

while ($attempt -lt $maxAttempts -and -not $apiReady) {
    $attempt++
    
    try {
        $apiCheck = Invoke-WebRequest -Uri "http://localhost:$Port/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($apiCheck -and $apiCheck.StatusCode -eq 200) {
            $apiReady = $true
            Write-Host "API v2 server is responding!" -ForegroundColor Green
        } else {
            Write-Host "Attempt $attempt/$maxAttempts..." -ForegroundColor Yellow
            Start-Sleep -Seconds 1
        }
    }
    catch {
        Write-Host "Attempt $attempt/$maxAttempts..." -ForegroundColor Yellow
        Start-Sleep -Seconds 1
    }
}

if (-not $apiReady) {
    Write-Host "API server did not respond after $maxAttempts attempts" -ForegroundColor Red
    Write-Host "The API process may have started but isn't responding to health checks" -ForegroundColor Yellow
    Write-Host ""
    exit 1
} else {
    Write-Host ""
    
    # Step 4: Start Worker
    Write-Host "STEP 4: STARTING GPU WORKER..." -ForegroundColor Magenta
    Write-Host "Starting local GPU worker on GPU $WorkerGpuId..." -ForegroundColor Yellow
    
    # Start worker using simple batch file
    Start-Process -FilePath "cmd" -ArgumentList "/c", "start_worker.bat" -WindowStyle Hidden
    
    Write-Host "GPU worker started in background" -ForegroundColor Green
    Write-Host ""
    
    # Final status
    Write-Host "SYSTEM STARTUP COMPLETE!" -ForegroundColor Green
    Write-Host ""
    Write-Host "API Server: http://localhost:$Port" -ForegroundColor Cyan
    Write-Host "Health Check: http://localhost:$Port/health" -ForegroundColor Cyan
    Write-Host "Queue Stats: http://localhost:$Port/queue/stats" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Management Commands:" -ForegroundColor Yellow
    Write-Host "Check status: .\status.ps1" -ForegroundColor Gray
    Write-Host "Stop everything: .\cleanup.ps1" -ForegroundColor Gray
    Write-Host "Restart: .\restart.ps1" -ForegroundColor Gray
}
Write-Host ""
