param(
    [int]$Port = 8000,
    [switch]$Detailed
)

Write-Host "FOTO RENDER SYSTEM STATUS" -ForegroundColor Cyan
Write-Host "=========================" -ForegroundColor Cyan
Write-Host ""

# API Health Check
Write-Host "API HEALTH CHECK" -ForegroundColor Magenta
Write-Host "================" -ForegroundColor Magenta

$apiUrl = "http://localhost:$Port"
$apiHealth = $false

try {
    $response = Invoke-WebRequest -Uri "$apiUrl/health" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    if ($response.StatusCode -eq 200) {
        Write-Host "API v2 is responding" -ForegroundColor Green
        $apiHealth = $true
    }
}
catch {
    Write-Host "API v2 not responding" -ForegroundColor Red
}

Write-Host ""

# Queue Statistics
Write-Host "QUEUE STATISTICS" -ForegroundColor Magenta
Write-Host "================" -ForegroundColor Magenta

$queueHealth = $false

try {
    $queueResponse = Invoke-WebRequest -Uri "$apiUrl/queue/stats" -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop
    if ($queueResponse.StatusCode -eq 200) {
        Write-Host "Queue system responding" -ForegroundColor Green
        $queueHealth = $true
    }
}
catch {
    Write-Host "Queue system not responding" -ForegroundColor Red
}

Write-Host ""

# Redis Container Status
Write-Host "REDIS CONTAINER" -ForegroundColor Magenta
Write-Host "===============" -ForegroundColor Magenta

$redisHealth = $false

try {
    $redisContainer = docker ps --filter "name=foto-render-redis" --format "{{.Names}}" 2>$null
    
    if ($redisContainer -match "foto-render-redis") {
        Write-Host "Redis container is running" -ForegroundColor Green
        $redisHealth = $true
    }
    else {
        Write-Host "Redis container not running" -ForegroundColor Red
    }
}
catch {
    Write-Host "Error checking Redis container" -ForegroundColor Yellow
}

Write-Host ""

# Python Processes
Write-Host "PYTHON PROCESSES" -ForegroundColor Magenta
Write-Host "================" -ForegroundColor Magenta

try {
    $pythonProcesses = Get-Process python* -ErrorAction SilentlyContinue
    
    if ($pythonProcesses) {
        $processCount = $pythonProcesses.Count
        Write-Host "Found $processCount Python process(es)" -ForegroundColor $(if ($processCount -eq 2) { "Green" } elseif ($processCount -eq 1) { "Yellow" } else { "Red" })
        
        if ($processCount -eq 2) {
            Write-Host "   Expected: API server + GPU worker" -ForegroundColor Gray
        }
        elseif ($processCount -eq 1) {
            Write-Host "   Warning: Only 1 process (missing API or worker)" -ForegroundColor Gray
        }
        else {
            Write-Host "   Warning: Unexpected number of processes" -ForegroundColor Gray
        }
    }
    else {
        Write-Host "No Python processes running" -ForegroundColor Red
        Write-Host "   System is down (missing API server + GPU worker)" -ForegroundColor Gray
    }
}
catch {
    Write-Host "Error checking Python processes" -ForegroundColor Yellow
}

Write-Host ""

# Overall System Summary
Write-Host "SYSTEM SUMMARY" -ForegroundColor Magenta
Write-Host "==============" -ForegroundColor Magenta

if ($apiHealth -and $queueHealth -and $redisHealth) {
    Write-Host "SYSTEM STATUS: HEALTHY" -ForegroundColor Green
}
elseif ($apiHealth -and $queueHealth) {
    Write-Host "SYSTEM STATUS: PARTIALLY HEALTHY" -ForegroundColor Yellow
}
else {
    Write-Host "SYSTEM STATUS: UNHEALTHY" -ForegroundColor Red
}

Write-Host ""
Write-Host "Quick Actions:" -ForegroundColor Yellow
Write-Host "Clean everything: .\cleanup.ps1" -ForegroundColor Gray
Write-Host "Start system: .\start_system.ps1" -ForegroundColor Gray
Write-Host "Restart system: .\cleanup.ps1; .\start_system.ps1" -ForegroundColor Gray
Write-Host "" 