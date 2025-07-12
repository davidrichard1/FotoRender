# Simple cleanup script for Foto Render
Write-Host "Cleaning up Foto Render processes..." -ForegroundColor Cyan

# Kill processes on ports 8000-8002
foreach ($port in 8000..8002) {
    $connections = netstat -ano | Select-String ":$port "
    if ($connections) {
        foreach ($line in $connections) {
            if ($line -match '\s+(\d+)$') {
                $processId = $matches[1]
                try {
                    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
                    Write-Host "Killed process on port $port (PID: $processId)" -ForegroundColor Red
                }
                catch {
                    # Ignore errors
                }
            }
        }
    }
}

# Kill all Python processes
$pythonProcs = Get-Process python* -ErrorAction SilentlyContinue
if ($pythonProcs) {
    foreach ($proc in $pythonProcs) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Write-Host "Killed Python process PID: $($proc.Id)" -ForegroundColor Red
        }
        catch {
            # Ignore errors
        }
    }
}

# Start Redis container
docker rm -f foto-render-redis 2>$null | Out-Null
docker run -d -p 6379:6379 --name foto-render-redis redis:7-alpine 2>$null | Out-Null

Write-Host "Cleanup complete!" -ForegroundColor Green 