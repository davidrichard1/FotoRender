@echo off
echo ========================================
echo        FOTO RENDER API LAUNCHER
echo ========================================
echo.
echo Choose your API mode:
echo   [M] Monolithic (main.py) - Steady power, single user
echo   [Q] Queue-based (api_v2.py) - Multi-user, power spikes
echo.

set /p choice="Enter your choice (M/Q): "

if /i "%choice%"=="M" (
    echo.
    echo 🏠 Starting MONOLITHIC API...
    echo   ✅ No power spikes ^(lights won't flicker^)
    echo   ✅ Simple and reliable
    echo   ❌ Single user only
    echo.
    set VIRTUAL_ENV=
    set PYTHONHOME=
    C:\Python313\python.exe main.py --port 8000
) else if /i "%choice%"=="Q" (
    echo.
    echo ⚡ Starting QUEUE-BASED API...
    echo   ✅ Multiple users
    echo   ✅ Non-blocking generation  
    echo   ❌ Potential power spikes
    echo.
    set VIRTUAL_ENV=
    set PYTHONHOME=
    C:\Python313\python.exe api_v2.py --port 8000
) else (
    echo Invalid choice. Please run again and choose M or Q.
    pause
    exit /b 1
)

pause 