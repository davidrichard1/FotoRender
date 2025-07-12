@echo off
title Foto Render - Quick Start

echo 🚀 FOTO RENDER - QUICK START
echo ==============================
echo.

echo 🧹 Step 1: Cleaning up any existing processes...
powershell -ExecutionPolicy Bypass -File ".\cleanup.ps1"

echo.
echo ⏳ Waiting 3 seconds for cleanup to complete...
timeout /t 3 /nobreak >nul

echo.
echo 🚀 Step 2: Starting the system...
powershell -ExecutionPolicy Bypass -File ".\start_system.ps1"

echo.
echo 📋 System startup completed!
echo.
echo 📊 To check status: .\status.ps1
echo 🧹 To clean up: .\cleanup.ps1
echo 🔄 To restart: .\restart.ps1
echo.
pause 