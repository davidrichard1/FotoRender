#!/usr/bin/env powershell
<#
.SYNOPSIS
    Foto Render - Complete System Restart
    
.DESCRIPTION
    One-click script to completely restart the Foto Render system:
    1. Runs cleanup to stop everything
    2. Starts the entire system fresh
    
.PARAMETER Port
    Port for API server (default: 8000)

.PARAMETER WorkerGpuId
    GPU ID for worker (default: 0)

.PARAMETER Verbose
    Show detailed output

.EXAMPLE
    .\restart.ps1
    .\restart.ps1 -Port 8001 -WorkerGpuId 1 -Verbose
#>

param(
    [int]$Port = 8000,
    [int]$WorkerGpuId = 0,
    [switch]$Verbose = $false
)

Write-Host "FOTO RENDER - COMPLETE SYSTEM RESTART" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Cyan
Write-Host "API Port: $Port | Worker GPU: $WorkerGpuId" -ForegroundColor Gray
Write-Host ""

try {
    # Step 1: Cleanup
    Write-Host "STEP 1: CLEANING UP..." -ForegroundColor Yellow
    if (Test-Path ".\cleanup.ps1") {
        & .\cleanup.ps1 -Verbose:$Verbose
    }
    else {
        Write-Host "ERROR: cleanup.ps1 not found!" -ForegroundColor Red
        exit 1
    }
    
    Write-Host ""
    Write-Host "Waiting 3 seconds for cleanup to settle..." -ForegroundColor Gray
    Start-Sleep -Seconds 3
    Write-Host ""
    
    # Step 2: Start System
    Write-Host "STEP 2: STARTING SYSTEM..." -ForegroundColor Yellow
    if (Test-Path ".\start_system.ps1") {
        & .\start_system.ps1 -Port $Port -WorkerGpuId $WorkerGpuId -Verbose:$Verbose -SkipCleanup
    }
    else {
        Write-Host "ERROR: start_system.ps1 not found!" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host ""
    Write-Host "RESTART FAILED" -ForegroundColor Red
    Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} 