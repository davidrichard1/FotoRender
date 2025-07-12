# 🛠️ Foto Render Backend - Script Documentation

This directory contains robust PowerShell scripts to manage the Foto Render system without conflicts.

## 🚀 Quick Reference

| Script             | Purpose                         | Usage                               |
| ------------------ | ------------------------------- | ----------------------------------- |
| `quick_start.bat`  | **One-click startup** (Windows) | Double-click or `.\quick_start.bat` |
| `start_system.ps1` | Complete system startup         | `.\start_system.ps1`                |
| `cleanup.ps1`      | Kill all processes & clean up   | `.\cleanup.ps1`                     |
| `restart.ps1`      | Clean restart of everything     | `.\restart.ps1`                     |
| `status.ps1`       | Check system health             | `.\status.ps1`                      |

## 📋 Detailed Script Documentation

### 🚀 `start_system.ps1` - Main Startup Script

**Purpose:** Safely starts the entire Foto Render system with conflict prevention.

**Features:**

- ✅ Automatic cleanup (unless skipped)
- ✅ Redis container management
- ✅ API v2 server startup
- ✅ GPU worker initialization
- ✅ Health verification
- ✅ Background job management

**Usage:**

```powershell
# Basic startup
.\start_system.ps1

# Custom port and GPU
.\start_system.ps1 -Port 8001 -WorkerGpuId 1

# Skip cleanup (not recommended)
.\start_system.ps1 -SkipCleanup

# Verbose output
.\start_system.ps1 -Verbose
```

**Parameters:**

- `-Port` - API server port (default: 8000)
- `-WorkerGpuId` - GPU ID for worker (default: 0)
- `-SkipCleanup` - Skip cleanup phase (not recommended)
- `-Verbose` - Show detailed output

### 🧹 `cleanup.ps1` - System Cleanup Script

**Purpose:** Completely cleans up all processes, ports, and temporary files.

**What it does:**

- 🔫 Kills all processes on ports 8000-8002
- 🔫 Kills all Python processes (nuclear option)
- 🐳 Restarts Redis container if needed
- 🗑️ Clears temporary files (**pycache**, \*.pyc, etc.)
- ✅ Verifies all ports are clear

**Usage:**

```powershell
# Basic cleanup
.\cleanup.ps1

# Verbose output with details
.\cleanup.ps1 -Verbose
```

**When to use:**

- Before starting the system
- When you have port conflicts
- When processes are stuck
- After system crashes

### 📊 `status.ps1` - System Health Checker

**Purpose:** Comprehensive system status and health check.

**What it checks:**

- 🔌 Port usage (8000-8002)
- 🐍 Python processes
- 🐳 Redis container status
- 🏥 API health endpoints
- 📊 Queue statistics
- 🔄 Background PowerShell jobs
- 📋 Overall system summary

**Usage:**

```powershell
# Quick status check
.\status.ps1

# Detailed status with full output
.\status.ps1 -Detailed

# Check different port
.\status.ps1 -Port 8001 -Detailed
```

### 🔄 `restart.ps1` - Complete System Restart

**Purpose:** One-click complete restart of the entire system.

**What it does:**

1. Runs `cleanup.ps1` to stop everything
2. Waits for cleanup to settle
3. Runs `start_system.ps1` to start fresh

**Usage:**

```powershell
# Basic restart
.\restart.ps1

# Custom configuration
.\restart.ps1 -Port 8001 -WorkerGpuId 1 -Verbose
```

### 📁 `quick_start.bat` - Windows Batch Launcher

**Purpose:** Simple double-click startup for Windows users.

**What it does:**

- Runs cleanup.ps1
- Runs start_system.ps1
- Shows completion message
- Pauses for user to see results

**Usage:**

- Double-click the file
- Or run: `.\quick_start.bat`

## 🔧 Advanced Usage

### Starting Multiple Workers

```powershell
# Start API on port 8000, worker on GPU 0
.\start_system.ps1 -Port 8000 -WorkerGpuId 0

# Start additional worker on GPU 1 (different terminal)
cd foto-render/backend
$env:CUDA_VISIBLE_DEVICES = 1
$env:WORKER_ID = "local-gpu-1"
python local_gpu_worker.py
```

### Development Mode

```powershell
# Start API with auto-reload for development
cd foto-render/backend
python api_v2.py --port 8000 --reload
```

### Monitoring Jobs

```powershell
# List all PowerShell background jobs
Get-Job

# Get output from a specific job
Receive-Job -Id 1

# Stop a background job
Stop-Job -Id 1
Remove-Job -Id 1
```

### Troubleshooting

```powershell
# Check what's using port 8000
netstat -ano | findstr :8000

# Kill specific process
taskkill /PID 12345 /F

# Check Redis
docker ps | findstr redis
docker exec foto-render-redis redis-cli ping

# Check API manually
curl http://localhost:8000/health
```

## 🚨 Emergency Recovery

If everything breaks:

1. **Nuclear option:**

   ```powershell
   .\cleanup.ps1 -Verbose
   # Wait 10 seconds
   .\start_system.ps1 -Verbose
   ```

2. **Manual cleanup:**

   ```powershell
   # Kill all Python
   Get-Process python* | Stop-Process -Force

   # Remove Redis container
   docker rm -f foto-render-redis

   # Restart everything
   .\start_system.ps1
   ```

3. **Reboot computer** (last resort)

## 📝 Script Dependencies

All scripts require:

- PowerShell 5.1+ (included with Windows 10/11)
- Docker Desktop (for Redis)
- Python 3.8+ with required packages
- Network access to localhost

## 🎯 Best Practices

1. **Always run cleanup first** if you're having issues
2. **Use status.ps1** to check system health regularly
3. **Use restart.ps1** for clean restarts
4. **Check background jobs** with `Get-Job` if things seem stuck
5. **Use verbose flags** when troubleshooting

## ⚠️ Common Issues

| Problem                 | Solution                                    |
| ----------------------- | ------------------------------------------- |
| "Port already in use"   | Run `.\cleanup.ps1` first                   |
| "Redis not responding"  | Check Docker Desktop is running             |
| "API not starting"      | Check Redis is running, run with `-Verbose` |
| "Worker not connecting" | Check API is healthy, check GPU drivers     |
| "Scripts won't run"     | Run `Set-ExecutionPolicy RemoteSigned`      |

---

**Never have startup conflicts again!** 🎉
