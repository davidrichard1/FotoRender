# 🚀 Foto Render - Startup Guide (Queue-Based Architecture)

**IMPORTANT:** This system now uses a queue-based architecture with robust startup scripts.

## 🎯 Quick Start (Recommended)

### **🚀 One-Click Startup**

```powershell
# From foto-render/backend directory
.\start_system.ps1
```

### **🔄 One-Click Restart**

```powershell
# From foto-render/backend directory
.\restart.ps1
```

### **🧹 Clean Everything**

```powershell
# From foto-render/backend directory
.\cleanup.ps1
```

### **📊 Check Status**

```powershell
# From foto-render/backend directory
.\status.ps1
```

---

## 📋 Manual Startup (If Needed)

### **Step 1: Clean Everything First** ⚠️ **ALWAYS DO THIS**

```powershell
cd foto-render/backend
.\cleanup.ps1
```

### **Step 2: Start API Server**

```powershell
cd foto-render/backend
python api_v2.py --port 8000
```

### **Step 3: Start GPU Worker**

```powershell
# Open new terminal
cd foto-render/backend
python start_worker.py
```

### **Step 4: Start Frontend**

```powershell
# Open another terminal
cd foto-render/frontend
npm run dev
```

## 🎯 System Architecture Overview

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│   Frontend      │   │   API Server    │   │   Redis Queue   │
│   :3000         │──▶│   api_v2.py     │──▶│   :6379         │
│                 │   │   :8000         │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘
                                                     │
                                                     ▼
                                         ┌─────────────────┐
                                         │  GPU Worker(s)  │
                                         │ start_worker.py │
                                         │                 │
                                         └─────────────────┘
```

## ✅ Verification Steps

### **1. Check API Health**

```bash
curl http://localhost:8000/health
```

**Should return:** `{"status": "healthy", "version": "2.0.0", ...}`

### **2. Check Queue Stats**

```bash
curl http://localhost:8000/queue/stats
```

**Should show:** `{"pending_jobs": 0, "workers_active": 1, ...}`

### **3. Test Image Generation**

1. Go to `http://localhost:3000`
2. Enter a prompt
3. Click "🚀 Generate Image (Instant Queue)"
4. Should see real-time progress updates!

## 🛠️ Troubleshooting

| Problem                 | Solution                                          |
| ----------------------- | ------------------------------------------------- |
| `api_v2.py` won't start | Check Redis is running: `docker ps \| grep redis` |
| Worker won't start      | Check CUDA: `nvidia-smi` and Redis connection     |
| Frontend can't connect  | Verify API server is on port 8000                 |
| No progress updates     | Check WebSocket connection in browser dev tools   |

## 📊 Management Commands

### **View Worker Status**

```bash
python manage_workers.py list
```

### **Scale Workers**

```bash
python manage_workers.py scale --count 2
```

### **Stop All Workers**

```bash
python manage_workers.py stop-all
```

### **Live Monitoring**

```bash
python manage_workers.py monitor
```

## 🔄 Key Differences from Old System

| Old System               | New System                   |
| ------------------------ | ---------------------------- |
| `python main.py`         | `python api_v2.py`           |
| Blocking API (60s waits) | Non-blocking queue (instant) |
| Single user only         | Multiple concurrent users    |
| No progress tracking     | Real-time progress updates   |
| No cancellation          | Job cancellation/retry       |
| Manual scaling           | Easy worker management       |

## 🎉 What You Get

- ⚡ **Instant responses** instead of 60-second waits
- 👥 **Multiple users** can generate simultaneously
- 📊 **Real-time progress** with WebSocket updates
- 🎛️ **Job management** (cancel, retry, queue stats)
- 🔧 **Easy scaling** with worker management UI
- 🖥️ **Beautiful dashboard** at `/admin` → Workers tab

---

**Remember:** Always use `api_v2.py` instead of the old `main.py`!
