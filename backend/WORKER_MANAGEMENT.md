# 🔧 Worker Management Guide

Complete guide to managing GPU workers for scalable image generation in Foto Render.

## 🚀 Quick Start Commands

### **Start a Worker (Simple)**

```bash
# Windows
start_worker.bat

# Python (cross-platform)
python start_worker.py

# Advanced management
python manage_workers.py start
```

### **View Worker Status**

```bash
python manage_workers.py list
```

### **Stop All Workers**

```bash
python manage_workers.py stop-all
```

## 📊 Worker Management CLI

### **Available Commands**

| Command    | Description          | Example                                                 |
| ---------- | -------------------- | ------------------------------------------------------- |
| `start`    | Start a new worker   | `python manage_workers.py start --gpu-id 0`             |
| `stop`     | Stop specific worker | `python manage_workers.py stop --worker-id local-gpu-0` |
| `list`     | Show all workers     | `python manage_workers.py list`                         |
| `stop-all` | Stop all workers     | `python manage_workers.py stop-all`                     |
| `scale`    | Scale to N workers   | `python manage_workers.py scale --count 3`              |
| `monitor`  | Live monitoring      | `python manage_workers.py monitor`                      |

### **Command Options**

- `--worker-id`: Specific worker identifier
- `--gpu-id`: GPU device ID (0, 1, 2, etc.)
- `--count`: Number of workers for scaling
- `--watch`: Auto-refresh mode for list command

### **Examples**

```bash
# Start worker on GPU 1
python manage_workers.py start --gpu-id 1

# Watch worker status with auto-refresh
python manage_workers.py list --watch

# Scale to 2 workers
python manage_workers.py scale --count 2

# Live monitoring dashboard
python manage_workers.py monitor
```

## 🖥️ Web Dashboard

Access the worker management dashboard through the admin interface:

1. Go to `http://localhost:3000/admin`
2. Click the **"Workers"** tab
3. View real-time worker status and queue statistics
4. Start/stop workers with one click

### **Dashboard Features**

- **Real-time Status**: Live worker monitoring with 5-second refresh
- **Queue Statistics**: Pending, processing, completed, and failed jobs
- **Performance Metrics**: Success rate, average wait time, jobs per worker
- **One-Click Controls**: Start/stop workers directly from the UI
- **Mixed Worker Support**: Shows both local and cloud workers

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   API Server    │    │  Redis Queue    │
│                 │───▶│                 │───▶│                 │
│  Job Submission │    │  Job Queueing   │    │  Job Storage    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
                        ┌─────────────────┐    ┌─────────────────┐
                        │  Worker Manager │    │  GPU Worker(s)  │
                        │                 │───▶│                 │
                        │ Process Control │    │ Image Generation│
                        └─────────────────┘    └─────────────────┘
```

## 🔧 Configuration

### **Worker Types**

1. **Local Workers** (`local`):

   - Run on your local GPU
   - Direct model access
   - Low latency
   - Full control

2. **Cloud Workers** (`cloud`):
   - Run on cloud instances
   - Elastic scaling
   - API-based communication
   - Cost-effective for bursts

### **Environment Variables**

| Variable               | Description              | Default                  |
| ---------------------- | ------------------------ | ------------------------ |
| `WORKER_ID`            | Unique worker identifier | `local-gpu-{timestamp}`  |
| `CUDA_VISIBLE_DEVICES` | GPU device(s) to use     | `0`                      |
| `REDIS_URL`            | Redis connection string  | `redis://localhost:6379` |
| `MAX_JOBS_PER_WORKER`  | Job limit per worker     | `unlimited`              |

## 📈 Scaling Strategies

### **Local Scaling**

- **Single GPU**: 1 worker per GPU for optimal performance
- **Multi-GPU**: 1 worker per GPU, or fewer for memory-intensive models
- **CPU Fallback**: Workers can use CPU if CUDA unavailable

### **Hybrid Scaling**

- **Local + Cloud**: Use local for low latency, cloud for overflow
- **Load Balancing**: Queue automatically distributes jobs
- **Cost Optimization**: Start cloud workers only when needed

### **Performance Tips**

1. **Model Loading**: Workers cache models to avoid reload overhead
2. **Memory Management**: Monitor GPU memory usage per worker
3. **Queue Tuning**: Adjust job batching for optimal throughput
4. **Monitoring**: Use the dashboard to identify bottlenecks

## 🛠️ Troubleshooting

### **Common Issues**

| Problem            | Solution                                      |
| ------------------ | --------------------------------------------- |
| Worker won't start | Check Redis connection and CUDA setup         |
| High memory usage  | Reduce concurrent workers or model batch size |
| Slow generation    | Check GPU utilization and model optimization  |
| Workers disappear  | Check process logs and system resources       |

### **Debug Commands**

```bash
# Check Redis connection
python -c "import redis; print(redis.Redis().ping())"

# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# View worker logs
python manage_workers.py list --verbose

# Test generation locally
python -m local_gpu_worker --test
```

### **Log Locations**

- **Worker Logs**: `worker_logs/{worker_id}.log`
- **Queue Logs**: `queue_system.log`
- **API Logs**: `api_v2.log`

## 🔄 Integration with Existing Systems

### **Preserves Your Setup**

- ✅ All existing models, LoRAs, and optimizations
- ✅ SageAttention, Triton, and RTX 5090 optimizations
- ✅ R2 storage and existing file handling
- ✅ Current prompt templates and generation settings

### **Adds New Capabilities**

- 🚀 Instant job submission (non-blocking API)
- 📊 Real-time progress tracking
- ⚡ Concurrent user support
- 🎛️ Job cancellation and retry
- 📈 Performance monitoring

## 🎯 Next Steps

1. **Test the System**:

   ```bash
   # Start Redis
   docker run -d -p 6379:6379 --name foto-render-redis redis:7-alpine

   # Start a worker
   python start_worker.py

   # Test the API
   python test_queue_system.py
   ```

2. **Monitor Performance**:

   - Use the web dashboard at `/admin`
   - Monitor GPU usage with `nvidia-smi`
   - Check queue stats regularly

3. **Scale Up**:

   - Add more local workers as needed
   - Consider cloud workers for peak loads
   - Monitor costs and performance

4. **Production Deployment**:
   - Set up monitoring and alerting
   - Configure automatic worker scaling
   - Implement backup and recovery

---

**Need Help?** Check the logs, use the web dashboard, or refer to this guide for troubleshooting common issues.
