# 📦 Package.json Scripts Guide - Foto Render v2.0

## 🚀 Quick Start (Recommended)

```bash
# From root directory - One command does it all!
npm start                    # Robust startup with cleanup
npm run dev                  # Same as above (alias)
npm run start:quick          # One-click startup (.bat file)
```

## 🎯 Primary Commands (Root Directory)

### Essential Operations

```bash
npm start                    # Start entire system (uses PowerShell scripts)
npm run dev                  # Same as start (recommended for development)
npm stop                     # Clean shutdown of all processes
npm run restart              # Full clean restart
npm run status               # Check system health
npm run cleanup              # Stop all processes
npm run emergency:fix        # Nuclear restart (if something's broken)
```

### System Health

```bash
npm run health               # Check API health
npm run queue                # Check Redis queue stats
npm run redis:ping           # Test Redis connection
npm run status:detailed      # Detailed system report
```

## 🏗️ Backend Scripts (backend/ directory)

### Core Operations

```bash
cd backend
npm start                    # Robust startup with PowerShell
npm stop                     # Clean shutdown
npm restart                  # Full restart with cleanup
npm run status               # System status check
npm run status:detailed      # Detailed status report
```

### Development

```bash
npm run api:start            # Start API server only
npm run api:dev              # Start API with hot reload
npm run worker:start         # Start GPU worker only
npm run worker:gpu           # Start local GPU worker directly
```

### Emergency & Cleanup

```bash
npm run cleanup:nuclear      # Nuclear cleanup (kills everything)
npm run emergency:fix        # Nuclear cleanup + restart
```

### Database Operations

```bash
npm run db:create            # Create database tables
npm run db:seed              # Seed database with sample data
npm run db:test              # Test database connection
```

## 🎨 Frontend Scripts (frontend/ directory)

### Development

```bash
cd frontend
npm run dev                  # Start Next.js dev server (port 3000)
npm run dev:network          # Start with network access (0.0.0.0)
npm run dev:turbo            # Start with Turbo mode
npm run dev:verbose          # Start with verbose logging
```

### Production

```bash
npm run build                # Build for production
npm run start                # Start production server
npm run start:production     # Start with network access
```

### Quality & Maintenance

```bash
npm run lint                 # Check code style
npm run lint:fix             # Fix linting issues
npm run type-check           # TypeScript type checking
npm run clean                # Clean build artifacts
```

### Backend Integration

```bash
npm run backend:start        # Start backend from frontend dir
npm run backend:stop         # Stop backend from frontend dir
npm run backend:status       # Check backend status
npm run system:health        # Quick backend health check
npm run system:check         # Check if backend is running
```

## 🔧 Installation & Setup

### First Time Setup

```bash
# From root directory
npm run system:setup         # Install all dependencies
npm run install:all          # Same as above
```

### Individual Components

```bash
npm run install:backend      # Install Python dependencies
npm run install:frontend     # Install Node.js dependencies
```

## 📊 System Monitoring

### Real-time Status

```bash
npm run status               # Overall system health
npm run health               # API health endpoint
npm run queue                # Redis queue statistics
npm run redis:logs           # View Redis logs
```

### Development Info

```bash
npm run help                 # Show available commands
```

## 🆘 Troubleshooting Commands

### When Things Go Wrong

```bash
npm run emergency:fix        # Nuclear restart (root)
npm run cleanup:nuclear      # Full cleanup (root)
npm run system:full-restart  # Complete system restart
```

### Manual Recovery

```bash
# If npm scripts fail, use PowerShell directly:
cd backend
.\cleanup.ps1               # Manual cleanup
.\start_system.ps1          # Manual startup
.\status.ps1                # Manual status check
```

## 🏗️ Architecture Integration

### PowerShell Scripts Integration

The package.json scripts are fully integrated with our robust PowerShell scripts:

- **npm start** → `start_system.ps1` (with cleanup first)
- **npm stop** → `cleanup.ps1` (kills conflicting processes)
- **npm restart** → `restart.ps1` (cleanup + start)
- **npm run status** → `status.ps1` (comprehensive health check)

### Port Management

- **Frontend**: Port 3000 (Next.js)
- **Backend API**: Port 8000 (FastAPI)
- **Redis**: Port 6379 (Docker container)
- **Worker**: No port (connects to Redis queue)

### Process Management

All scripts handle:

- ✅ Process conflict detection
- ✅ Port cleanup before startup
- ✅ Redis container management
- ✅ GPU worker lifecycle
- ✅ Graceful shutdowns

## 🎯 Best Practices

### Development Workflow

```bash
# 1. First time setup
npm run system:setup

# 2. Daily development
npm start                    # One command to rule them all!

# 3. Check health anytime
npm run status

# 4. Clean restart if needed
npm restart

# 5. Emergency recovery
npm run emergency:fix
```

### Production Deployment

```bash
# Build everything
npm run build:all

# Start production
npm run start:robust
```

## 🔄 Migration from v1.0

Old v1.0 commands → New v2.0 equivalent:

- `npm run dev:full` → `npm start`
- `npm run backend:start` → `npm run start:backend`
- `npm run redis:start` → Handled automatically
- `npm run worker:start` → Handled automatically

## 📝 Notes

- All scripts are **Windows-compatible** (PowerShell + cmd)
- **Error handling** built into every command
- **Process conflicts** automatically resolved
- **Redis management** fully automated
- **GPU workers** start automatically
- **Frontend** connects seamlessly to backend

🎉 **The system is now completely "airtight" - no more startup conflicts!**
