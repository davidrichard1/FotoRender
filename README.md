# Foto-Render

A high-performance local image generation app using non-gated SDXL models, optimized for RTX 5090 GPU.

## Features

- High-quality SDXL image generation
- GPU-accelerated with CUDA support
- Modern Next.js web interface
- Customizable generation parameters
- Responsive design
- Optimized for RTX 5090
- Non-gated models only (no login required)

## Requirements

- **GPU**: NVIDIA RTX 5090 (or compatible CUDA GPU)
- **VRAM**: 12GB+ recommended
- **Python**: 3.9+
- **Node.js**: 18+
- **CUDA**: 12.1+
- **Storage**: ~10GB free space

## Project Structure

```
foto-render/
├── backend/
│   ├── main.py              # FastAPI server with image generation
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── src/                 # Next.js frontend source
│   └── package.json         # Frontend dependencies
├── models/                  # SDXL model storage
├── loras/                   # LoRA files
├── embeddings/              # Text embeddings
├── upscalers/               # Upscaling models
├── vaes/                    # VAE models
├── package.json             # Root project configuration
└── README.md               # This file
```

## Quick Setup

### 1. Install All Dependencies

```bash
# Install both frontend and backend dependencies
npm run install:all
```

### 2. Download the Model

You'll need to manually download the SDXL model:

1. Visit Civitai and find a model that suits your needs (Must be SDXL-based, but sometimes it's not too obvious)
2. Download the latest version SafeTensor file (6.46 GB) - or any version as long as it is SD / SDXL Compatible
3. Save as `models/myselectedmodel.safetensors`

### 3. Start the Application

```bash
# Start both frontend and backend together
npm run dev:full

# Or start them separately:
npm run frontend:dev    # Frontend only (Next.js)
npm run backend:dev     # Backend only (FastAPI)
```

### 4. Open the App

Navigate to: [http://localhost:3000](http://localhost:3000)

## Available Scripts

### Development

- `npm run dev:full` - Start both frontend and backend
- `npm run dev:full:network` - Start both with network access
- `npm run frontend:dev` - Start frontend only
- `npm run backend:dev` - Start backend only

### Production

- `npm run build` - Build frontend for production
- `npm run start` - Start both frontend and backend in production mode
- `npm run backend:start` - Start backend with uvicorn

### Maintenance

- `npm run install:all` - Install all dependencies
- `npm run lint` - Run frontend linting
- `npm run clean` - Clean build artifacts and cache

## Usage

1. Ensure the model is downloaded and placed in `models/`
2. Start the application: `npm run dev:full`
3. Open [http://localhost:3000](http://localhost:3000)
4. Enter your prompt and generate images!

## API Endpoints

- `GET /health` - Check server and model status
- `POST /generate` - Generate images with parameters

## Performance Optimization

The app includes several optimizations for RTX 5090:

- **Flash Attention**: PyTorch native attention optimization
- **VAE Slicing**: Reduced memory usage
- **VAE Tiling**: Support for large images
- **FP16/FP8**: Half and quarter precision for speed
- **CUDA**: GPU acceleration with TF32 support

## Memory Usage

Typical VRAM usage:

- Model Loading: ~7GB
- 1024x1024 generation: ~8-9GB
- 1344x1344 generation: ~10-11GB

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**

   - Reduce image dimensions
   - Lower batch size
   - Enable VAE slicing (auto-enabled)

2. **Model Not Loading**

   - Verify model file exists in `models/` directory
   - Check file permissions
   - Ensure sufficient disk space

3. **Frontend Not Loading**

   - Run `npm run frontend:install`
   - Check if port 3000 is available
   - Verify Node.js version (>=18)

4. **Backend Not Starting**
   - Run `npm run backend:install`
   - Check Python version (>=3.9)
   - Verify CUDA installation

## Development

### Adding New Features

1. **Frontend**: Work in `frontend/src/`
2. **Backend**: Modify `backend/main.py`
3. **Dependencies**: Update respective `package.json` or `requirements.txt`

### Hot Reloading

Both frontend and backend support hot reloading during development:

- Frontend: Next.js hot reload
- Backend: FastAPI auto-reload

## License

This project is for educational/research purposes. Please respect the model's license terms from CivitAI.

## Support

For issues with:

- This app: Check GitHub issues
- CUDA/GPU: Check NVIDIA documentation
