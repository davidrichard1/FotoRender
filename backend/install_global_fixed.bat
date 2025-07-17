@echo off
echo ========================================
echo   FOTO RENDER GLOBAL INSTALL (FIXED)
echo ========================================
echo.

echo 🧹 Clearing virtual environment variables...
set VIRTUAL_ENV=
set PYTHONHOME=
set CONDA_DEFAULT_ENV=
set CONDA_PREFIX=

echo.
echo 🐍 Using Python 3.13 directly...
C:\Python313\python.exe --version

echo.
echo 📈 Upgrading pip...
C:\Python313\python.exe -m pip install --upgrade pip

echo.
echo 🚀 Installing PyTorch with CUDA...
C:\Python313\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 🧠 Installing AI/ML libraries...
C:\Python313\python.exe -m pip install transformers>=4.25.0
C:\Python313\python.exe -m pip install diffusers>=0.21.0  
C:\Python313\python.exe -m pip install accelerate>=0.20.0
C:\Python313\python.exe -m pip install safetensors>=0.3.0

echo.
echo 🌐 Installing web framework...
C:\Python313\python.exe -m pip install fastapi>=0.95.0
C:\Python313\python.exe -m pip install uvicorn>=0.20.0
C:\Python313\python.exe -m pip install python-multipart

echo.
echo 🛠️ Installing utilities...
C:\Python313\python.exe -m pip install pillow>=9.0.0
C:\Python313\python.exe -m pip install numpy>=1.24.0
C:\Python313\python.exe -m pip install requests>=2.28.0
C:\Python313\python.exe -m pip install pydantic>=2.0.0

echo.
echo 🧪 Testing imports...
C:\Python313\python.exe -c "import sys; print(f'Python: {sys.version}')"
C:\Python313\python.exe -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
C:\Python313\python.exe -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
C:\Python313\python.exe -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

echo.
echo 🎉 Installation complete using Python 3.13!
echo.
pause 