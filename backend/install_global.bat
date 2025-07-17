@echo off
echo ========================================
echo   FOTO RENDER GLOBAL INSTALL (Python 3.13)
echo ========================================
echo.

echo 🐍 Checking Python version...
python --version
echo.

echo 📈 Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 🚀 Installing PyTorch with CUDA...
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 🧠 Installing AI/ML libraries...
python -m pip install transformers>=4.25.0
python -m pip install diffusers>=0.21.0  
python -m pip install accelerate>=0.20.0
python -m pip install safetensors>=0.3.0

echo.
echo 🌐 Installing web framework...
python -m pip install fastapi>=0.95.0
python -m pip install uvicorn>=0.20.0
python -m pip install python-multipart

echo.
echo 🛠️ Installing utilities...
python -m pip install pillow>=9.0.0
python -m pip install numpy>=1.24.0
python -m pip install requests>=2.28.0
python -m pip install pydantic>=2.0.0

echo.
echo 🎯 Installing local wheels (if available)...
if exist "setup\*.whl" (
    for %%f in (setup\*.whl) do (
        echo Installing %%f...
        python -m pip install "%%f" --force-reinstall
    )
)

echo.
echo 🧪 Testing imports...
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

echo.
echo 🎉 Global installation complete!
echo 💡 Now using system Python 3.13 (no venv needed)
echo.
pause 