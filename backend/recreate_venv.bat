@echo off
echo ========================================
echo    FOTO RENDER VENV RECREATOR
echo ========================================
echo.

echo 🗑️ Removing old virtual environment...
if exist "venv" (
    rmdir /s /q venv
    echo ✅ Old venv removed
) else (
    echo ℹ️ No existing venv found
)

echo.
echo 🐍 Checking Python version...
python --version
echo.

echo 📦 Creating fresh virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ❌ Failed to create venv. Check if Python is properly installed.
    pause
    exit /b 1
)

echo ✅ Virtual environment created
echo.

echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

echo 📈 Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 🚀 Installing PyTorch (this may take a while)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo 🧠 Installing core AI libraries...
pip install transformers>=4.25.0
pip install diffusers>=0.21.0  
pip install accelerate>=0.20.0
pip install safetensors>=0.3.0

echo.
echo 🌐 Installing web framework...
pip install fastapi>=0.95.0
pip install uvicorn>=0.20.0
pip install python-multipart

echo.
echo 🛠️ Installing utilities...
pip install pillow>=9.0.0
pip install numpy>=1.24.0
pip install requests>=2.28.0
pip install pydantic>=2.0.0

echo.
echo 🎯 Installing local wheels (if available)...
if exist "setup\*.whl" (
    for %%f in (setup\*.whl) do (
        echo Installing %%f...
        pip install "%%f" --force-reinstall
    )
)

echo.
echo 🧪 Testing core imports...
python -c "import sys; print(f'Python: {sys.version}')"
python -c "import torch; print(f'PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

echo.
echo 🎉 Virtual environment recreated successfully!
echo 💡 You can now try running: npm start
echo.
pause 