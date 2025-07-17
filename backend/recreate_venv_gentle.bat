@echo off
echo ========================================
echo   FOTO RENDER VENV RECREATOR (GENTLE)
echo ========================================
echo.

echo 🔄 Renaming old virtual environment...
if exist "venv" (
    if exist "venv_old" rmdir /s /q venv_old >nul 2>&1
    ren venv venv_old >nul 2>&1
    echo ✅ Old venv renamed to venv_old
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
echo 🚀 Installing basic packages first...
pip install wheel setuptools

echo.
echo 🧠 Installing PyTorch (simplified)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo 🌐 Installing core libraries...
pip install fastapi uvicorn python-multipart
pip install pillow numpy requests pydantic

echo.
echo 🧪 Quick test...
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"

echo.
echo 🎉 Basic environment ready!
echo 💡 You can now try: npm start
echo 🗑️ To clean up later: rmdir /s /q venv_old
echo.
pause 