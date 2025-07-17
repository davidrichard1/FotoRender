@echo off
echo ========================================
echo      FOTO RENDER DEPENDENCY FIXER
echo ========================================
echo.

echo 🔧 Fixing Python dependencies...
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

echo ✅ Virtual environment activated
echo.

echo 🗑️ Uninstalling problematic packages...
pip uninstall -y xformers diffusers transformers torch torchvision torchaudio

echo.
echo 📦 Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo 🚀 Installing AI/ML dependencies...
pip install transformers>=4.40.0
pip install accelerate>=0.24.0
pip install diffusers>=0.24.0

echo.
echo 🛠️ Installing XFormers (might take a while)...
pip install xformers>=0.0.22 --no-deps

echo.
echo 🎯 Installing local wheels...
if exist "setup\sageattention-2.2.0+cu128torch2.7.1-cp313-cp313-win_amd64.whl" (
    pip install setup\sageattention-2.2.0+cu128torch2.7.1-cp313-cp313-win_amd64.whl
    echo ✅ SageAttention installed
)

if exist "setup\triton_windows-3.3.1.post19-cp313-cp313-win_amd64.whl" (
    pip install setup\triton_windows-3.3.1.post19-cp313-cp313-win_amd64.whl
    echo ✅ Triton installed
)

if exist "setup\basicsr-1.4.2-py3-none-any.whl" (
    pip install setup\basicsr-1.4.2-py3-none-any.whl
    echo ✅ BasicSR installed
)

echo.
echo 📋 Installing remaining requirements...
pip install -r requirements.txt --no-deps

echo.
echo 🧪 Testing imports...
python -c "import torch; print(f'✅ PyTorch {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python -c "import diffusers; print(f'✅ Diffusers {diffusers.__version__}')"
python -c "import transformers; print(f'✅ Transformers {transformers.__version__}')"

echo.
echo 🎉 Dependencies fixed! Try running the API now.
echo.
pause 