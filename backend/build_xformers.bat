@echo off
REM ================================================================
REM build_xformers.bat - Helper to compile and install xformers from source
REM Works for Windows 10/11 with:
REM   • Visual Studio 2022 Build Tools (MSVC v143, C++ CMake tools for Windows)
REM   • CUDA Toolkit 12.1 matching PyTorch 2.1 wheels
REM   • Python 3.10 / 3.11 (3.13 NOT yet officially supported)
REM   • Git in PATH
REM ================================================================

REM ---- PRE-FLIGHT CHECKS ----
where cl >nul 2>&1 || (
  echo [ERROR] MSVC compiler (cl.exe) not found in PATH.^^
  echo Install "Desktop development with C++" workload in Visual Studio Build Tools and reopen terminal.^^
  goto :eof
)

where cmake >nul 2>&1 || (
  echo [ERROR] CMake not found. Install CMake and ensure it is in PATH.^^
  goto :eof
)

python -c "import sys; exit(0 if sys.version_info[:2] <= (3, 11) else 1)" || (
  echo [WARN] Python version ^> 3.11 detected. xformers may not compile.^^
  echo Continue anyway? (Y/N)^^
  set /p _ans=^^
  if /I not "%_ans%"=="Y" exit /b 1
)

REM ---- CLONE SOURCE ----
set REPO_DIR=%TEMP%\xformers_src
if exist "%REPO_DIR%" rmdir /S /Q "%REPO_DIR%"

git clone --depth 1 --branch v0.0.23.post2 https://github.com/facebookresearch/xformers "%REPO_DIR%"
if %errorlevel% neq 0 (
  echo [ERROR] Failed to clone xformers repo.^^
  goto :eof
)

REM ---- INSTALL BUILD DEPENDENCIES ----
cd /d "%REPO_DIR%"
echo Installing Python build dependencies... this can take a moment.
pip install --upgrade pip setuptools wheel ninja packaging typing_extensions --quiet

REM ---- SET ENV VARIABLES ----
set "TORCH_CUDA_ARCH_LIST=7.5;8.0;8.6"
set USE_NINJA=1

REM ---- BUILD & INSTALL ----
python setup.py install --no-build-isolation
if %errorlevel% neq 0 (
  echo [ERROR] xformers build failed. Check the error log above.^^
  goto :eof
)

echo. 
echo [SUCCESS] xformers installed successfully!
cd "%~dp0" 