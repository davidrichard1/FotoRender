@echo off
echo ========================================
echo       CLEAN PYTHON TEST
echo ========================================
echo.

echo 🧹 Clearing virtual environment variables...
set VIRTUAL_ENV=
set PYTHONHOME=
set CONDA_DEFAULT_ENV=
set CONDA_PREFIX=

echo.
echo 🔍 Testing Python with full path...
C:\Python313\python.exe --version
if %errorlevel% neq 0 (
    echo ❌ Python 3.13 failed with full path
) else (
    echo ✅ Python 3.13 works with full path
)

echo.
echo 🔍 Testing Python with PATH...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python failed via PATH
    echo 🔧 PATH might need a system restart
) else (
    echo ✅ Python works via PATH
)

echo.
echo 📋 Current PATH (Python entries):
echo %PATH% | findstr /i python

echo.
pause 