@echo off
cd /d "%~dp0"
call venv\Scripts\activate.bat
set CUDA_VISIBLE_DEVICES=0
python local_gpu_worker.py 