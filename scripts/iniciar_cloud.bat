@echo off
title IVOA Cloud Backend (Puerto 8001)
cd /d "%~dp0\.."
echo Iniciando IVOA Cloud Backend en http://localhost:8001...
.\.venv\Scripts\python.exe -m uvicorn src.cloud_backend.main:app --host 0.0.0.0 --port 8001
pause
