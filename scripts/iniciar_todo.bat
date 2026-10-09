@echo off
title Lanzador IVOA - Supervision
echo =======================================================
echo          INICIANDO PLATAFORMA DE SUPERVISION IVOA
echo =======================================================
echo.

cd /d "%~dp0"

echo [*] Cargando base de datos historica...
echo    -^> Historial y registros mantenidos.

echo [1/2] Iniciando Cloud Backend API (FastAPI en http://localhost:8001)...
start "IVOA - Cloud Backend (Puerto 8001)" cmd /k "title IVOA Cloud Backend && .\.venv\Scripts\python.exe -m uvicorn src.cloud_backend.main:app --host 0.0.0.0 --port 8001"

timeout /t 2 /nobreak >nul

echo [2/2] Iniciando Edge Sensor (Flask + IA en http://localhost:5000)...
start "IVOA - Edge Sensor (Puerto 5000)" cmd /k "title IVOA Edge Sensor && .\.venv\Scripts\python.exe src/edge_sensor/app_api.py"

echo.
echo =======================================================
echo Ambos servicios estan iniciando en ventanas separadas.
echo.
echo  * Dashboard de Sucursal (Edge Sensor): http://localhost:5000
echo  * Dashboard Central de Auditoria:     http://localhost:8001
echo    (Usuario: admin / Contrasena: admin123)
echo =======================================================
echo.
pause
