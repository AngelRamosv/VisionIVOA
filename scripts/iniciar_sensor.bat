@echo off
title IVOA Edge Sensor (Puerto 5000)
cd /d "%~dp0\.."
echo Cargando base de datos historica...
echo Iniciando IVOA Edge Sensor en http://localhost:5000...
.\.venv\Scripts\python.exe src/edge_sensor/app_api.py
pause
