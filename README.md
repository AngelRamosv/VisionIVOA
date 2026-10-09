# IVOA (Inteligencia Operativa y Visión Artificial)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-OpenAPI-009688?logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Edge%20AI-yellow)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

**IVOA** es un ecosistema de visión artificial (Edge AI) y auditoría operativa desatendida. Utiliza modelos en cascada (Ultralytics YOLOv8) y algoritmos de Tracking para auditar reglas de negocio en sucursales físicas (códigos de vestimenta, uso de celulares, abandono de clientes) e informar incidencias en tiempo real mediante integraciones multicanal (WhatsApp/Dashboard).

## 📐 Arquitectura (Open-Spec-Driven)

El proyecto está diseñado bajo una arquitectura de especificación abierta (Open-Spec), donde los contratos de la API (OpenAPI/Swagger) y los modelos de datos actúan como el núcleo orquestador entre los nodos *Edge* (Sensores de IA) y el *Cloud Backend* (Control de Mando).

### Estructura del Repositorio

- src/edge_sensor/: Nodos de visión artificial y procesamiento de video local. Implementa inferencia asíncrona, *frame-skipping* inteligente y tracking de objetos (ByteTrack).
- src/cloud_backend/: Capa de orquestación (FastAPI). Sirve interfaces UI, almacena analíticas (SQLite) y gestiona el envío de notificaciones. Cumple estrictamente con el estándar **OpenAPI 3.0**.
- dataset_v9/: (Ignorado en git) Entornos locales de datos.
- *.pt: (Ignorado en git) Pesos de las redes neuronales de Ultralytics (YOLOv8).

## 🚀 Guía de Despliegue

### 1. Instalación de Dependencias
`ash
python -m venv .venv
source .venv/bin/activate  # (En Windows: .venv\Scripts\activate)
pip install -r requirements.txt
`

### 2. Ejecución de Nodos
La arquitectura es altamente acoplable gracias a la especificación de la API. Debes levantar ambos nodos:

**Cloud Backend (Orquestador y UI):**
`ash
./iniciar_cloud.bat
# La API Swagger (OpenAPI UI) estará disponible en: http://localhost:8000/docs
`

**Edge Sensor (Motor de Visión):**
`ash
./iniciar_sensor.bat
`

## 🧠 Modelos de IA Incluidos (Pipeline en Cascada)
- **model_base:** Tracking universal y detección general.
- **model_custom:** Especialista en uniformes, uso de celulares y consumo de alimentos.
- **model_screens:** Detección de pantallas y sitios web no autorizados.
- **model_grooming_cash:** Especialista en arreglo personal y manipulación de efectivo.

## 📄 Contratos de API (OpenAPI)
Al correr el backend, el motor FastAPI auto-genera la especificación OpenAPI basada en los esquemas Pydantic (schemas.py). Todos los endpoints (ej. /v1/findings, /v1/notify_config) están fuertemente tipados.
