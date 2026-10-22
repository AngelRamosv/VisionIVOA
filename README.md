# 👁️ IVOA (Inteligencia Operativa y Visión Artificial)

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-OpenAPI-009688?logo=fastapi)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Edge%20AI-yellow)
![UniFace](https://img.shields.io/badge/UniFace-ArcFace%20%2B%20SCRFD-purple)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-Inference%20Engine-brightgreen)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

**IVOA** es un ecosistema de visión artificial (Edge AI) y auditoría operativa desatendida. Utiliza modelos en cascada (Ultralytics YOLOv8), algoritmos de Tracking multiobjeto (ByteTrack) y reconocimiento biométrico facial avanzado impulsado por **UniFace (ONNX Runtime)** para auditar reglas de negocio en sucursales físicas (códigos de vestimenta, uso de celulares, abandono de clientes, persistencia de identidades) e informar incidencias en tiempo real mediante integraciones multicanal (WhatsApp/Dashboard).

## 📐 Arquitectura (Open-Spec-Driven)

El proyecto está diseñado bajo una arquitectura de especificación abierta (Open-Spec), donde los contratos de la API (OpenAPI/Swagger) y los modelos de datos actúan como el núcleo orquestador entre los nodos *Edge* (Sensores de IA) y el *Cloud Backend* (Control de Mando).

### Estructura del Repositorio

- `src/edge_sensor/`: Nodos de visión artificial y procesamiento de video local. Implementa inferencia asíncrona, *frame-skipping* inteligente, tracking de personas (ByteTrack) y motor biométrico facial desacoplado en ONNX (`biometric_engine.py`).
- `src/cloud_backend/`: Capa de orquestación (FastAPI). Sirve interfaces UI, almacena analíticas (`data/ivoa.db`) y gestiona el envío de notificaciones. Cumple estrictamente con el estándar **OpenAPI 3.0**.
- `dataset_v9/`: (Ignorado en git) Entornos locales de datos.
- `*.pt`: (Ignorado en git) Pesos de las redes neuronales de Ultralytics (YOLOv8).

## 🚀 Guía de Despliegue

### 1. Instalación de Dependencias
```bash
python -m venv .venv
# Activar entorno (En Windows):
.\.venv\Scripts\activate

# Instalar dependencias base y motor ONNX
pip install -r requirements.txt

# Vincular la biblioteca UniFace en modo desarrollo
pip install -e C:\Users\Angel\uniface
```

### 2. Ejecución de Nodos
La arquitectura es altamente acoplable gracias a la especificación de la API. Debes levantar ambos nodos:

**Cloud Backend (Orquestador y UI):**
```bash
./scripts/iniciar_cloud.bat
# La API Swagger (OpenAPI UI) estará disponible en: http://localhost:8000/docs
```

**Edge Sensor (Motor de Visión):**
```bash
./scripts/iniciar_sensor.bat
```

## 🧠 Modelos de IA Incluidos (Pipeline en Cascada)

1. **YOLOv8 (Detección de Objetos y Comportamiento):**
   - **model_base:** Tracking universal y detección general de personas y mobiliario.
   - **model_custom:** Especialista en uniformes, uso de celulares y consumo de alimentos.
   - **model_screens:** Detección de pantallas y sitios web no autorizados.
   - **model_grooming_cash:** Especialista en arreglo personal y manipulación de efectivo.

2. **UniFace (Motor Biométrico Facial Edge en ONNX Runtime):**
   - **SCRFD:** Detección facial ultra rápida y extracción de 5 landmarks anatómicos dentro del recorte superior de la persona.
   - **ArcFace (MNet):** Generación de embeddings faciales normalizados de 512 dimensiones para persistencia y re-identificación de clientes/empleados contra la base de datos local (`data/ivoa.db`).
   - **Caché en RAM de Identidades:** Inferencia en sub-milisegundos (\(O(1)\) en memoria) que previene caídas de FPS y bloqueos en disco.

## 📄 Contratos de API (OpenAPI)
Al correr el backend, el motor FastAPI auto-genera la especificación OpenAPI basada en los esquemas Pydantic (`schemas.py`). Todos los endpoints (ej. `/v1/findings`, `/v1/notify_config`) están fuertemente tipados.
