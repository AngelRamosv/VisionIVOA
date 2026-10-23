# Guía para Agentes de IA — IVOA

Este documento es el punto de entrada para cualquier agente de IA o desarrollador que trabaje en el repositorio.

---

## 1. Estado Actual de Prioridades

| Prioridad | Módulo / Skill | Descripción | Estado |
| :--- | :--- | :--- | :---: |
| **P1** | `hardware-optimizer` | Exportador y cargador automático acelerado (ONNX, OpenVINO, CoreML) con caché persistente en disco. | ✅ Completado |
| **P2** | `webrtc-streaming` / `frame-governor` | Streaming de baja latencia y descarte de frames acumulados en memoria (*drop older*). | ✅ Completado |
| **P3** | Desacoplamiento de `ivoa_engine.py` | Orquestador delgado delegando a 6 submódulos especializados (`detection_pipeline`, `bill_validator`, `business_rules`, `queue_analyzer`, `alert_manager`, `visual_annotator`). | ✅ Completado |
| **P4** | `benchmark-harness` | Suite automatizada de benchmarks de FPS, latencia p95 y matriz de confusión/FPR en `eval/`. | ✅ Completado |
| **P5** | `privacy-blur` | Anonimización y ofuscación facial selectiva (LFPDPPP/GDPR). | ✅ Completado |
| **P6** | `vlm-forensics` | Consultas forenses en lenguaje natural. | ⏳ Pendiente |

---

## 2. Mapa de Módulos Core (`src/edge_sensor/`)

1. **`ivoa_engine.py`**: Orquestador principal (~300 líneas). Coordina el ciclo de vida de los frames en 7 pasos y mantiene compatibilidad total con `app_api.py`.
2. **`model_optimizer.py`**: Detecta hardware disponible (CUDA, ROCm, OpenVINO, MPS, ONNX, CPU) y optimiza modelos YOLO en caché local.
3. **`detection_pipeline.py`**: Inferencia en cascada de modelos YOLO, recortes de zoom óptico, ByteTrack y biometría UniFace ArcFace.
4. **`bill_validator.py`**: Validación relacional anti-alucinaciones en conteo de efectivo (pantallas, paredes, interacción con manos).
5. **`business_rules.py`**: Reglas operativas (código de vestimenta HSV, comida a la boca, pantallas, posturas atípicas, geofencing).
6. **`queue_analyzer.py`**: Aforo de fila/módulo, conteo de cruces con `LineZone`, detección de saturación, tailgating y tiempos de servicio.
7. **`alert_manager.py`**: Máquina de estados con debouncing y cooldown, deduplicación de eventos y sync de `notify_config.json`.
8. **`visual_annotator.py`**: Renderizado gráfico Supervision, trazado de identidades, máscara de geofencing y banners.

---

## 3. Guía de Ejecución Rápida

- **Iniciar Edge Sensor**: `python src/edge_sensor/app_api.py`
- **Iniciar Cloud Backend**: `uvicorn src.cloud_backend.app:app --reload --port 8000`
- **Benchmark de Modelo**: `python -m eval.runners.benchmark_model --weights yolov8n.pt`
- **Evaluación de Pipeline**: `python -m eval.runners.evaluate_pipeline --dataset eval/datasets/cellphone_confusion`
