# Skill: Hardware Optimizer

> **ID:** `hardware-optimizer`  
> **Versión:** 1.0.0  
> **Autor:** IVOA Core Team  
> **Tipo:** edge | performance | hardware-acceleration

## Descripción

Motor de detección automática de aceleradores de cómputo (NVIDIA CUDA, Intel OpenVINO, Apple Silicon y CPU) que convierte y cachea los modelos YOLOv8 de IVOA en formatos de alto rendimiento (TensorRT FP16, OpenVINO IR, ONNX Runtime).

## Cuándo usarla

- Durante la inicialización de `ivoa_engine.py` para cargar los 5 modelos de visión en su versión más rápida.
- Al desplegar en hardware heterogéneo en sucursales físicas (NUCs Intel, estaciones con GPUs RTX/GTX o procesadores ARM/Jetson).
- Al actualizar pesos neuronales (`.pt`) para re-optimizar de forma transparente.

## Entradas

| Parámetro       | Tipo          | Descripción                                                                 | Requerido | Default |
|-----------------|---------------|-----------------------------------------------------------------------------|-----------|---------|
| `model_path`    | `str \| Path` | Nombre o ruta relativa al archivo `.pt` original (ej. `yolov8_ivoa_v9.pt`). | Sí        | —       |
| `imgsz`         | `int`         | Resolución espacial cuadrada de inferencia (640 estándar).                  | No        | `640`   |
| `half`          | `bool`        | Habilitar cuantización FP16 si el hardware lo soporta.                      | No        | `True`  |
| `force_rebuild` | `bool`        | Forzar recompilación ignorando la caché.                                    | No        | `False` |

## Salidas

| Retorno        | Tipo                        | Descripción                                                                     |
|----------------|-----------------------------|---------------------------------------------------------------------------------|
| `model`        | `YOLO` / wrapper compatible | Instancia de modelo lista para inferir fotogramas en memoria RAM.               |
| `backend_info` | `dict` (vía `get_report()`) | Metadatos de la aceleración aplicada (formato, dispositivo, caché).             |

## Matriz de Selección de Backend

```
¿torch.cuda.is_available() y TensorRT disponible?
├── SÍ ──► TensorRT (.engine) [FP16]
└── NO ──► ¿OpenVINO instalado o CPU Intel?
    ├── SÍ ──► OpenVINO IR (_openvino_model) [FP16]
    └── NO ──► ¿Apple Silicon (Darwin ARM64)?
        ├── SÍ ──► CoreML / ONNX CoreMLExecutionProvider
        └── NO ──► ONNX Runtime CPU / Fallback PyTorch (.pt)
```

## Dependencias

### Core (ya presentes normalmente)
- `ultralytics>=8.3.0`
- `torch>=2.0.0`
- `numpy`
- `onnx>=1.14.0`

### Opcionales según silicio de la sucursal
```bash
# NVIDIA (TensorRT)
pip install tensorrt

# Intel NUC / OpenVINO
pip install openvino

# CPU / fallback acelerado
pip install onnxruntime
```

## Ejemplo de uso

```python
from model_optimizer import get_optimized_model, HardwareOptimizer

# Carga automática con hardware acceleration y caché
model = get_optimized_model("yolov8_ivoa_v9.pt", imgsz=640, half=True)

# Inferencia directa en memoria (NumPy array)
results = model(frame, verbose=False, imgsz=640, conf=0.25)[0]

# Diagnóstico
report = HardwareOptimizer.get_instance().get_report()
print(report["backend"], report["device_name"])
```

## Notas de implementación

- Todo el ciclo de optimización e inferencia ocurre en memoria (VRAM/RAM).
- La exportación es *lazy* (solo la primera vez o cuando cambia el hash del `.pt`).
- Fallback graceful: si falta el compilador (ej. TensorRT no instalado), degrada a ONNX o `.pt` sin tumbar el sensor.
- Compatible con `supervision.Detections.from_ultralytics`.
- Cumple LFPDPPP: cero escritura de frames/imágenes a disco.

## Evaluación

- Smoke test: cargar `yolov8n.pt`, exportar, inferir un frame sintético `np.zeros((640,640,3), dtype=np.uint8)`.
- Métricas objetivo: reducción ≥ 40 % de latencia de inferencia vs PyTorch puro en GPU NVIDIA; ≥ 25 % en Intel OpenVINO.
