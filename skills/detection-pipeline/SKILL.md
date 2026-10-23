# Skill: Detection Pipeline

> **ID:** `detection-pipeline`  
> **Versión:** 1.0.0  
> **Autor:** IVOA Core Team  
> **Tipo:** edge | inference | computer-vision | multitracking

## Descripción

Pipeline de bajo nivel de visión por computadora para el nodo Edge de IVOA. Administra la inferencia en cascada de los 5 modelos neuronales especializados (base, v5 vestimenta/accesorios, v7 pantallas, v8 arreglo personal, v9 billetes/llaves/situación atípica), ejecución de recortes de zoom óptico multi-escala, tracking de personas con ByteTrack, persistencia de identidades por posición/firmas de color HSV y extracción biométrica periódica UniFace ArcFace (SCRFD + ONNX Runtime).

## Cuándo usarla

- Como el motor de inferencia primario en cada cuadro procesado por el sensor de sucursal.
- Para rastrear de forma continua el movimiento de personas en piso de venta o mostradores de atención.
- Para mantener la identidad persistente de un colaborador o cliente aun cuando se siente, se levante o sufra oclusiones temporales detrás de un escritorio.
- Para asociar rostros con identidades biométricas almacenadas en la base de datos SQLite de IVOA.

## Entradas

| Parámetro | Tipo | Descripción | Requerido | Default |
| :--- | :--- | :--- | :--- | :--- |
| `frame` | `np.ndarray` | Fotograma original en memoria (matriz NumPy BGR). | Sí | — |
| `frame_count` | `int` | Número incremental de cuadro (para control de `FRAME_SKIP` y biometría cada 15 cuadros). | Sí | — |
| `now_ts` | `float` | Timestamp epoch actual. | No | `time.time()` |

## Salidas

| Retorno | Tipo | Descripción |
| :--- | :--- | :--- |
| `raw_outputs` | `dict[str, Any]` | Diccionario con las detecciones crudas de cada modelo (`base`, `custom_raw`, `screens_raw`, `v8_raw`, `v9_raw`, `zoom`, `peripherals`). |
| `detections_base` | `sv.Detections` | Personas confirmadas por el modelo base (confianza $\ge 0.55$, altura $\ge 60\,\text{px}$). |
| `detections_tracked` | `sv.Detections` | Personas con ID asignado y re-identificado de ByteTrack. |

## Modelos Integrados en Cascada

1. **`yolov8n.pt` (Base):** Tracking general de personas, celulares en mano y detección de periféricos/alimentos.
2. **`yolov8_ivoa_v5.pt` (Custom):** Especialista en audífonos de diadema, celulares y alimentos.
3. **`yolov8_ivoa_v7.pt` (Screens):** Especialista en monitores y páginas/redes sociales no autorizadas.
4. **`yolov8_ivoa_v8.pt` (Grooming):** Especialista en arreglo personal / maquillaje.
5. **`yolov8_ivoa_v9.pt` (V9):** Especialista en billetes, llaves y situación atípica.
6. **UniFace ONNX (SCRFD + ArcFace MNet):** Extracción de embeddings faciales de 512 dimensiones para re-ID contra SQLite.

## Dependencias

- `ultralytics>=8.3.0`
- `torch>=2.0.0`
- `supervision>=0.18.0`
- `numpy`
- `opencv-python`
- `uniface` (ONNX Runtime)
- `model_optimizer.py` (Hardware Optimizer P1)

## Ejemplo de uso

```python
from detection_pipeline import DetectionPipeline

pipeline = DetectionPipeline(width=1280, height=720, frame_skip=5)

# En cada cuadro del bucle de video:
raw_outputs, detections_base, detections_tracked = pipeline.process(
    frame=frame,
    frame_count=frame_count,
    now_ts=now_ts
)

# Acceso a identidades persistentes y reconocidas
print(pipeline.id_map)
print(pipeline.known_identities)
```

## Notas de Rendimiento y Cumplimiento

- **Aceleración de Hardware:** Si `model_optimizer.py` está presente, los modelos se cargan automáticamente desde la caché compilada en TensorRT FP16 (`.engine`) o ONNX Runtime.
- **Inferencia en memoria RAM:** Todo el procesamiento ocurre sobre tensores en GPU/RAM sin escribir jamás fotogramas en disco (cumplimiento LFPDPPP).
- **Frame-Skipping Inteligente:** Procesa inferencias pesadas cada $N$ cuadros (por defecto 5), reutilizando los estados en cuadros intermedios para garantizar 30 FPS fluidos en el stream.
