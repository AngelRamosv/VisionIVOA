# Skill: Visual Annotator

> **ID:** `visual-annotator`  
> **Versión:** 1.0.0  
> **Autor:** IVOA Core Team  
> **Tipo:** edge | visualization | annotation | ui-rendering

## Descripción

Motor de renderizado visual y superposición gráfica sobre fotogramas de video en tiempo real. Aísla toda la lógica de dibujo de `BoxAnnotator`, `LabelAnnotator` y `TraceAnnotator`, generación de etiquetas dinámicas (estados de debounce, indicador `[TICKET EMITIDO]`, nombres biométricos), máscaras de oscurecimiento de geofencing poligonal y carteles de advertencia de apertura tardía.

## Cuándo usarla

- Para dibujar en el frame de OpenCV los recuadros de detección y seguimiento que se transmiten al panel web / dashboard.
- Para informar visualmente al supervisor en el stream sobre el progreso de confirmación de una infracción (ej. `celular (0.3s / 0.5s)`).
- Para mostrar un distintivo visible `[TICKET EMITIDO]` durante $3.0\,\text{s}$ inmediatamente después de que se despacha una alerta a WhatsApp.
- Para oscurecer de forma suave las áreas fuera del polígono activo de la sucursal.
- Para desplegar un banner superior rojo cuando se detecta apertura tardía.

## Entradas

| Parámetro | Tipo | Descripción | Requerido | Default |
| :--- | :--- | :--- | :--- | :--- |
| `frame` | `np.ndarray` | Fotograma original en formato BGR. | Sí | — |
| `detections` | `sv.Detections` | Detecciones filtradas de objetos e infracciones operativas. | Sí | — |
| `detections_tracked` | `sv.Detections` | Personas con ID asignado por ByteTrack. | Sí | — |
| `id_map` | `dict[int, int]` | Mapeo de IDs de seguimiento a identidades persistentes. | No | `{}` |
| `known_identities` | `dict[str, str]` | Diccionario de nombres reconocidos por UniFace ArcFace. | No | `{}` |
| `custom_polygon_points` | `np.ndarray` | Puntos del polígono de geofencing `[[x, y], ...]`. | No | `None` |
| `last_alert_times` | `dict[str, float]` | Últimos timestamps de emisión por tag de alerta. | No | `{}` |
| `detection_start_times` | `dict[str, float]` | Timestamps iniciales de debounce por tag. | No | `{}` |
| `late_opening_trigger_time` | `float` | Timestamp del evento de apertura tardía. | No | `0.0` |
| `late_opening_time_str` | `str` | Texto con la hora de apertura tardía (ej. `09:23`). | No | `""` |

## Salidas

| Retorno | Tipo | Descripción |
| :--- | :--- | :--- |
| `annotated` | `np.ndarray` | Fotograma renderizado con todas las anotaciones gráficas. |

## Lógica y Paleta Calibrada

- **Paleta de Colores Personalizada:**
  - Naranja brillante (`#FFA500` / RGB `255, 165, 0`) para Arreglo Personal (clase 10).
  - Naranja alerta (`#FF5722` / RGB `255, 87, 34`) para Uniforme Incompleto (clase 9).
  - Amarillo (`#00FFFF` / BGR `0, 255, 255`) para el perímetro del polígono activo.
- **Trazas de Seguimiento:** Trazas suaves de movimiento de longitud 20 cuadros para cada persona rastreada.
- **Geofencing:** Máscara de transparencia con `cv2.addWeighted` al $40\%$ de luminosidad sobre zonas inactivas.

## Dependencias

- `numpy`
- `opencv-python`
- `supervision`

## Ejemplo de uso

```python
from visual_annotator import VisualAnnotator

annotator = VisualAnnotator(width=1280, height=720)

# Renderizado en cada cuadro de visualización
annotated_frame = annotator.render(
    frame=frame,
    detections=detections,
    detections_tracked=detections_tracked,
    id_map=id_map,
    known_identities=known_identities,
    custom_polygon_points=custom_polygon_points,
    last_alert_times=alert_manager.last_alert_times,
    detection_start_times=alert_manager.detection_start_times,
    late_opening_trigger_time=queue_analyzer.late_opening_trigger_time,
    late_opening_time_str=queue_analyzer.late_opening_time_str
)
```

## Notas de Rendimiento

- Todo el renderizado gráfico ocurre en memoria RAM mediante operaciones vectorizadas de OpenCV y NumPy.
- La ejecución toma $\approx 1.5\,\text{ms}$ por fotograma en resoluciones de $720\text{p}$, sin impacto perceptible en los FPS de transmisión MJPEG o WebRTC.
