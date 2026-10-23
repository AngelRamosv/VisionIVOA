# Skill: Queue Analyzer

> **ID:** `queue-analyzer`  
> **Versión:** 1.0.0  
> **Autor:** IVOA Core Team  
> **Tipo:** edge | queue-analytics | customer-flow | session-lifecycle

## Descripción

Motor de análisis espacial y temporal de afluencia de personas en sucursales físicas. Gestiona polígonos de interés (fila de espera y módulo de atención), cruce bidireccional de línea virtual, detección de saturación de clientes, control de acceso no autorizado (tailgating), tiempos de atención y ciclo de vida de sesiones (atendido vs abandonado).

## Cuándo usarla

- Para monitorear en tiempo real la cantidad de clientes formados en fila (`fila_count`) y en atención en módulo (`modulo_count`).
- Para auditar el cumplimiento del horario de apertura de la sucursal (apertura tardía después de las 09:15 AM).
- Para alertar en tiempo real cuando la fila excede la capacidad recomendada ($\ge 4$ personas) y requiere apertura de ventanilla de apoyo.
- Para identificar accesos indebidos o simultáneos (tailgating) en puertas o torniquetes en ventanas $< 3.5\,\text{s}$.
- Para registrar métricas operativas de tiempo de espera y duración de trámite (`total_service_time`).
- Para detectar abandono de clientes (salida tras $> 60\,\text{s}$ sin recibir atención).

## Entradas

| Método | Parámetros | Descripción |
| :--- | :--- | :--- |
| `update` | `frame: np.ndarray`, `detections_tracked: sv.Detections`, `now_ts: float` | Procesa fotograma y posiciones de personas con ID de tracking. |

## Salidas y Propiedades Expuestas

### Propiedades Públicas:
| Propiedad | Tipo | Descripción |
| :--- | :---: | :--- |
| `fila_count` | `int` | Número actual de personas dentro del polígono de fila de espera. |
| `modulo_count` | `int` | Número actual de personas dentro del módulo de atención / mostrador. |
| `in_count` | `int` | Total acumulado de entradas cruzadas a través de la línea virtual. |
| `out_count` | `int` | Total acumulado de salidas cruzadas a través de la línea virtual. |
| `late_opening_trigger_time` | `float` | Timestamp del evento de apertura tardía para banner visual temporal. |
| `late_opening_time_str` | `str` | Hora formateada leída por OCR o sistema de la apertura tardía. |

### Retorno de `update(...)`:
- `alerts`: Lista de tuplas `(event_type, mensaje, severidad)`.
- `sessions_to_update`: Lista de tuplas `(tracker_id, session_dict)` para persistencia en base de datos.
- `tailgating_boxes`: Recuadros de personas involucradas en acceso no autorizado (para clase visual 15).

## Eventos Generados

1. **`incidencia_saturacion` (warning):** Emitida cuando `fila_count >= 4` de forma continua durante `DEBOUNCE_SECONDS`.
2. **`incidencia_seguridad` (critical):** Emitida cuando dos IDs de personas distintas cruzan la línea en $< 3.5\,\text{s}$ (tailgating).
3. **`late_opening` (warning):** Emitida al detectar al primer cliente si la hora supera las 09:15 AM.
4. **`excessive_service_time` (warning):** Emitida si una persona permanece en módulo $> 15\,\text{s}$ (calibrado para demo).
5. **`service_completed` (info):** Emitida cuando una persona concluye su atención y abandona el módulo, registrando `total_service_time`.
6. **`customer_abandonment` (warning):** Emitida si una persona estuvo en fila y desaparece de escena tras $> 60\,\text{s}$ sin haber entrado a módulo.

## Dependencias

- `numpy`
- `opencv-python`
- `supervision`
- *(Opcional)* `easyocr` para lectura de reloj de cámara en video

## Ejemplo de uso

```python
from queue_analyzer import QueueAnalyzer

analyzer = QueueAnalyzer(width=1280, height=720)

# En cada cuadro:
alerts, sessions, tailgating_boxes = analyzer.update(
    frame=frame,
    detections_tracked=detections_tracked,
    now_ts=now_ts
)

# Acceso a métricas en tiempo real
print(f"Fila: {analyzer.fila_count} | Módulo: {analyzer.modulo_count}")
print(f"Entradas: {analyzer.in_count} | Salidas: {analyzer.out_count}")
```

## Notas de Rendimiento

- Todo el cálculo de contención poligonal y cruce de líneas se ejecuta en $< 0.5\,\text{ms}$ por frame utilizando las estructuras vectorizadas de `supervision`.
- Las sesiones inactivas se purgan automáticamente de memoria para mantener un consumo de RAM plano durante semanas de operación continua.
