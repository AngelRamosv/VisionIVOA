# Skill: Alert Manager

> **ID:** `alert-manager`  
> **Versión:** 1.0.0  
> **Autor:** IVOA Core Team  
> **Tipo:** edge | rules | alerts | business-logic

## Descripción

Módulo de gestión de incidencias, máquinas de estado de persistencia temporal (debounce), enfriamiento anti-spam (cooldown) y atribución espacial de infractores/culpables para el sensor Edge de IVOA. Sincroniza dinámicamente las directivas de negocio con `config/notify_config.json`.

## Cuándo usarla

- Para evaluar detecciones continuas y determinar si una infracción (celular, alimentos, páginas indebidas, uniforme incompleto, arreglo personal, billetes, etc.) superó el umbral de persistencia temporal.
- Para evitar spam de notificaciones ejecutivas mediante ventanas de enfriamiento (`ALERT_COOLDOWN = 8.0s`).
- Para asociar geométricamente un objeto ilícito detectado con la persona rastreada más cercana en la escena.
- Para filtrar etiquetas y recuadros visuales en función de la configuración de privacidad y directivas activas de la sucursal.

## Entradas

| Parámetro | Tipo | Descripción | Requerido | Default |
| :--- | :--- | :--- | :--- | :--- |
| `detections` | `sv.Detections` | Objetos detectados en el cuadro actual tras filtrado espacial. | Sí | — |
| `detections_tracked` | `sv.Detections` | Personas con ID de seguimiento asignado por ByteTrack. | Sí | — |
| `id_map` | `dict[int, int]` | Mapeo de IDs de seguimiento a identidades persistentes. | No | `{}` |
| `known_identities` | `dict[str, str]` | Diccionario de identidades reconocidas por UniFace ArcFace. | No | `{}` |
| `now_ts` | `float` | Timestamp epoch actual. | No | `time.time()` |
| `extra_active_flags` | `dict[str, bool]` | Banderas auxiliares (ej. situación atípica por postura parada). | No | `{}` |

## Salidas

| Retorno | Tipo | Descripción |
| :--- | :--- | :--- |
| `alerts` | `list[tuple[str, str, str]]` | Lista de tuplas con formato `(event_type, mensaje, severidad)`. |

## Configuración y Reglas Calibradas

Las reglas incorporan tiempos de debounce específicos calibrados para el entorno de sucursales:

| Infracción | Clase ID | Debounce | Cooldown | Severidad | Mensaje Emitido |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Celular** | 5 | `0.5 s` | `8.0 s` | `warning` | `ALERTA: {culprit} detectado(a) usando dispositivo móvil (Celular).` |
| **Audífonos** | 7 | `1.0 s` | `8.0 s` | `warning` | `ALERTA: {culprit} detectado(a) usando audífonos de diadema.` |
| **Billetes / Dinero** | 2 | `1.0 s` | `8.0 s` | `warning` | `ALERTA: {culprit} detectado con manejo de billetes o efectivo.` |
| **Alimentos / Bebidas**| 4 | `1.0 s` | `8.0 s` | `warning` | `ALERTA: {culprit} detectado(a) consumiendo alimentos/bebidas.` |
| **Páginas no autorizadas** | 8 | `1.0 s` | `8.0 s` | `warning` | `ALERTA: {culprit} con ingreso a páginas / redes no autorizadas en monitor.` |
| **Arreglo Personal** | 10 | `1.5 s` | `8.0 s` | `warning` | `ALERTA: {culprit} realizando arreglo personal / maquillaje.` |
| **Uniforme Incompleto**| 9 | `1.0 s` | `8.0 s` | `warning` | `ALERTA: {culprit} con código de vestimenta incompleto.` |
| **Llaves** | 6 | `1.0 s` | `8.0 s` | `warning` | `ALERTA: {culprit} con llaves detectadas en área de atención.` |
| **Situación Atípica** | 11 | `1.0 s` | `8.0 s` | `warning` | `ALERTA: {culprit} detectado en Situación atípica (Persona de pie / Movimiento anómalo).` |

## Dependencias

- `numpy`
- `opencv-python`
- `supervision` (para estructuras `sv.Detections`)

## Ejemplo de uso

```python
from alert_manager import AlertManager

# Inicialización en el orquestador
alert_mgr = AlertManager(width=1280, height=720)

# En cada fotograma:
# 1. Sincronización automática de notify_config.json cada 30 cuadros
alert_mgr.sync_config(frame_count)

# 2. Evaluación de incidencias y debounce
new_alerts = alert_mgr.evaluate_detections(
    detections=detections,
    detections_tracked=detections_tracked,
    id_map=id_map,
    known_identities=known_identities,
    now_ts=now_ts,
    extra_active_flags={"atipica": alguien_parado}
)

# 3. Filtrado maestro según ajustes activos
final_alerts = alert_mgr.filter_active_alerts(new_alerts)
```

## Notas de Implementación

- **Aislamiento total:** No depende de modelos de inferencia pesados (YOLO/PyTorch), lo que permite probarlo con mocks de `Detections` de forma instantánea.
- **Gracia temporal:** Si un objeto se pierde brevemente por oclusión durante $< 1.0\,\text{s}$, el cronómetro de debounce no se reinicia inmediatamente.
- **Cumplimiento de Privacidad (LFPDPPP):** Procesa y formatea texto de alertas en memoria; no almacena ni escribe capturas en disco.
