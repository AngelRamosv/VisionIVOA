# Skill: Bill Validator

> **ID:** `bill-validator`  
> **Versión:** 1.0.0  
> **Autor:** IVOA Core Team  
> **Tipo:** edge | cash-handling | anti-hallucination | business-rules

## Descripción

Validador multi-capa especializado en detección y confirmación de billetes y manipulación de efectivo en áreas de cajas, ventanillas bancarias y mostradores de retail. Suprime falsos positivos provocados por sombras en paredes, escritorios, reflejos en monitores y paquetes de comida, otorgando prioridad estricta al flujo de efectivo.

## Cuándo usarla

- Al procesar candidatos crudos de billetes provenientes del modelo especializado (`yolov8_ivoa_v9.pt`, clase 1).
- Para evitar que la presencia de monitores encendidos o paredes blancas genere alertas falsas de efectivo.
- Para verificar que el billete está siendo efectivamente manipulado por una persona (dentro del radio de alcance de brazos y manos).
- Para desambiguar situaciones de contacto donde un billete pueda solaparse visualmente con alimentos (clase 4) o productos de maquillaje (clase 10).

## Entradas

| Parámetro | Tipo | Descripción | Requerido | Default |
| :--- | :--- | :--- | :--- | :--- |
| `bbox` | `np.ndarray \| list[float]` | Coordenadas `[x1, y1, x2, y2]` del billete candidato. | Sí | — |
| `conf` | `float` | Nivel de confianza retornado por el modelo neuronal. | Sí | — |
| `monitors_xyxy` | `np.ndarray` | Recuadros de monitores/pantallas detectados por el modelo base (COCO 62/63). | Sí | — |
| `persons_xyxy` | `np.ndarray` | Recuadros de personas detectadas por el modelo base (COCO 0). | Sí | — |

## Salidas

| Retorno | Tipo | Descripción |
| :--- | :--- | :--- |
| `is_valid` | `bool` | `True` si el candidato supera todos los filtros contextuales. |
| `valid_candidates` | `tuple[list, list, list]` | Recuadros, clases remapeadas (clase 2) y confianzas de los billetes confirmados. |

## Filtros Anti-Alucinación Implementados

1. **Restricción de Escala ($< 40\%$ de pantalla):**
   Un billete nunca ocupa más del $40\%$ del ancho o alto de la cámara. Cajas de mayor tamaño corresponden a falsas detecciones sobre paredes o mostradores.
2. **Filtro Anti-Monitor:**
   Si el centro geométrico del candidato cae dentro de un monitor (con un margen de tolerancia de $20\,\text{px}$), se descarta como reflejo o imagen en pantalla.
3. **Filtro de Manipulación Humana:**
   El billete debe solapar con la caja delimitadora de una persona humana expandida en $80\,\text{px}$ (área de alcance de manos y torso). Si no hay persona cerca, se ignora.
4. **Desambiguación de Prioridad (Efectivo > Alimentos / Maquillaje):**
   En ventanillas bancarias y cajas de cobro, la manipulación de dinero tiene prioridad absoluta. Si una caja de comida o maquillaje colisiona con una de billetes, las clases secundarias son automáticamente suprimidas.

## Dependencias

- `numpy`
- `opencv-python`
- `supervision`
- *(Opcional)* `easyocr` para verificación de denominaciones

## Ejemplo de uso

```python
from bill_validator import BillValidator

# Inicialización
validator = BillValidator(width=1280, height=720, min_conf=0.06)

# 1. Validación de candidatos del modelo v9
v9_boxes, v9_classes, v9_confs = validator.validate_v9_candidates(
    detections_v9_raw=detections_v9_raw,
    monitors_xyxy=monitores_reales,
    persons_xyxy=personas_reales
)

# 2. Desambiguación de prioridades frente a comida o maquillaje
detections_limpias = validator.disambiguate_detections(detections_fusionadas)
```

## Notas de Rendimiento y Cumplimiento

- **Tiempo de ejecución en memoria:** La validación geométrica toma $< 0.1\,\text{ms}$ por cuadro sobre matrices NumPy.
- **Sin I/O de disco:** Los recortes y análisis permanecen estrictamente en RAM, garantizando total cumplimiento con la normativa de protección de datos (LFPDPPP).
