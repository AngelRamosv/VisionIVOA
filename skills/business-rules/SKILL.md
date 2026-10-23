# Skill: Business Rules

> **ID:** `business-rules`  
> **Versión:** 1.0.0  
> **Autor:** IVOA Core Team  
> **Tipo:** edge | operational-audit | heuristics | compliance

## Descripción

Motor de auditoría analítica y cumplimiento operativo en sucursales físicas. Evalúa el código de vestimenta de empleados (segmentación de color HSV), consumo de alimentos ("sólo a la boca"), pantallas o navegación en páginas no autorizadas, arreglo personal/maquillaje, posturas anómalas (situación atípica) y geofencing espacial por zonas o polígonos.

## Cuándo usarla

- Para auditar que el personal cumpla con el código de vestimenta corporativo (playera blanca/negra y pantalón formal negro, sin mezclilla).
- Para detectar colaboradores consumiendo alimentos o bebidas en mostradores de atención al cliente.
- Para identificar el uso de monitores en páginas o plataformas no autorizadas (YouTube, redes sociales).
- Para detectar colaboradores de pie o en posturas atípicas en áreas de escritorio.
- Para desambiguar falsos positivos de maquillaje cuando la persona en realidad está comiendo.
- Para aplicar geofencing poligonal manteniendo el rastreo de personas en toda la escena.

## Entradas

| Método | Parámetros Principales | Descripción |
| :--- | :--- | :--- |
| `evaluate_uniform` | `frame: np.ndarray`, `persons_xyxy: np.ndarray` | Recorta torso y piernas para análisis cromático HSV. |
| `filter_food_candidates` | `food_boxes`, `persons_xyxy`, `peripherals_xyxy` | Aplica regla de contención en el 50% superior del cuerpo. |
| `filter_unauthorized_screens` | `raw_boxes`, `raw_classes`, `raw_confs` | Valida escala de ventanas en pantalla (< 35%). |
| `detect_standing_posture` | `persons_xyxy: np.ndarray` | Evalúa ratio vertical (>= 1.65) o proximidad al techo (< 10%). |
| `disambiguate_food_vs_grooming` | `detections: sv.Detections` | Suprime arreglo personal solapado con alimentos. |
| `apply_geofencing` | `detections`, `custom_polygon_points`, `zona_izq`, `zona_der` | Filtra objetos fuera del área activa. |

## Salidas

- Cajas, clases y confianzas de las infracciones operativas:
  - Clase 9: `uniforme_incompleto` (confianza 0.92).
  - Clase 8: `pagina_no_autorizada`.
  - Clase 4: `comida` confirmada en boca/torso.
  - Clase 10: `arreglo_personal` centrado en rostro.
  - Clase 11: `situacion_atipica` por postura vertical.
- Objeto `sv.Detections` filtrado por reglas espaciales y de prioridad.

## Reglas y Lógica Calibrada

1. **Código de Vestimenta (HSV):**
   - **Torso central:** Porcentaje de blanco $\ge 55\%$ (`sat < 40`, `val > 180`).
   - **Piernas:** Porcentaje de negro $\ge 45\%$ (`val < 70`) y azul $< 15\%$ (descarta jeans/mezclilla).
   - Si no cumple, genera recuadro de `uniforme_incompleto`.
2. **Consumo de Alimentos:**
   - Centro geométrico en el $50\%$ superior del cuerpo humano.
   - Supresión automática si colisiona con teclado, mouse o esquina inferior de cámara.
3. **Situación Atípica:**
   - Persona completamente de pie: ratio $\text{altura}/\text{ancho} \ge 1.65$ y altura $> 40\%$ de la cámara, o cabeza en el $10\%$ superior de la escena.
4. **Desambiguación:**
   - La comida en mano o boca suprime el maquillaje superpuesto para evitar alertar maquillaje mientras se come un refrigerio.
5. **Geofencing:**
   - Las personas (clase 0) siempre se conservan en toda la cámara para no interrumpir el seguimiento de identidades.
   - Los objetos infractores fuera del polígono o de la zona activa son descartados.

## Dependencias

- `numpy`
- `opencv-python`
- `supervision`

## Ejemplo de uso

```python
from business_rules import BusinessRules

rules = BusinessRules(width=1280, height=720)

# 1. Auditar uniforme en el fotograma actual
u_boxes, u_classes, u_confs = rules.evaluate_uniform(frame, persons_xyxy)

# 2. Verificar postura de pie
alguien_parado, atipica_boxes = rules.detect_standing_posture(persons_xyxy)

# 3. Desambiguación comida vs maquillaje
detections = rules.disambiguate_food_vs_grooming(detections)

# 4. Geofencing espacial
detections = rules.apply_geofencing(detections, custom_polygon_points)
```

## Notas de Rendimiento y Cumplimiento

- **Procesamiento en memoria:** Toda la segmentación HSV y pruebas de polígono transcurren en memoria RAM en $< 1\,\text{ms}$.
- **Privacidad:** No se graban fotogramas a disco; se analizan únicamente los perfiles cromáticos numéricos requeridos para la regla de negocio.
