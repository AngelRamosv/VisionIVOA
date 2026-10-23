# Skill: [Nombre de la Skill]

> **ID:** `nombre-kebab-case`  
> **Versión:** 0.1.0  
> **Autor:** IVOA Team  
> **Tipo:** edge | cloud | hybrid | analysis | streaming | biometric

## Descripción

Breve descripción de lo que hace esta skill (1-3 frases).

## Cuándo usarla

- Caso de uso 1
- Caso de uso 2

## Entradas

| Nombre       | Tipo          | Descripción                     | Requerido |
|--------------|---------------|---------------------------------|-----------|
| `frame`      | np.ndarray    | Frame BGR de OpenCV             | Sí        |
| `config`     | dict          | Configuración específica        | No        |

## Salidas

| Nombre       | Tipo          | Descripción                     |
|--------------|---------------|---------------------------------|
| `result`     | dict          | Resultado estructurado          |
| `annotated`  | np.ndarray    | Frame anotado (opcional)        |

## Dependencias

- Python packages: `ultralytics`, `onnxruntime`, ...
- Modelos: `yolov8n.pt`, ...
- Hardware: CPU / CUDA / OpenVINO

## Configuración

```yaml
# Ejemplo de configuración
threshold: 0.45
device: auto
```

## Ejemplo de uso

```python
from skills.nombre_skill import run

result = run(frame=frame, config={"threshold": 0.5})
print(result)
```

## Notas de implementación

- Mantener todo en memoria (sin escribir a disco).
- Preferir ONNX / TensorRT cuando sea posible.
- Debe ser autocontenida y fácil de testear.

## Evaluación

- Dataset de prueba: `eval/datasets/...`
- Métricas: precisión, recall, FPS, latencia p95
