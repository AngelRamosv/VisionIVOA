---
name: benchmark-harness
description: Guía de uso y thresholds para la suite de evaluación y regresión automatizada de modelos y pipeline IVOA.
---

# Benchmark & Evaluation Harness (P4)

Suite automatizada para perfilar nuevos pesos (`.pt`, `.onnx`) y validar la tasa de falsos positivos en las reglas de negocio antes de liberar a producción.

## Estructura
- `eval/metrics/performance.py`: Cálculo de FPS, latencia media, p50, p95, p99.
- `eval/metrics/business_rules_metrics.py`: Matriz de confusión (TP, FP, TN, FN), Precision, Recall, FPR y veredicto.
- `eval/runners/benchmark_model.py`: Benchmark aislado de inferencia por modelo.
- `eval/runners/evaluate_pipeline.py`: Evaluación end-to-end con datasets y ground-truth.
- `eval/reports/`: Registro JSON de métricas por corrida (ignorado en git).

## Comandos de Uso

### 1. Benchmark Aislado de Modelo
```bash
python -m eval.runners.benchmark_model --weights yolov8n.pt --warmup 15 --iterations 100 --imgsz 640
```

### 2. Evaluación End-to-End contra Dataset de Video
```bash
python -m eval.runners.evaluate_pipeline --dataset eval/datasets/cellphone_confusion --manifest eval/datasets/manifests/confusion.json --max-frames 300
```

## Criterios de Aprobación (Pass / Fail)

Una evaluación resulta en **PASS** únicamente si se cumplen simultáneamente los 3 umbrales:
1. **Latencia p95:** $\le 65.0\text{ ms}$ (garantiza $> 15\text{ FPS}$ en edge).
2. **Falsos Positivos (FPR):** $\le 5.0\%$ en casos de confusión o fondos limpios.
3. **Sensibilidad (Recall):** $\ge 90.0\%$ en eventos verdaderos anotados en el manifest.

Si cualquiera de estos umbrales se transgrede, el veredicto es **FAIL** y se detallan las razones exactas en el log y reporte JSON.
