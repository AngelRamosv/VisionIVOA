# Evaluation Harness — IVOA

Suite de evaluación y regresión para modelos y reglas de negocio.

## Objetivo

Antes de subir cualquier nuevo peso `.pt` o cambiar umbrales de detección, se debe pasar por este harness.

## Estructura recomendada

```
eval/
├── README.md
├── datasets/                 # Clips y anotaciones (no subir videos grandes a git)
│   ├── cellphone_confusion/
│   ├── uniform_violations/
│   ├── cash_handling/
│   └── privacy_cases/
├── metrics/
│   ├── detection.py          # mAP, precision, recall, FPS
│   └── business_rules.py     # Tasa de falsos positivos por regla
├── runners/
│   ├── run_detection_bench.py
│   └── run_full_pipeline.py
└── reports/                  # Salida de cada corrida (gitignore)
```

## Métricas mínimas a medir

- FPS promedio y p95 de latencia por frame
- Precisión / Recall por clase (celular, uniforme, billete, pantalla…)
- Tasa de falsos positivos en casos difíciles (celular vs mouse/cartera)
- Tiempo de re-identificación biométrica
- Uso de VRAM / RAM

## Uso futuro

```bash
python -m eval.runners.run_detection_bench --model yolov8_ivoa_v9.pt --dataset cellphone_confusion
```
