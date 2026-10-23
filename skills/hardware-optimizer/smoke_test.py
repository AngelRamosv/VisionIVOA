"""
Smoke test del Hardware Optimizer (P1).

Uso (desde la raíz del repo):
    python skills/hardware-optimizer/smoke_test.py
    python skills/hardware-optimizer/smoke_test.py --model yolov8n.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Asegurar que src/edge_sensor esté en el path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "edge_sensor"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("smoke_test")


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test — IVOA Hardware Optimizer")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Nombre o ruta del modelo .pt a probar",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--force", action="store_true", help="Forzar re-exportación")
    args = parser.parse_args()

    from model_optimizer import HardwareOptimizer, get_optimized_model

    # 1. Reporte de hardware
    opt = HardwareOptimizer.get_instance()
    report = opt.get_report()
    print("\n--- Hardware Report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("-----------------------\n")

    # 2. Carga / exportación
    t0 = time.perf_counter()
    try:
        model = get_optimized_model(
            args.model,
            imgsz=args.imgsz,
            half=True,
            force_rebuild=args.force,
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        logger.error(
            "Coloca el archivo .pt en ./models/ o pásalo con ruta absoluta."
        )
        return 1
    load_time = time.perf_counter() - t0
    logger.info("Modelo cargado en %.2f s", load_time)

    # 3. Inferencia sobre frame sintético
    frame = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    t1 = time.perf_counter()
    results = model(frame, verbose=False, imgsz=args.imgsz, conf=0.25)[0]
    infer_time = (time.perf_counter() - t1) * 1000
    logger.info("Inferencia sintética: %.1f ms", infer_time)

    # 4. Compatibilidad con supervision (si está instalado)
    try:
        import supervision as sv
        dets = sv.Detections.from_ultralytics(results)
        logger.info(
            "Compatible con supervision.Detections — %d detecciones",
            len(dets),
        )
    except ImportError:
        logger.info("supervision no instalado — se omite check de Detections")
    except Exception as e:
        logger.warning("from_ultralytics falló: %s", e)

    print("\n✅ Smoke test OK\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
