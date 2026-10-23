"""
eval/runners/benchmark_model.py — Benchmark aislado de inferencia y latencia p95 para modelos .pt / .onnx.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
import numpy as np

from eval.metrics.performance import PerformanceTracker
from src.edge_sensor.model_optimizer import get_optimized_model, HardwareOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_model")


def run_benchmark(
    weights_path: str,
    imgsz: int = 640,
    warmup: int = 15,
    iterations: int = 100,
    save_report: bool = True,
) -> dict:
    opt = HardwareOptimizer.get_instance()
    logger.info("Inicializando modelo: %s...", weights_path)
    model = get_optimized_model(weights_path, imgsz=imgsz)
    
    hw_report = opt.get_report()
    logger.info("Hardware Report: Backend=%s, Device=%s", hw_report["backend"], hw_report["device_name"])

    # Generar frame sintético
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    logger.info("Ejecutando warmup (%d iteraciones)...", warmup)
    for _ in range(warmup):
        _ = model(dummy_frame, verbose=False, imgsz=imgsz)

    logger.info("Ejecutando benchmark (%d iteraciones)...", iterations)
    tracker = PerformanceTracker()

    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = model(dummy_frame, verbose=False, imgsz=imgsz)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        tracker.record_ms(elapsed_ms)

    summary = tracker.compute_summary()
    result = {
        "model": str(weights_path),
        "imgsz": imgsz,
        "iterations": iterations,
        "hardware": hw_report,
        "metrics": summary,
        "verdict": "PASS" if summary["latency_p95_ms"] <= 65.0 else "FAIL",
    }

    logger.info("Resultados: FPS=%.1f | Media=%.2f ms | p95=%.2f ms | Veredicto=%s",
                summary["fps"], summary["latency_mean_ms"], summary["latency_p95_ms"], result["verdict"])

    if save_report:
        reports_dir = Path("eval/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = Path(weights_path).stem
        report_file = reports_dir / f"benchmark_{stem}_{timestamp}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        logger.info("Reporte guardado en: %s", report_file)

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark de Inferencia YOLO para IVOA")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Ruta o nombre del modelo")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamaño de entrada")
    parser.add_argument("--warmup", type=int, default=15, help="Rondas de calentamiento")
    parser.add_argument("--iterations", type=int, default=100, help="Iteraciones de prueba")
    args = parser.parse_args()

    run_benchmark(
        weights_path=args.weights,
        imgsz=args.imgsz,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
