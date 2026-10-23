"""
eval/runners/evaluate_pipeline.py — Evaluación End-to-End del pipeline y reglas de negocio contra datasets/clips.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src" / "edge_sensor"))
sys.path.insert(0, str(_ROOT))

from eval.metrics.performance import PerformanceTracker
from eval.metrics.business_rules_metrics import BenchmarkEvaluator
from ivoa_engine import IVOAVisionEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_pipeline")


def evaluate_dataset(
    dataset_dir: str,
    manifest_file: str | None = None,
    max_frames: int = 300,
    save_report: bool = True,
) -> Dict[str, Any]:
    d_path = Path(dataset_dir)
    if not d_path.exists():
        logger.warning("El directorio de dataset no existe: %s. Creando estructura de prueba...", d_path)
        d_path.mkdir(parents=True, exist_ok=True)

    manifest_data: Dict[str, Any] = {}
    if manifest_file and Path(manifest_file).exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    # Buscar clips de video (.mp4, .avi)
    video_files = list(d_path.glob("*.mp4")) + list(d_path.glob("*.avi"))

    perf_tracker = PerformanceTracker()
    evaluator = BenchmarkEvaluator()

    # Si no hay videos, correr una evaluación sintética rápida para verificar pipeline
    if not video_files:
        logger.info("No se encontraron archivos de video en %s. Ejecutando suite sintética de control...", d_path)
        engine = IVOAVisionEngine(width=1280, height=720, frame_skip=5)
        for frame_idx in range(50):
            fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            t0 = time.perf_counter()
            annotated, alerts, sessions = engine.process_frame(fake_frame, frame_idx)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            perf_tracker.record_ms(elapsed_ms)

            # En un frame negro/limpio, no debería haber alertas (Control Negativo FPR = 0%)
            has_celular = any("celular" in a[1].lower() for a in alerts)
            has_billete = any("dinero" in a[1].lower() or "efectivo" in a[1].lower() for a in alerts)
            evaluator.record_prediction("celular", predicted=has_celular, ground_truth=False)
            evaluator.record_prediction("billetes", predicted=has_billete, ground_truth=False)
    else:
        for vid in video_files:
            logger.info("Evaluando video: %s", vid.name)
            cap = cv2.VideoCapture(str(vid))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
            engine = IVOAVisionEngine(width=w, height=h, frame_skip=5)

            frame_idx = 0
            vid_manifest = manifest_data.get(vid.name, {})
            expected_rules = vid_manifest.get("expected_alerts", [])

            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                t0 = time.perf_counter()
                annotated, alerts, sessions = engine.process_frame(frame, frame_idx)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                perf_tracker.record_ms(elapsed_ms)

                active_tags = set()
                for _, msg, _ in alerts:
                    msg_low = msg.lower()
                    if "celular" in msg_low: active_tags.add("celular")
                    if "dinero" in msg_low or "billete" in msg_low: active_tags.add("billetes")
                    if "uniforme" in msg_low: active_tags.add("uniforme")
                    if "alimento" in msg_low or "comida" in msg_low: active_tags.add("alimentos")

                for rule_name in ["celular", "billetes", "uniforme", "alimentos"]:
                    is_pred = rule_name in active_tags
                    is_gt = rule_name in expected_rules
                    evaluator.record_prediction(rule_name, predicted=is_pred, ground_truth=is_gt)

                frame_idx += 1
            cap.release()

    perf_summary = perf_tracker.compute_summary()
    verdict_report = evaluator.evaluate_thresholds(
        max_p95_ms=65.0,
        max_fpr=0.05,
        min_recall=0.90,
        p95_ms=perf_summary["latency_p95_ms"],
    )

    final_report = {
        "dataset": str(dataset_dir),
        "performance": perf_summary,
        "rules_evaluation": verdict_report,
        "overall_verdict": verdict_report["verdict"],
    }

    logger.info("==================================================")
    logger.info("VEREDICTO GLOBAL: %s", final_report["overall_verdict"])
    logger.info("FPS: %.1f | Latencia p95: %.2f ms", perf_summary["fps"], perf_summary["latency_p95_ms"])
    if verdict_report["failures"]:
        logger.warning("Fallas detectadas:\n - " + "\n - ".join(verdict_report["failures"]))
    logger.info("==================================================")

    if save_report:
        reports_dir = Path("eval/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"pipeline_eval_{timestamp}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(final_report, f, indent=2)
        logger.info("Reporte guardado en: %s", report_file)

    return final_report


def main():
    parser = argparse.ArgumentParser(description="Evaluación End-to-End de Pipeline IVOA")
    parser.add_argument("--dataset", type=str, default="eval/datasets/cellphone_confusion", help="Directorio con videos")
    parser.add_argument("--manifest", type=str, default=None, help="Archivo JSON con ground-truth")
    parser.add_argument("--max-frames", type=int, default=300, help="Límite de frames por clip")
    args = parser.parse_args()

    evaluate_dataset(dataset_dir=args.dataset, manifest_file=args.manifest, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
