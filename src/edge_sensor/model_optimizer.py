"""
model_optimizer.py — Optimizador de Hardware y Aceleración Edge para IVOA (P1).

Detecta dinámicamente el silicio disponible (NVIDIA CUDA/TensorRT, Intel OpenVINO,
Apple Silicon CoreML o CPU ONNX Runtime), compila los modelos YOLOv8 y los almacena
en caché en memoria secundaria sin alterar la API de Ultralytics ni Supervision.
"""

from __future__ import annotations

import hashlib
import logging
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import torch
from ultralytics import YOLO

logger = logging.getLogger("model_optimizer")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [HardwareOptimizer] [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# Rutas base del proyecto
CURRENT_DIR = Path(__file__).resolve().parent
EDGE_SENSOR_DIR = CURRENT_DIR
SRC_DIR = CURRENT_DIR.parent
PROJECT_ROOT = SRC_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
CACHE_ROOT = MODELS_DIR / "optimized"


class HardwareOptimizer:
    """Gestor singleton de aceleración de hardware para modelos de visión IVOA."""

    _instance: Optional["HardwareOptimizer"] = None

    def __init__(self) -> None:
        self.system = platform.system()
        self.machine = platform.machine().lower()
        self.processor = platform.processor()
        
        self.cuda_available = torch.cuda.is_available()
        self.device_name = "CPU"
        self.vram_total_mb = 0
        
        if self.cuda_available:
            self.device_name = torch.cuda.get_device_name(0)
            try:
                self.vram_total_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
            except Exception:
                self.vram_total_mb = 0
        else:
            self.device_name = platform.processor() or platform.machine() or "CPU"

        # Detección del mejor backend
        self.backend, self.target_format, self.fp16_supported = self._detect_backend()
        self.cache_dir = CACHE_ROOT / self.backend
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "HardwareOptimizer":
        """Obtiene la instancia única del optimizador (Singleton)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _detect_backend(self) -> Tuple[str, str, bool]:
        """
        Determina el backend de compilación más rápido disponible:
        1. CUDA -> TensorRT FP16 (.engine) si tensorrt está presente; fallback a CUDA/ONNX
        2. Intel -> OpenVINO FP16 si la librería openvino está instalada
        3. Apple Silicon -> CoreML / ONNX
        4. Fallback estándar -> ONNX Runtime
        """
        # 1. NVIDIA CUDA
        if self.cuda_available:
            try:
                import tensorrt  # noqa: F401
                return "cuda", "engine", True
            except ImportError:
                # Si CUDA existe pero falta el SDK TensorRT, usamos ONNX acelerado o PyTorch CUDA
                logger.info("CUDA detectado pero 'tensorrt' no está instalado. Usando fallback ONNX.")
                return "cuda_onnx", "onnx", False

        # 2. Intel OpenVINO
        try:
            import openvino  # noqa: F401
            return "intel", "openvino", True
        except ImportError:
            pass

        # 3. Apple Silicon
        if self.system == "Darwin" and ("arm" in self.machine or "arm" in self.processor.lower()):
            return "mps", "coreml", False

        # 4. Fallback CPU / Genérico
        return "cpu", "onnx", False

    def get_report(self) -> dict[str, Any]:
        """Genera un reporte del entorno de cómputo y estado de aceleración."""
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "target_format": self.target_format,
            "fp16_supported": self.fp16_supported,
            "cache_dir": str(self.cache_dir),
            "cuda_available": self.cuda_available,
            "vram_total_mb": self.vram_total_mb,
            "os": f"{self.system} ({self.machine})",
            "pytorch_version": torch.__version__,
        }

    def _resolve_source_path(self, model_name_or_path: str | Path) -> Path:
        """Resuelve el archivo .pt origen a una ruta absoluta existente."""
        path = Path(model_name_or_path)
        if path.is_file():
            return path.resolve()

        # Buscar en ./models/
        candidate = MODELS_DIR / path.name
        if candidate.is_file():
            return candidate.resolve()

        # Buscar relativo al directorio actual
        candidate_rel = Path.cwd() / path
        if candidate_rel.is_file():
            return candidate_rel.resolve()

        raise FileNotFoundError(f"No se encontró el modelo origen: '{model_name_or_path}'. Buscado en: {path}, {candidate}")

    def _get_target_cached_path(self, source_pt: Path, target_format: str) -> Path:
        """Calcula la ruta esperada del artefacto compilado dentro de la caché."""
        stem = source_pt.stem
        if target_format == "engine":
            return self.cache_dir / f"{stem}.engine"
        elif target_format == "openvino":
            return self.cache_dir / f"{stem}_openvino_model"
        elif target_format == "onnx":
            return self.cache_dir / f"{stem}.onnx"
        elif target_format == "coreml":
            return self.cache_dir / f"{stem}.mlpackage"
        return source_pt

    def get_optimized_path(
        self,
        model_name_or_path: str | Path,
        imgsz: int = 640,
        half: bool = True,
        force_rebuild: bool = False,
    ) -> Tuple[Path, str]:
        """
        Retorna la ruta al modelo compilado en caché.
        Si no existe o el .pt es más reciente, compila de forma transparente.
        """
        source_pt = self._resolve_source_path(model_name_or_path)
        target_path = self._get_target_cached_path(source_pt, self.target_format)

        # 1. Verificar si la caché es válida
        is_cached = False
        if target_path.exists() and not force_rebuild:
            try:
                # Comprobar mtime: si el .pt es más nuevo, se invalida
                if target_path.stat().st_mtime >= source_pt.stat().st_mtime:
                    is_cached = True
            except Exception:
                is_cached = False

        if is_cached:
            logger.info("Usando modelo compilado en caché: %s", target_path.name)
            return target_path, self.target_format

        # 2. Compilar / Exportar
        logger.info("Compilando %s -> %s (half=%s, imgsz=%d)...", source_pt.name, self.target_format, half and self.fp16_supported, imgsz)
        t0 = time.perf_counter()

        try:
            pt_model = YOLO(str(source_pt))
            export_half = half and self.fp16_supported
            
            # Exportar mediante la API oficial de Ultralytics
            exported_output = pt_model.export(
                format=self.target_format,
                imgsz=imgsz,
                half=export_half,
                verbose=False,
            )
            
            elapsed = time.perf_counter() - t0
            logger.info("Exportación completada en %.2f s: %s", elapsed, exported_output)

            # Mover el artefacto generado a nuestra carpeta estructurada de caché si no quedó allí
            exported_path = Path(exported_output)
            if exported_path.resolve() != target_path.resolve():
                if exported_path.is_dir():
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.move(str(exported_path), str(target_path))
                else:
                    shutil.move(str(exported_path), str(target_path))

            return target_path, self.target_format

        except Exception as err:
            logger.warning(
                "Falló la compilación a '%s': %s. Aplicando fallback controlado al modelo base .pt.",
                self.target_format,
                err,
            )
            return source_pt, "pytorch"

    def load_model(
        self,
        model_name_or_path: str | Path,
        imgsz: int = 640,
        half: bool = True,
        force_rebuild: bool = False,
    ) -> YOLO:
        """
        Carga el modelo optimizado utilizando la clase nativa YOLO.
        Ultralytics soporta de forma transparente .engine, .onnx, _openvino_model y .pt
        manteniendo idénticos los objetos de detección para supervision.Detections.
        """
        model_path, backend_used = self.get_optimized_path(
            model_name_or_path, imgsz=imgsz, half=half, force_rebuild=force_rebuild
        )
        
        logger.info("Cargando modelo '%s' (backend: %s)...", model_path.name, backend_used)
        model = YOLO(str(model_path), task="detect")
        return model


def get_optimized_model(
    model_name_or_path: str | Path,
    imgsz: int = 640,
    half: bool = True,
    force_rebuild: bool = False,
) -> YOLO:
    """Punto de entrada funcional estándar para IVOA."""
    optimizer = HardwareOptimizer.get_instance()
    return optimizer.load_model(
        model_name_or_path,
        imgsz=imgsz,
        half=half,
        force_rebuild=force_rebuild,
    )
