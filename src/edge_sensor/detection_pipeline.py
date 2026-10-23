"""
detection_pipeline.py — Pipeline de Inferencia en Cascada, Zoom y Tracking para IVOA.

Núcleo computacional de visión por computadora:
1. Carga optimizada de redes neuronales (integrado con HardwareOptimizer / TensorRT / ONNX).
2. Inferencia en cascada multimodelo:
   - model_base (YOLOv8 COCO): Humanos, celulares de perfil, alimentos/bebidas y periféricos.
   - model_custom (v5): Audífonos de diadema, celulares y alimentos.
   - model_screens (v7): Pantallas y páginas web no autorizadas.
   - model_grooming_cash (v8): Detección de arreglo personal.
   - model_v9 (v9): Billetes, llaves y situación atípica.
3. Inferencia multiescala en recortes de Zoom Óptico (manos, cabeza y torso).
4. Tracking multiobjeto (ByteTrack) y re-identificación espacial y cromática (HSV).
5. Extracción biométrica facial periódica mediante UniFace ArcFace (SCRFD + ONNX Runtime).
"""

from __future__ import annotations

import logging
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger("detection_pipeline")

# Intentar importar el Hardware Optimizer (P1)
try:
    from model_optimizer import get_optimized_model
    _HAS_OPTIMIZER = True
except ImportError:
    try:
        from src.edge_sensor.model_optimizer import get_optimized_model
        _HAS_OPTIMIZER = True
    except ImportError:
        _HAS_OPTIMIZER = False
        from ultralytics import YOLO


def extract_color_signature(frame: np.ndarray, bbox: np.ndarray | List[float]) -> Optional[np.ndarray]:
    """Extrae una huella digital matemática basada en los colores de la ropa (torso superior)."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    if pw < 10 or ph < 10:
        return None

    # Recorte del torso central para aislar el fondo y sombras
    ty1 = max(0, y1 + int(ph * 0.15))
    ty2 = min(h, y1 + int(ph * 0.60))
    tx1 = max(0, x1 + int(pw * 0.15))
    tx2 = min(w, x2 - int(pw * 0.15))

    crop = frame[ty1:ty2, tx1:tx2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


class DetectionPipeline:
    """Pipeline de visión por computadora: modelos en cascada, tracking ByteTrack y re-identificación."""

    CONF_MAP = {
        2: 0.05,  # Billetes
        4: 0.02,  # Comida
        5: 0.12,  # Celular
        6: 0.99,  # Llaves
        7: 0.08,  # Audífonos diadema
        8: 0.01,  # Páginas no autorizadas
    }

    def __init__(
        self,
        width: int,
        height: int,
        frame_skip: int = 5,
        models_dir: Optional[str | Path] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.FRAME_SKIP = frame_skip

        if models_dir is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.models_dir = base_dir / "models"
        else:
            self.models_dir = Path(models_dir)

        # 1. Carga de Modelos con Hardware Optimizer (o fallback YOLO directo)
        self._load_models()

        # 2. Inicialización de ByteTrack
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.55,
                lost_track_buffer=180,
                minimum_matching_threshold=0.25,
                frame_rate=30,
            )

        # 3. Estado interno de tracking e identidades
        self.id_map: Dict[int, int] = {}
        self.persistent_identities: Dict[int, Dict[str, Any]] = {}
        self.known_identities: Dict[str, str] = {"999": "persona_id_999"}

        # 4. Caché de último frame procesado para frame skipping
        self.last_raw_outputs: Dict[str, Any] = {}
        self.last_detections_base = sv.Detections.empty()
        self.last_detections_tracked = sv.Detections.empty()

    def _load_models(self) -> None:
        """Carga los 5 modelos de visión acelerados."""
        m_dir = self.models_dir

        def _get(name: str):
            if _HAS_OPTIMIZER:
                return get_optimized_model(name)
            pt_path = m_dir / name
            return YOLO(str(pt_path) if pt_path.exists() else name)

        logger.info("Inicializando pipeline de modelos neuronales...")
        self.model_base = _get("yolov8n.pt")
        self.model_custom = _get("yolov8_ivoa_v5.pt")
        self.model_screens = _get("yolov8_ivoa_v7.pt")

        v8_path = m_dir / "yolov8_ivoa_v8.pt"
        self.model_grooming_cash = _get("yolov8_ivoa_v8.pt") if v8_path.exists() else self.model_custom

        self.model_v9 = _get("yolov8_ivoa_v9.pt")
        logger.info("Modelos neuronales cargados y listos.")

    def run_inference(self, frame: np.ndarray) -> Dict[str, Any]:
        """Ejecuta la inferencia en cascada de los modelos y recortes de zoom."""
        # A. Modelo Base (COCO)
        res_base = self.model_base(frame, verbose=False, imgsz=640, conf=0.08)[0]
        dets_base_all = sv.Detections.from_ultralytics(res_base)

        # Filtrar solo personas reales (conf >= 0.55, altura >= 60)
        is_human = (dets_base_all.class_id == 0) & (dets_base_all.confidence >= 0.55)
        if np.any(is_human):
            valid_humans = []
            for i, (cid, conf, b) in enumerate(
                zip(dets_base_all.class_id, dets_base_all.confidence, dets_base_all.xyxy)
            ):
                if cid == 0 and conf >= 0.55 and (b[3] - b[1]) >= 60:
                    valid_humans.append(i)
            detections_base = dets_base_all[valid_humans] if valid_humans else sv.Detections.empty()
        else:
            detections_base = sv.Detections.empty()

        # Celulares en modelo base (COCO 67 -> remapeado a 5)
        phones_idx = []
        for i, (cid, conf, bbox) in enumerate(
            zip(dets_base_all.class_id, dets_base_all.confidence, dets_base_all.xyxy)
        ):
            if cid == 67:
                bw = bbox[2] - bbox[0]
                req = 0.04 if bw > self.width * 0.12 else 0.35
                if conf >= req:
                    phones_idx.append(i)
        phones_base = dets_base_all[phones_idx] if phones_idx else sv.Detections.empty()
        if len(phones_base) > 0:
            phones_base.class_id = np.array([5] * len(phones_base))

        # Periféricos (Mouse 64, Teclado 66, Laptop 63, Monitor 62)
        peripherals = dets_base_all[np.isin(dets_base_all.class_id, [62, 63, 64, 66])]

        # Alimentos en modelo base
        food_coco_ids = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
        food_mask = np.isin(dets_base_all.class_id, food_coco_ids) & (dets_base_all.confidence >= 0.18)
        food_base = dets_base_all[food_mask]
        if len(food_base) > 0:
            food_base.class_id = np.array([4] * len(food_base))

        # B. Modelo Custom v5 (Audífonos, Celular, Comida)
        res_custom = self.model_custom(frame, verbose=False, imgsz=640, conf=0.05)[0]
        dets_custom_raw = sv.Detections.from_ultralytics(res_custom)

        # C. Modelo Screens v7 (Pantallas y Páginas No Autorizadas)
        res_screens = self.model_screens(frame, verbose=False, imgsz=640, conf=0.05)[0]
        dets_screens_raw = sv.Detections.from_ultralytics(res_screens)

        # D. Modelo Grooming v8 (Arreglo Personal)
        res_v8 = self.model_grooming_cash(frame, verbose=False, imgsz=640, conf=0.05)[0]
        dets_v8_raw = sv.Detections.from_ultralytics(res_v8)

        # E. Modelo v9 (Billetes, Llaves y Situación Atípica)
        res_v9 = self.model_v9(frame, verbose=False, imgsz=640, conf=0.02)[0]
        dets_v9_raw = sv.Detections.from_ultralytics(res_v9)

        # F. Recortes de Zoom Óptico Multi-escala
        zoom_boxes, zoom_classes, zoom_confs = self._run_optical_zoom(frame, detections_base)
        if len(zoom_boxes) > 0:
            detections_zoom = sv.Detections(
                xyxy=np.array(zoom_boxes),
                class_id=np.array(zoom_classes),
                confidence=np.array(zoom_confs),
            )
        else:
            detections_zoom = sv.Detections.empty()

        return {
            "detections_base": detections_base,
            "phones_base": phones_base,
            "food_base": food_base,
            "peripherals": peripherals,
            "dets_custom_raw": dets_custom_raw,
            "dets_screens_raw": dets_screens_raw,
            "dets_v8_raw": dets_v8_raw,
            "dets_v9_raw": dets_v9_raw,
            "detections_zoom": detections_zoom,
        }

    def _run_optical_zoom(
        self, frame: np.ndarray, detections_base: sv.Detections
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        """Extrae recuadros focales de manos/torso/cabeza e infiere a alta resolución."""
        zoom_boxes: List[List[float]] = []
        zoom_classes: List[int] = []
        zoom_confs: List[float] = []

        if len(detections_base) == 0:
            return zoom_boxes, zoom_classes, zoom_confs

        for pbox in detections_base.xyxy:
            px1, py1, px2, py2 = map(int, pbox)
            pw, ph = px2 - px1, py2 - py1

            zx1 = max(0, px1 - int(pw * 0.15))
            zy1 = max(0, py1 - int(ph * 0.05))
            zx2 = min(self.width, px1 + int(pw * 0.85))
            zy2 = min(self.height, py1 + int(ph * 0.85))

            crop_zoom = frame[zy1:zy2, zx1:zx2]
            if crop_zoom.size == 0 or crop_zoom.shape[0] < 50 or crop_zoom.shape[1] < 50:
                continue

            # Inferencia Custom v5 sobre el zoom
            res_zoom = self.model_custom(crop_zoom, verbose=False, imgsz=640, conf=0.05)[0]
            if len(res_zoom.boxes) > 0:
                for zbox in res_zoom.boxes:
                    zcid = int(zbox.cls[0])
                    zconf = float(zbox.conf[0])
                    zb = zbox.xyxy[0].cpu().numpy()
                    gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]

                    if zcid == 7 and zconf >= 0.15 and (gx2 - gx1) < self.width * 0.40:
                        zoom_boxes.append([gx1, gy1, gx2, gy2])
                        zoom_classes.append(7)
                        zoom_confs.append(zconf)
                    elif zcid == 5 and zconf >= 0.25 and (gx2 - gx1) < self.width * 0.30:
                        zoom_boxes.append([gx1, gy1, gx2, gy2])
                        zoom_classes.append(5)
                        zoom_confs.append(zconf)
                    elif zcid == 4 and zconf >= 0.15:
                        zoom_boxes.append([gx1, gy1, gx2, gy2])
                        zoom_classes.append(4)
                        zoom_confs.append(zconf)

            # Inferencia Base COCO sobre el zoom (Celular y Alimentos)
            res_zoom_base = self.model_base(crop_zoom, verbose=False, imgsz=640, conf=0.08)[0]
            if len(res_zoom_base.boxes) > 0:
                for zbox in res_zoom_base.boxes:
                    zcid = int(zbox.cls[0])
                    zconf = float(zbox.conf[0])
                    zb = zbox.xyxy[0].cpu().numpy()
                    gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]

                    if zcid == 67 and zconf >= 0.25 and (gx2 - gx1) < self.width * 0.30:
                        zoom_boxes.append([gx1, gy1, gx2, gy2])
                        zoom_classes.append(5)
                        zoom_confs.append(zconf)
                    elif zcid in [39, 40, 41, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55] and zconf >= 0.02:
                        zoom_boxes.append([gx1, gy1, gx2, gy2])
                        zoom_classes.append(4)
                        zoom_confs.append(zconf)

        return zoom_boxes, zoom_classes, zoom_confs

    def update_tracking(
        self,
        frame: np.ndarray,
        detections_base: sv.Detections,
        frame_count: int,
        now_ts: Optional[float] = None,
    ) -> sv.Detections:
        """Actualiza ByteTrack, persistencia por firmas de color HSV y biometría ArcFace."""
        now_ts = now_ts or time.time()
        detections_tracked = self.tracker.update_with_detections(detections_base)

        if detections_tracked.tracker_id is None or len(detections_tracked.tracker_id) == 0:
            return detections_tracked

        new_tracker_ids = []
        active_mapped_ids: Set[int] = set()

        for tid, bbox in zip(detections_tracked.tracker_id, detections_tracked.xyxy):
            signature = extract_color_signature(frame, bbox)
            mapped_tid = self.id_map.get(tid, tid)

            cur_cx = (bbox[0] + bbox[2]) / 2.0
            cur_cy = (bbox[1] + bbox[3]) / 2.0
            cur_w = bbox[2] - bbox[0]

            # Si el ID no existe en memoria persistente, buscar correlación por posición o firma de ropa
            if mapped_tid not in self.persistent_identities:
                best_match_id = None
                best_match_score = -1.0

                for old_tid, old_data in self.persistent_identities.items():
                    if old_tid in active_mapped_ids:
                        continue
                    time_diff = now_ts - old_data["last_seen"]
                    if time_diff > 300.0:  # Expiración a los 5 minutos
                        continue

                    ocx, ocy = old_data.get("last_center", (cur_cx, cur_cy))
                    old_bbox = old_data.get("last_bbox", bbox)
                    old_w = old_bbox[2] - old_bbox[0]

                    dist = np.hypot(cur_cx - ocx, cur_cy - ocy)
                    dx = abs(cur_cx - ocx)

                    # Solapamiento horizontal en X (mismo escritorio)
                    x_inter = max(0, min(bbox[2], old_bbox[2]) - max(bbox[0], old_bbox[0]))
                    x_overlap = x_inter / float(max(1, min(cur_w, old_w)))

                    color_score = 0.0
                    if signature is not None and old_data.get("signature") is not None:
                        color_score = cv2.compareHist(
                            signature, old_data["signature"], cv2.HISTCMP_CORREL
                        )

                    is_same_workstation = (x_overlap >= 0.40) and (dx < 140) and (time_diff < 60.0)
                    is_nearby_continuation = (dist < 220) and (time_diff < 10.0)
                    is_color_match = color_score > 0.60

                    if is_same_workstation or is_nearby_continuation or is_color_match:
                        score = max(x_overlap, color_score, 1.0 - (dist / 500.0))
                        if score > best_match_score:
                            best_match_score = score
                            best_match_id = old_tid

                if best_match_id is not None:
                    self.id_map[tid] = best_match_id
                    mapped_tid = best_match_id
                else:
                    self.id_map[tid] = tid
                    mapped_tid = tid

            self.persistent_identities[mapped_tid] = {
                "signature": signature,
                "last_center": (cur_cx, cur_cy),
                "last_bbox": bbox,
                "last_seen": now_ts,
            }
            active_mapped_ids.add(mapped_tid)
            new_tracker_ids.append(mapped_tid)

            # Integración biométrica UniFace periódica (cada 15 frames para IDs desconocidos)
            if frame_count % 15 == 0:
                self._attempt_biometrics(frame, bbox, mapped_tid)

        detections_tracked.tracker_id = np.array(new_tracker_ids)
        return detections_tracked

    def _attempt_biometrics(self, frame: np.ndarray, bbox: np.ndarray, mapped_tid: int) -> None:
        """Intenta identificar o registrar el rostro mediante UniFace ArcFace en segundo plano."""
        str_mapped = str(mapped_tid)
        if str_mapped in self.known_identities:
            return

        try:
            from biometric_engine import find_face_and_embed, identify_person_in_db, register_new_face

            face_emb = find_face_and_embed(frame, bbox)
            if face_emb:
                db_id, db_name = identify_person_in_db(face_emb)
                if db_id:
                    self.known_identities[str_mapped] = db_name or f"persona_id_{db_id}"
                else:
                    nuevo_id = register_new_face(embedding=face_emb, nombre=f"persona_id_{mapped_tid}")
                    self.known_identities[str_mapped] = f"persona_id_{nuevo_id}"
        except Exception:
            pass

    def process(
        self,
        frame: np.ndarray,
        frame_count: int,
        now_ts: Optional[float] = None,
    ) -> Tuple[Dict[str, Any], sv.Detections, sv.Detections]:
        """
        Punto de entrada principal:
        Si frame_count % FRAME_SKIP == 0, ejecuta inferencia pesada y tracking.
        De lo contrario, reutiliza las detecciones previas manteniendo fluidez.
        """
        now_ts = now_ts or time.time()

        if frame_count % self.FRAME_SKIP == 0:
            raw_outputs = self.run_inference(frame)
            detections_base = raw_outputs["detections_base"]
            detections_tracked = self.update_tracking(frame, detections_base, frame_count, now_ts)

            self.last_raw_outputs = raw_outputs
            self.last_detections_base = detections_base
            self.last_detections_tracked = detections_tracked
        else:
            raw_outputs = self.last_raw_outputs
            detections_base = self.last_detections_base
            detections_tracked = self.last_detections_tracked

        return raw_outputs, detections_base, detections_tracked
