"""
bill_validator.py — Validador Especialista de Efectivo y Billetes para IVOA.

Implementa la validación de billetes en múltiples capas para entornos de caja y mostrador:
1. Capa Neuronal: Filtrado de candidatos de alta sensibilidad (YOLOv8 v9 clase 1).
2. Capa Espacial:
   - Restricción de escala (< 40% de la pantalla para evitar falsos positivos gigantes).
   - Filtro Anti-Monitor: Supresión de reflejos y alucinaciones sobre pantallas.
   - Filtro de Manipulación: Verificación de proximidad con manos/cuerpo de personas (COCO clase 0).
3. Capa de Desambiguación: Prioridad absoluta de Efectivo sobre Comida (clase 4) y Maquillaje (clase 10).
4. Capa OCR (Opcional/Asíncrona): Validación de denominaciones numéricas mexicanas (20, 50, 100, 200, 500, 1000).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger("bill_validator")


class BillValidator:
    """Validador modular de efectivo con supresión de falsos positivos en pared, monitores y escritorios."""

    VALID_DENOMINATIONS: Set[str] = {"20", "50", "100", "200", "500", "1000"}

    def __init__(
        self,
        width: int,
        height: int,
        min_conf: float = 0.06,
        max_size_ratio: float = 0.40,
        monitor_padding: int = 20,
        person_reach_margin: int = 80,
        enable_ocr: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.min_conf = min_conf
        self.max_size_ratio = max_size_ratio
        self.monitor_padding = monitor_padding
        self.person_reach_margin = person_reach_margin

        # Estado y caché OCR
        self.enable_ocr = enable_ocr
        self._ocr_reader: Any = None
        self._ocr_cache: Dict[str, Tuple[bool, Optional[str]]] = {}

        if self.enable_ocr:
            self._start_async_ocr_loader()

    def _start_async_ocr_loader(self) -> None:
        """Carga EasyOCR en segundo plano para evitar bloqueos en el arranque del sensor."""
        def _loader():
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                logger.info("Motor EasyOCR cargado para validación de billetes.")
            except Exception as e:
                logger.debug("EasyOCR no disponible para validación de billetes: %s", e)
                self._ocr_reader = None

        threading.Thread(target=_loader, daemon=True).start()

    def is_candidate_valid(
        self,
        bbox: np.ndarray | List[float],
        conf: float,
        monitors_xyxy: np.ndarray,
        persons_xyxy: np.ndarray,
    ) -> bool:
        """
        Aplica los filtros geométricos y contextuales sobre una caja candidata de billete.
        Retorna True si pasa todas las reglas anti-alucinación.
        """
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1

        # 1. Filtro de tamaño: descarta alucinaciones gigantes de pared o escritorio
        if bw > self.width * self.max_size_ratio or bh > self.height * self.max_size_ratio:
            return False

        # 2. Filtro de confianza base calibrada
        if conf < self.min_conf:
            return False

        # 3. Filtro Anti-Monitor: Si el centro está dentro de un monitor, es alucinación de pantalla
        bcx = x1 + (bw / 2.0)
        bcy = y1 + (bh / 2.0)

        if len(monitors_xyxy) > 0:
            for mb in monitors_xyxy:
                mx1 = mb[0] - self.monitor_padding
                my1 = mb[1] - self.monitor_padding
                mx2 = mb[2] + self.monitor_padding
                my2 = mb[3] + self.monitor_padding
                if mx1 < bcx < mx2 and my1 < bcy < my2:
                    return False

        # 4. Filtro de Manipulación: El billete debe estar sostenido o próximo a una persona
        if len(persons_xyxy) > 0:
            is_held = False
            for pb in persons_xyxy:
                px1 = pb[0] - self.person_reach_margin
                py1 = pb[1] - self.person_reach_margin
                px2 = pb[2] + self.person_reach_margin
                py2 = pb[3] + self.person_reach_margin

                # Comprobación de solapamiento en 2D
                if max(x1, px1) < min(x2, px2) and max(y1, py1) < min(y2, py2):
                    is_held = True
                    break

            if not is_held:
                return False
        else:
            # Si no hay ninguna persona en escena, un billete no puede estar siendo manipulado
            return False

        return True

    def validate_v9_candidates(
        self,
        detections_v9_raw: sv.Detections,
        monitors_xyxy: np.ndarray,
        persons_xyxy: np.ndarray,
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        """
        Filtra las detecciones crudas del modelo v9 (clase 1: billetes).
        Retorna (valid_boxes, valid_classes, valid_confs) con billetes remapeados a clase 2.
        """
        valid_boxes: List[List[float]] = []
        valid_classes: List[int] = []
        valid_confs: List[float] = []

        if len(detections_v9_raw) == 0:
            return valid_boxes, valid_classes, valid_confs

        for cid, conf, bbox in zip(
            detections_v9_raw.class_id,
            detections_v9_raw.confidence,
            detections_v9_raw.xyxy,
        ):
            if cid == 1:  # Clase 1 en v9 corresponde a billetes
                if self.is_candidate_valid(bbox, conf, monitors_xyxy, persons_xyxy):
                    valid_boxes.append(bbox.tolist() if isinstance(bbox, np.ndarray) else bbox)
                    valid_classes.append(2)  # Remapeo a clase interna del sistema (2: billetes)
                    valid_confs.append(float(conf))

        return valid_boxes, valid_classes, valid_confs

    def disambiguate_detections(self, detections: sv.Detections) -> sv.Detections:
        """
        Aplica desambiguación estricta de prioridades en caja:
        - Si Billetes (2) colisionan con Comida (4), se preserva el billete y se suprime la comida.
        - Si Billetes (2) colisionan con Maquillaje/Arreglo (10), se suprime el maquillaje.
        """
        if len(detections) == 0:
            return detections

        has_bills = 2 in detections.class_id
        if not has_bills:
            return detections

        bill_boxes = [b for cid, b in zip(detections.class_id, detections.xyxy) if cid == 2]

        keep_idx = []
        for idx, (cid, b) in enumerate(zip(detections.class_id, detections.xyxy)):
            if cid == 2:
                keep_idx.append(idx)
            elif cid in (4, 10):
                # Comprobar si colisiona con algún billete validado
                collides_with_bill = False
                for bb in bill_boxes:
                    if max(b[0], bb[0]) < min(b[2], bb[2]) and max(b[1], bb[1]) < min(b[3], bb[3]):
                        collides_with_bill = True
                        break
                if not collides_with_bill:
                    keep_idx.append(idx)
            else:
                keep_idx.append(idx)

        return detections[keep_idx] if len(keep_idx) > 0 else sv.Detections.empty()

    def verify_denomination(
        self, frame: np.ndarray, bbox: np.ndarray | List[float]
    ) -> Tuple[bool, Optional[str]]:
        """
        (Opcional) Lee la denominación numérica del billete mediante OCR de alta velocidad.
        Retorna (is_verified, denomination_str).
        """
        if self._ocr_reader is None:
            return True, None  # Si OCR no está activo, la validación geométrica es suficiente

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[max(0, y1):min(self.height, y2), max(0, x1):min(self.width, x2)]
        if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return False, None

        # Redimensionar para mejorar contraste de dígitos
        h, w = crop.shape[:2]
        resized = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        try:
            texts = self._ocr_reader.readtext(gray, detail=0)
            for text in texts:
                digits = re.findall(r"\b(20|50|100|200|500|1000)\b", text)
                if digits:
                    return True, digits[0]
        except Exception:
            pass

        return True, None
