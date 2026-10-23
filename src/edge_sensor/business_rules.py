"""
business_rules.py — Reglas de Auditoría Operativa y Código de Vestimenta para IVOA.

Implementa la lógica analítica de negocio en sucursales físicas:
1. Código de Vestimenta: Segmentación cromática HSV (playera blanca/negra y pantalón negro).
2. Consumo de Alimentos/Bebidas: Validación geométrica ("sólo a la boca", 50% superior del cuerpo).
3. Pantallas No Autorizadas: Validación de escala y contención (< 35% de la pantalla).
4. Arreglo Personal: Ajuste geométrico y centrado de recuadro en rostro/torso superior.
5. Situación Atípica: Detección de persona de pie por ratio antropométrico vertical.
6. Desambiguación: Supresión de maquillaje cuando colisiona con alimentos.
7. Geofencing Espacial: Filtrado por polígono personalizado o división izquierda/derecha.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger("business_rules")


class BusinessRules:
    """Motor de evaluación de reglas de negocio operativas y auditoría visual."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    # =========================================================================
    # 1. CÓDIGO DE VESTIMENTA (UNIFORME)
    # =========================================================================
    def evaluate_uniform(
        self, frame: np.ndarray, persons_xyxy: np.ndarray
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        """
        Analiza el código de vestimenta de cada persona detectada mediante segmentación HSV.
        Reglamento: Playera blanca (o negra) y pantalón negro (sin mezclilla azul).
        Retorna (boxes, classes, confidences) de las infracciones (clase 9: uniforme_incompleto).
        """
        uniform_boxes: List[List[float]] = []
        uniform_classes: List[int] = []
        uniform_confs: List[float] = []

        if len(persons_xyxy) == 0:
            return uniform_boxes, uniform_classes, uniform_confs

        for pbox in persons_xyxy:
            px1, py1, px2, py2 = map(int, pbox)
            pw, ph = px2 - px1, py2 - py1

            # Descartar personas muy pequeñas o lejanas
            if ph < self.height * 0.20 or pw < 20:
                continue

            # 1. Recorte del Torso (Playera: 15% a 45% de altura, 30% a 70% de ancho central)
            ty1 = max(0, py1 + int(ph * 0.15))
            ty2 = min(self.height, py1 + int(ph * 0.45))
            tx1 = max(0, px1 + int(pw * 0.30))
            tx2 = min(self.width, px2 - int(pw * 0.30))
            t_crop = frame[ty1:ty2, tx1:tx2]

            # 2. Recorte de Piernas (Pantalón: 55% a 90% de altura, 25% a 75% de ancho)
            ly1 = max(0, py1 + int(ph * 0.55))
            ly2 = min(self.height, py1 + int(ph * 0.90))
            lx1 = max(0, px1 + int(pw * 0.25))
            lx2 = min(self.width, px2 - int(pw * 0.25))
            l_crop = frame[ly1:ly2, lx1:lx2]

            if t_crop.size > 0 and l_crop.size > 0:
                t_hsv = cv2.cvtColor(t_crop, cv2.COLOR_BGR2HSV)
                l_hsv = cv2.cvtColor(l_crop, cv2.COLOR_BGR2HSV)

                # Analizar el 50% central del torso para evitar bordes con la pared del fondo
                th, tw = t_crop.shape[:2]
                center_t_hsv = t_hsv[
                    int(th * 0.25) : int(th * 0.75), int(tw * 0.25) : int(tw * 0.75)
                ]

                if center_t_hsv.size > 0:
                    # Blanco: baja saturación (< 40) y alto brillo (> 180)
                    t_white = np.sum(
                        (center_t_hsv[:, :, 1] < 40) & (center_t_hsv[:, :, 2] > 180)
                    ) / float(center_t_hsv.shape[0] * center_t_hsv.shape[1])
                else:
                    t_white = 0.0

                # Negro en pantalón: brillo muy bajo (< 70)
                l_black = np.sum(l_hsv[:, :, 2] < 70) / float(l_crop.shape[0] * l_crop.shape[1])
                # Azul (mezclilla no autorizada): tono 90-130 con saturación > 35
                l_blue = np.sum(
                    (l_hsv[:, :, 0] >= 90)
                    & (l_hsv[:, :, 0] <= 130)
                    & (l_hsv[:, :, 1] > 35)
                ) / float(l_crop.shape[0] * l_crop.shape[1])

                has_white_shirt = t_white >= 0.55
                has_black_pants = (l_black >= 0.45) and (l_blue < 0.15)
                is_valid = has_white_shirt and has_black_pants

                if not is_valid:
                    uniform_boxes.append([px1, py1, px2, py2])
                    uniform_classes.append(9)  # uniforme_incompleto
                    uniform_confs.append(0.92)

        return uniform_boxes, uniform_classes, uniform_confs

    # =========================================================================
    # 2. VALIDACIÓN DE ALIMENTOS / BEBIDAS ("SÓLO A LA BOCA")
    # =========================================================================
    def filter_food_candidates(
        self,
        food_boxes: np.ndarray | List[List[float]],
        persons_xyxy: np.ndarray,
        peripherals_xyxy: Optional[np.ndarray] = None,
        max_size_ratio: float = 0.35,
    ) -> List[int]:
        """
        Filtra candidatos de comida/bebidas asegurando que estén próximos a la boca o manos
        y que no correspondan a periféricos (mouse, teclado, monitor) ni a la esquina de cámara.
        Retorna los índices de las cajas válidas.
        """
        valid_indices: List[int] = []
        if len(food_boxes) == 0:
            return valid_indices

        peripherals = peripherals_xyxy if peripherals_xyxy is not None else np.empty((0, 4))

        for idx, bbox in enumerate(food_boxes):
            bx1, by1, bx2, by2 = bbox
            bw, bh = bx2 - bx1, by2 - by1

            # Descartar si excede el tamaño máximo razonable
            if bw > self.width * max_size_ratio or bh > self.height * max_size_ratio:
                continue

            # Descartar si está en la esquina inferior izquierda del mouse/teclado
            if bx1 < 220 and by1 > self.height * 0.50:
                continue

            # Descartar si colisiona directamente con un periférico
            is_peripheral = False
            for pbox in peripherals:
                if max(bx1, pbox[0]) < min(bx2, pbox[2]) and max(by1, pbox[1]) < min(by2, pbox[3]):
                    is_peripheral = True
                    break
            if is_peripheral:
                continue

            # Regla "Sólo a la boca": El centro del objeto debe situarse en el 50% superior del cuerpo
            fcx = bx1 + (bw / 2.0)
            fcy = by1 + (bh / 2.0)
            is_at_mouth_or_torso = False

            for pb in persons_xyxy:
                head_y_limit = pb[1] + (pb[3] - pb[1]) * 0.50
                # Margen X expandido en 50px para manos extendidas hacia la cara
                if (pb[0] - 50) < fcx < (pb[2] + 50) and pb[1] < fcy < head_y_limit:
                    is_at_mouth_or_torso = True
                    break

            if is_at_mouth_or_torso:
                valid_indices.append(idx)

        return valid_indices

    # =========================================================================
    # 3. PANTALLAS Y PÁGINAS NO AUTORIZADAS
    # =========================================================================
    def filter_unauthorized_screens(
        self,
        raw_boxes: np.ndarray | List[List[float]],
        raw_classes: np.ndarray | List[int],
        raw_confs: np.ndarray | List[float],
        min_conf: float = 0.04,
        max_size_ratio: float = 0.35,
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        """
        Filtra candidatos de pantallas no autorizadas (clase 8 de v7).
        Evita que espaldas o paredes se clasifiquen erróneamente como pantallas gigantes.
        """
        valid_boxes: List[List[float]] = []
        valid_classes: List[int] = []
        valid_confs: List[float] = []

        for bbox, cid, conf in zip(raw_boxes, raw_classes, raw_confs):
            if cid == 8 and conf >= min_conf:
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                # Una ventana en monitor nunca supera el 35% de la cámara
                if bw < self.width * max_size_ratio and bh < self.height * max_size_ratio:
                    valid_boxes.append(bbox if isinstance(bbox, list) else bbox.tolist())
                    valid_classes.append(8)
                    valid_confs.append(float(conf))

        return valid_boxes, valid_classes, valid_confs

    # =========================================================================
    # 4. ARREGLO PERSONAL / MAQUILLAJE
    # =========================================================================
    def format_grooming_boxes(
        self,
        raw_boxes: np.ndarray | List[List[float]],
        raw_classes: np.ndarray | List[int],
        raw_confs: np.ndarray | List[float],
        min_conf: float = 0.25,
    ) -> Tuple[List[List[float]], List[int], List[float]]:
        """
        Ajusta y centra los recuadros de arreglo personal (v8 clase 0 -> sistema clase 10)
        hacia el rostro y torso superior para evitar superposición con la etiqueta del tracker.
        """
        boxes: List[List[float]] = []
        classes: List[int] = []
        confs: List[float] = []

        for bbox, cid, conf in zip(raw_boxes, raw_classes, raw_confs):
            if cid == 0 and conf >= min_conf:
                x1, y1, x2, y2 = bbox
                w, h = x2 - x1, y2 - y1

                # Recuadro centrado en el rostro
                nx1 = x1 + (w * 0.20)
                nx2 = x2 - (w * 0.20)
                ny1 = y1 + (h * 0.08)  # Separación visual respecto al techo del bbox
                ny2 = y1 + (h * 0.35)  # Altura del pecho

                boxes.append([nx1, ny1, nx2, ny2])
                classes.append(10)  # Remapeo a clase 10
                confs.append(float(conf))

        return boxes, classes, confs

    # =========================================================================
    # 5. SITUACIÓN ATÍPICA (POSTURA DE PIE / ANÓMALA)
    # =========================================================================
    def detect_standing_posture(
        self, persons_xyxy: np.ndarray
    ) -> Tuple[bool, List[List[float]]]:
        """
        Evalúa si alguna persona en el área de atención está completamente de pie
        o en movimiento atípico según relaciones antropométricas y posición en cámara.
        """
        standing_boxes: List[List[float]] = []
        has_standing = False

        if len(persons_xyxy) == 0:
            return has_standing, standing_boxes

        for pbox in persons_xyxy:
            px1, py1, px2, py2 = map(int, pbox)
            pw, ph = px2 - px1, py2 - py1

            # 1. Cabeza cerca del borde superior de la cámara (py1 < 10%)
            is_touching_ceiling = py1 < self.height * 0.10
            # 2. Ratio vertical estricto (ph / pw >= 1.65) y altura representativa (> 40% cámara)
            ratio = ph / max(1, pw)
            is_standing_ratio = (ratio >= 1.65) and (ph > self.height * 0.40)

            if is_touching_ceiling or is_standing_ratio:
                has_standing = True
                standing_boxes.append([float(px1), float(py1), float(px2), float(py2)])

        return has_standing, standing_boxes

    # =========================================================================
    # 6. DESAMBIGUACIÓN COMIDA VS ARREGLO PERSONAL
    # =========================================================================
    def disambiguate_food_vs_grooming(self, detections: sv.Detections) -> sv.Detections:
        """
        Si se detectó Comida/Bebida (clase 4) en mano o boca, suprime Arreglo Personal (clase 10)
        solapado para evitar falsas alarmas de maquillaje al estar comiendo.
        """
        if len(detections) == 0:
            return detections

        if 4 not in detections.class_id or 10 not in detections.class_id:
            return detections

        food_boxes = [b for cid, b in zip(detections.class_id, detections.xyxy) if cid == 4]
        keep_idx = []

        for idx, (cid, b) in enumerate(zip(detections.class_id, detections.xyxy)):
            if cid == 10:
                # Verificar solapamiento con cajas de comida
                overlap = any(
                    max(b[0], fb[0]) < min(b[2], fb[2]) and max(b[1], fb[1]) < min(b[3], fb[3])
                    for fb in food_boxes
                )
                if not overlap:
                    keep_idx.append(idx)
            else:
                keep_idx.append(idx)

        return detections[keep_idx] if len(keep_idx) > 0 else sv.Detections.empty()

    # =========================================================================
    # 7. GEOFENCING ESPACIAL (ZONAS ACTIVAS)
    # =========================================================================
    def apply_geofencing(
        self,
        detections: sv.Detections,
        custom_polygon_points: Optional[np.ndarray] = None,
        zona_izquierda: bool = True,
        zona_derecha: bool = True,
    ) -> sv.Detections:
        """
        Filtra detecciones fuera del área de interés geográfica.
        Las personas (clase 0) siempre se conservan en todo el encuadre para mantener el tracking.
        """
        if len(detections) == 0:
            return detections

        mid_x = self.width / 2.0
        keep_indices: List[int] = []

        if custom_polygon_points is not None and len(custom_polygon_points) >= 3:
            for idx, (cid, bbox) in enumerate(zip(detections.class_id, detections.xyxy)):
                if cid == 0:  # Humano siempre se conserva
                    keep_indices.append(idx)
                    continue

                pcx = (bbox[0] + bbox[2]) / 2.0
                pcy = (bbox[1] + bbox[3]) / 2.0
                is_inside = cv2.pointPolygonTest(
                    custom_polygon_points, (float(pcx), float(pcy)), False
                ) >= 0
                if is_inside:
                    keep_indices.append(idx)
        else:
            for idx, (cid, bbox) in enumerate(zip(detections.class_id, detections.xyxy)):
                if cid == 0:
                    keep_indices.append(idx)
                    continue

                pcx = (bbox[0] + bbox[2]) / 2.0
                if pcx < mid_x and not zona_izquierda:
                    continue
                if pcx >= mid_x and not zona_derecha:
                    continue
                keep_indices.append(idx)

        return detections[keep_indices] if len(keep_indices) > 0 else sv.Detections.empty()
