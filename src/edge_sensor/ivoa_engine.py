"""
ivoa_engine.py — Orquestador Ligero de Visión por Computadora para IVOA.

Coordina de manera desacoplada los subsistemas especializados:
1. detection_pipeline.py: Carga de modelos neuronales YOLO, inferencia base, zoom óptico, ByteTrack y biometría UniFace.
2. bill_validator.py: Filtrado especialista anti-alucinaciones en conteo de efectivo y validación espacial de billetes.
3. business_rules.py: Auditoría cromática de uniforme HSV, comida a la boca, pantallas, posturas atípicas y geofencing.
4. queue_analyzer.py: Análisis de colas (PolygonZone), cruces (LineZone), tailgating, sesiones de cliente y tiempos de servicio.
5. alert_manager.py: Persistencia temporal de incidencias, debouncing, cooldown, culpabilidad y sincronización de notify_config.json.
6. visual_annotator.py: Renderizado gráfico Supervision, trazado, [TICKET EMITIDO], banner de apertura tardía y máscaras.

Mantiene compatibilidad 100% con la interfaz pública requerida por app_api.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import supervision as sv

from alert_manager import AlertManager
from bill_validator import BillValidator
from business_rules import BusinessRules
from detection_pipeline import DetectionPipeline
from model_optimizer import HardwareOptimizer
from queue_analyzer import QueueAnalyzer
from visual_annotator import VisualAnnotator

logger = logging.getLogger("ivoa_engine")


def extract_color_signature(frame: np.ndarray, bbox: List[float] | np.ndarray) -> Optional[np.ndarray]:
    """Extrae una huella digital matemática basada en los colores de la ropa (torso superior)."""
    import cv2
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    if pw < 10 or ph < 10:
        return None
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


class IVOAVisionEngine:
    """Orquestador delgado de visión artificial e incidencias operativas para IVOA."""

    def __init__(
        self,
        width: int,
        height: int,
        frame_skip: int = 5,
        config_path: Optional[str | Path] = None,
        models_dir: Optional[str | Path] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.FRAME_SKIP = frame_skip

        # 1. Reporte de diagnóstico del Hardware Optimizer al arranque
        hw_report = HardwareOptimizer.get_instance().get_report()
        logger.info(
            ">> [IVOA Hardware] Backend: %s | Dispositivo: %s | FP16: %s",
            hw_report["backend"].upper(),
            hw_report["device_name"],
            hw_report["fp16_supported"],
        )

        # 2. Inicialización de Subsistemas Especializados Desacoplados
        self.pipeline = DetectionPipeline(width=width, height=height, frame_skip=frame_skip, models_dir=models_dir)
        self.bill_validator = BillValidator(width=width, height=height)
        self.business_rules = BusinessRules(width=width, height=height)
        self.queue_analyzer = QueueAnalyzer(width=width, height=height)
        self.alert_manager = AlertManager(width=width, height=height, config_path=config_path)
        self.visual_annotator = VisualAnnotator(width=width, height=height)

    # -------------------------------------------------------------------------
    # Propiedades Públicas Delegadas para Compatibilidad Total con app_api.py
    # -------------------------------------------------------------------------
    @property
    def fila_count(self) -> int:
        return self.queue_analyzer.fila_count

    @property
    def modulo_count(self) -> int:
        return self.queue_analyzer.modulo_count

    @property
    def in_count(self) -> int:
        return self.queue_analyzer.in_count

    @property
    def out_count(self) -> int:
        return self.queue_analyzer.out_count

    @property
    def alerta_saturacion(self) -> bool:
        return "saturacion" in self.queue_analyzer.triggered_alerts

    @property
    def id_map(self) -> Dict[int, int]:
        return self.pipeline.id_map

    @property
    def known_identities(self) -> Dict[str, str]:
        return self.pipeline.known_identities

    @property
    def alert_config(self) -> Dict[str, Any]:
        return self.alert_manager.alert_config

    # Delegación de modelos para tests o inspección externa
    @property
    def model_base(self):
        return self.pipeline.model_base

    @property
    def model_custom(self):
        return self.pipeline.model_custom

    @property
    def model_screens(self):
        return self.pipeline.model_screens

    @property
    def model_grooming_cash(self):
        return self.pipeline.model_grooming_cash

    @property
    def model_v9(self):
        return self.pipeline.model_v9

    # -------------------------------------------------------------------------
    # Flujo de Procesamiento Principal de Fotogramas (7 Pasos)
    # -------------------------------------------------------------------------
    def process_frame(
        self, frame: np.ndarray, frame_count: int
    ) -> Tuple[np.ndarray, List[Tuple[str, str, str]], List[Tuple[int, Dict[str, Any]]]]:
        """
        Orquesta el ciclo de vida por fotograma siguiendo 7 pasos:
        1. Sincronización de configuración dinámica.
        2. Inferencia neuronal en cascada y tracking de personas.
        3. Validación especialista anti-alucinación de billetes.
        4. Aplicación de reglas de negocio, auditoría de uniforme y geofencing.
        5. Análisis de colas, aforo, tailgating y tiempos de servicio.
        6. Evaluación de debouncing, enfriamiento y generación de incidencias.
        7. Renderizado gráfico y anotación visual del fotograma resultante.
        """
        now_ts = time.time()

        # Paso 1: Sincronización de Configuración
        self.alert_manager.sync_config(frame_count)

        # Paso 2: Inferencia en Cascada & Tracking
        raw_outputs, detections_base, detections_tracked = self.pipeline.process(
            frame, frame_count, now_ts=now_ts
        )

        # Extracción de recuadros de personas y monitores para validaciones relacionales
        persons_xyxy = (
            detections_base.xyxy[detections_base.class_id == 0]
            if len(detections_base) > 0
            else np.empty((0, 4))
        )
        monitors_xyxy = raw_outputs.get("monitors_xyxy", np.empty((0, 4)))
        peripherals_xyxy = raw_outputs.get("peripherals_xyxy", np.empty((0, 4)))

        # Paso 3: Validación Especialista de Billetes
        dets_v9_raw = raw_outputs.get("detections_v9_raw", sv.Detections.empty())
        v_boxes, v_classes, v_confs = self.bill_validator.validate_v9_candidates(
            dets_v9_raw, monitors_xyxy=monitors_xyxy, persons_xyxy=persons_xyxy
        )

        # Paso 4: Reglas Operativas y Fusión
        boxes = list(raw_outputs.get("boxes", [])) + v_boxes
        classes = list(raw_outputs.get("classes", [])) + v_classes
        confs = list(raw_outputs.get("confs", [])) + v_confs

        # 4.1 Auditoría cromática de uniforme
        u_boxes, u_classes, u_confs = self.business_rules.evaluate_uniform(frame, persons_xyxy)
        boxes.extend(u_boxes)
        classes.extend(u_classes)
        confs.extend(u_confs)

        # 4.2 Filtrado de pantallas no autorizadas
        scr_boxes, scr_classes, scr_confs = self.business_rules.filter_unauthorized_screens(
            raw_boxes=raw_outputs.get("scr_boxes_raw", []),
            raw_classes=raw_outputs.get("scr_classes_raw", []),
            raw_confs=raw_outputs.get("scr_confs_raw", []),
        )
        boxes.extend(scr_boxes)
        classes.extend(scr_classes)
        confs.extend(scr_confs)

        # 4.3 Formateo y remapeo de arreglo personal
        g_boxes, g_classes, g_confs = self.business_rules.format_grooming_boxes(
            raw_boxes=raw_outputs.get("v8_boxes_raw", []),
            raw_classes=raw_outputs.get("v8_classes_raw", []),
            raw_confs=raw_outputs.get("v8_confs_raw", []),
        )
        boxes.extend(g_boxes)
        classes.extend(g_classes)
        confs.extend(g_confs)

        # 4.4 Filtrado espacial de comida ("sólo a la boca")
        food_indices = [i for i, c in enumerate(classes) if c == 4]
        if food_indices:
            food_candidates = [boxes[i] for i in food_indices]
            valid_sub_idx = set(
                self.business_rules.filter_food_candidates(
                    food_candidates, persons_xyxy, peripherals_xyxy=peripherals_xyxy
                )
            )
            # Descartar comida fuera de la boca/torso
            keep_all = [
                i for i in range(len(boxes))
                if i not in food_indices or (food_indices.index(i) in valid_sub_idx)
            ]
            boxes = [boxes[i] for i in keep_all]
            classes = [classes[i] for i in keep_all]
            confs = [confs[i] for i in keep_all]

        # 4.5 Detección de situación atípica de pie
        has_standing, standing_boxes = self.business_rules.detect_standing_posture(persons_xyxy)
        for sbox in standing_boxes:
            boxes.append(sbox)
            classes.append(11)  # situacion_atipica
            confs.append(0.85)

        # Construcción de objeto Detections unificado
        if len(boxes) > 0:
            detections = sv.Detections(
                xyxy=np.array(boxes, dtype=np.float32),
                class_id=np.array(classes, dtype=int),
                confidence=np.array(confs, dtype=np.float32),
            )
        else:
            detections = sv.Detections.empty()

        # 4.6 Desambiguación de prioridades (Billetes > Comida > Arreglo personal)
        detections = self.bill_validator.disambiguate_detections(detections)
        detections = self.business_rules.disambiguate_food_vs_grooming(detections)

        # 4.7 Geofencing espacial activo
        detections = self.business_rules.apply_geofencing(
            detections,
            custom_polygon_points=self.alert_manager.custom_polygon_points,
            zona_izquierda=self.alert_manager.alert_config.get("zona_izquierda", True),
            zona_derecha=self.alert_manager.alert_config.get("zona_derecha", True),
        )

        # Paso 5: Análisis de Colas, Tránsito y Sesiones
        queue_alerts, sessions_to_update, tailgating_boxes = self.queue_analyzer.update(
            frame, detections_tracked, now_ts=now_ts
        )

        # Incorporar cajas visuales de tailgating (seguridad) si las hay
        if tailgating_boxes:
            tb_arr = np.array(tailgating_boxes, dtype=np.float32)
            tc_arr = np.array([15] * len(tailgating_boxes), dtype=int)
            tconf_arr = np.array([0.99] * len(tailgating_boxes), dtype=np.float32)
            if len(detections) > 0:
                detections = sv.Detections(
                    xyxy=np.vstack([detections.xyxy, tb_arr]),
                    class_id=np.concatenate([detections.class_id, tc_arr]),
                    confidence=np.concatenate([detections.confidence, tconf_arr]),
                )
            else:
                detections = sv.Detections(xyxy=tb_arr, class_id=tc_arr, confidence=tconf_arr)

        # Paso 6: Evaluación de Incidencias & Debounce
        extra_flags = {"atipica": has_standing}
        business_alerts = self.alert_manager.evaluate_detections(
            detections=detections,
            detections_tracked=detections_tracked,
            id_map=self.pipeline.id_map,
            known_identities=self.pipeline.known_identities,
            now_ts=now_ts,
            extra_active_flags=extra_flags,
        )

        # Fusión y filtrado de alertas habilitadas según notify_config.json
        all_candidate_alerts = business_alerts + queue_alerts
        final_alerts = self.alert_manager.filter_active_alerts(all_candidate_alerts)

        # Paso 7: Renderizado Visual Desacoplado
        annotated_frame = self.visual_annotator.render(
            frame=frame,
            detections=detections,
            detections_tracked=detections_tracked,
            id_map=self.pipeline.id_map,
            known_identities=self.pipeline.known_identities,
            custom_polygon_points=self.alert_manager.custom_polygon_points,
            last_alert_times=self.alert_manager.last_alert_times,
            detection_start_times=self.alert_manager.detection_start_times,
            debounce_seconds=self.alert_manager.DEBOUNCE_SECONDS,
            late_opening_trigger_time=self.queue_analyzer.late_opening_trigger_time,
            late_opening_time_str=self.queue_analyzer.late_opening_time_str,
            show_late_opening=self.alert_manager.is_alert_enabled("late_opening"),
            enable_privacy_blur=bool(self.alert_manager.alert_config.get("enable_privacy_blur", False)),
            privacy_blur_mode=str(self.alert_manager.alert_config.get("privacy_blur_mode", "non_employees")),
            now_ts=now_ts,
        )

        return annotated_frame, final_alerts, sessions_to_update
