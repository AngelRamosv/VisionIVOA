"""
visual_annotator.py — Renderizado Gráfico y Anotación Visual para IVOA.

Aísla la capa de presentación visual sobre fotogramas NumPy:
1. Anotadores Supervision (BoxAnnotator, LabelAnnotator, TraceAnnotator) con paleta corporativa calibrada.
2. Formateo de Etiquetas Dinámicas:
   - Indicador [TICKET EMITIDO] durante la emisión de alertas.
   - Cronómetro de Debounce activo (ej. "celular (0.4s / 0.5s)").
   - Etiquetas de personas trackeadas asociadas a identidades biométricas de UniFace ArcFace.
3. Geofencing Gráfico: Oscurecimiento con máscara alpha (cv2.addWeighted) de zonas inactivas y polígono anti-aliased.
4. Banner Superior de Apertura Tardía con recuadro de advertencia rojo.
5. Filtrado visual de recuadros temporales (ej. supresión de uniforme durante enfriamiento post-ticket).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger("visual_annotator")


class VisualAnnotator:
    """Motor desacoplado de renderizado visual y anotaciones sobre el video."""

    CLASS_NAMES: Dict[int, str] = {
        0: "pantalon_negro",
        1: "playera_blanca",
        2: "billetes",
        3: "zapatos",
        4: "comida",
        5: "celular",
        6: "llaves",
        7: "audifonos_diadema",
        8: "pagina_no_autorizada",
        9: "uniforme_incompleto",
        10: "arreglo_personal",
        11: "situacion_atipica",
        13: "audifonos_no_detectados",
        14: "uniforme_completo",
        15: "seguridad",  # Tailgating
    }

    CLASS_TO_TAG: Dict[int, Optional[str]] = {
        5: "celular",
        7: "audifonos",
        2: "dinero",
        6: "llaves",
        4: "alimentos",
        8: "paginas",
        9: "uniforme",
        10: "arreglo",
        11: "atipica",
        13: "audifonos",
        14: None,
        15: "seguridad",
    }

    def __init__(self, width: int, height: int, trace_length: int = 20) -> None:
        self.width = width
        self.height = height

        # 1. Paleta de colores personalizada de IVOA
        # Naranja (#FFA500) para Arreglo Personal (clase 10), Naranja Alerta (#FF5722) para Uniforme Incompleto (clase 9)
        palette_colors = list(sv.ColorPalette.DEFAULT.colors)
        if len(palette_colors) > 10:
            palette_colors[10] = sv.Color(255, 165, 0)
            palette_colors[9] = sv.Color(255, 87, 34)
        self.custom_palette = sv.ColorPalette(colors=palette_colors)

        # 2. Inicialización de Anotadores Supervision
        self.box_annotator = sv.BoxAnnotator(color=self.custom_palette, thickness=2)
        self.label_annotator = sv.LabelAnnotator(
            color=self.custom_palette, text_color=sv.Color(255, 255, 255)
        )
        self.trace_annotator = sv.TraceAnnotator(trace_length=trace_length)

    def apply_privacy_blur(
        self,
        frame: np.ndarray,
        detections_tracked: sv.Detections,
        id_map: Optional[Dict[int, int]] = None,
        known_identities: Optional[Dict[str, str]] = None,
        blur_mode: str = "non_employees",
    ) -> np.ndarray:
        """
        Aplica anonimización y difuminado mosaico ultra rápido a rostros.
        - blur_mode 'all': difumina a todas las personas en pantalla.
        - blur_mode 'non_employees': difumina sólo a clientes/no empleados (sin nombre en BD).
        Costo: < 0.2 ms por persona en CPU.
        """
        if (
            not hasattr(detections_tracked, "tracker_id")
            or detections_tracked.tracker_id is None
            or len(detections_tracked.tracker_id) == 0
        ):
            return frame

        id_map = id_map or {}
        known_identities = known_identities or {}
        h_frame, w_frame = frame.shape[:2]

        for tid, bbox in zip(detections_tracked.tracker_id, detections_tracked.xyxy):
            mapped_tid = id_map.get(tid, tid)
            str_mapped = str(mapped_tid)
            assigned_name = known_identities.get(str_mapped, f"persona_id_{mapped_tid}")

            # Identificar si es empleado (tiene nombre real en BD distinto a persona_id_X)
            is_employee = not assigned_name.startswith("persona_id_")

            if blur_mode == "non_employees" and is_employee:
                continue

            # Región anatómica de cabeza/rostro (22% superior del cuerpo, 60% central)
            x1, y1, x2, y2 = bbox
            pw = x2 - x1
            ph = y2 - y1

            fx1 = max(0, int(x1 + pw * 0.18))
            fx2 = min(w_frame, int(x2 - pw * 0.18))
            fy1 = max(0, int(y1))
            fy2 = min(h_frame, int(y1 + ph * 0.23))

            rw = fx2 - fx1
            rh = fy2 - fy1
            if rw < 8 or rh < 8:
                continue

            # Pixelación por downsampling bicúbico + upsampling nearest (irreversible y ligero)
            face_roi = frame[fy1:fy2, fx1:fx2]
            small = cv2.resize(face_roi, (8, 8), interpolation=cv2.INTER_LINEAR)
            mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
            frame[fy1:fy2, fx1:fx2] = mosaic

        return frame

    def filter_uniform_visual(
        self,
        detections: sv.Detections,
        last_alert_uniforme: float,
        now_ts: float,
    ) -> sv.Detections:
        """
        Oculta el recuadro de uniforme incompleto (clase 9) una vez que pasaron 3 segundos
        del ticket emitido y durante el período de enfriamiento para no sobrecargar la pantalla.
        """
        if len(detections) == 0:
            return detections

        tiempo_desde_alerta = now_ts - last_alert_uniforme
        keep_idx = []
        for i, cid in enumerate(detections.class_id):
            if cid == 9:
                if 3.0 < tiempo_desde_alerta < 300.0:
                    continue
            keep_idx.append(i)

        return detections[keep_idx] if len(keep_idx) > 0 else sv.Detections.empty()

    def build_detection_labels(
        self,
        detections: sv.Detections,
        last_alert_times: Dict[str, float],
        detection_start_times: Dict[str, float],
        debounce_seconds: float = 1.0,
        now_ts: Optional[float] = None,
    ) -> List[str]:
        """Construye las etiquetas de objetos infractores con estado de debounce o ticket emitido."""
        now_ts = now_ts or time.time()
        labels: List[str] = []

        if len(detections) == 0:
            return labels

        for class_id, confidence in zip(detections.class_id, detections.confidence):
            name = self.CLASS_NAMES.get(class_id, f"obj_{class_id}")
            tag = self.CLASS_TO_TAG.get(class_id)

            if class_id == 13:
                name = "audifonos_no_detectados"
            elif class_id == 7:
                name = "audifonos_diadema"
            elif class_id == 14:
                name = "uniforme_completo"

            if tag and (now_ts - last_alert_times.get(tag, 0.0) < 3.0):
                labels.append(f"{name} [TICKET EMITIDO]")
            elif tag and tag in detection_start_times:
                elapsed = now_ts - detection_start_times[tag]
                labels.append(f"{name} ({elapsed:.1f}s / {debounce_seconds:.1f}s)")
            else:
                labels.append(f"{name} {confidence:.2f}")

        return labels

    def build_tracked_labels(
        self,
        detections_tracked: sv.Detections,
        id_map: Optional[Dict[int, int]] = None,
        known_identities: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """Construye las etiquetas para personas trackeadas asociando nombres conocidos en biometría."""
        labels: List[str] = []
        if (
            not hasattr(detections_tracked, "tracker_id")
            or detections_tracked.tracker_id is None
            or len(detections_tracked.tracker_id) == 0
        ):
            return labels

        id_map = id_map or {}
        known_identities = known_identities or {}

        for tid in detections_tracked.tracker_id:
            mapped_tid = id_map.get(tid, tid)
            str_mapped = str(mapped_tid)

            if str_mapped in known_identities:
                labels.append(f"{known_identities[str_mapped]}")
            else:
                labels.append(f"persona_id_{mapped_tid}")

        return labels

    def draw_geofencing(
        self,
        scene: np.ndarray,
        custom_polygon_points: Optional[np.ndarray],
    ) -> np.ndarray:
        """Oscurece con transparencia el área fuera del polígono activo y traza sus bordes en amarillo."""
        if custom_polygon_points is None or len(custom_polygon_points) < 3:
            return scene

        annotated = scene.copy()

        # 1. Máscara de oscurecimiento de zona inactiva con addWeighted
        mask = np.zeros(annotated.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [custom_polygon_points], 255, lineType=cv2.LINE_AA)

        black_image = np.zeros_like(annotated)
        darkened = cv2.addWeighted(annotated, 0.4, black_image, 0.6, 0)
        annotated = np.where(mask[:, :, None] == 0, darkened, annotated)

        # 2. Borde exterior del polígono activo en amarillo con anti-aliasing
        cv2.polylines(
            annotated,
            [custom_polygon_points],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        return annotated

    def draw_late_opening_banner(
        self,
        scene: np.ndarray,
        time_str: str,
    ) -> np.ndarray:
        """Dibuja un banner flotante rojo centrado en la parte superior advirtiendo apertura tardía."""
        annotated = scene
        text = f"ALERTA: APERTURA TARDIA [{time_str}]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        tx = (self.width - tw) // 2
        ty = 35

        # Rectángulo rojo con borde blanco centrado
        cv2.rectangle(
            annotated, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 255), -1
        )
        cv2.rectangle(
            annotated, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (255, 255, 255), 1
        )
        cv2.putText(
            annotated, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        return annotated

    def render(
        self,
        frame: np.ndarray,
        detections: sv.Detections,
        detections_tracked: sv.Detections,
        id_map: Optional[Dict[int, int]] = None,
        known_identities: Optional[Dict[str, str]] = None,
        custom_polygon_points: Optional[np.ndarray] = None,
        last_alert_times: Optional[Dict[str, float]] = None,
        detection_start_times: Optional[Dict[str, float]] = None,
        debounce_seconds: float = 1.0,
        late_opening_trigger_time: float = 0.0,
        late_opening_time_str: str = "",
        show_late_opening: bool = True,
        enable_privacy_blur: bool = False,
        privacy_blur_mode: str = "non_employees",
        now_ts: Optional[float] = None,
    ) -> np.ndarray:
        """
        Ejecuta el renderizado gráfico completo sobre el cuadro de video y retorna el frame anotado.
        """
        annotated = frame.copy()
        now_ts = now_ts or time.time()
        last_alerts = last_alert_times or {}
        det_starts = detection_start_times or {}

        # 0. Anonimización y Privacidad Selectiva (LFPDPPP / GDPR)
        if enable_privacy_blur:
            annotated = self.apply_privacy_blur(
                frame=annotated,
                detections_tracked=detections_tracked,
                id_map=id_map,
                known_identities=known_identities,
                blur_mode=privacy_blur_mode,
            )

        # 1. Filtrado visual de uniforme durante período de enfriamiento
        last_u = last_alerts.get("uniforme", 0.0)
        visible_detections = self.filter_uniform_visual(detections, last_u, now_ts)

        # 2. Generación de etiquetas y dibujo de objetos/infracciones
        if len(visible_detections) > 0:
            labels = self.build_detection_labels(
                visible_detections, last_alerts, det_starts, debounce_seconds, now_ts
            )
            annotated = self.box_annotator.annotate(scene=annotated, detections=visible_detections)
            annotated = self.label_annotator.annotate(
                scene=annotated, detections=visible_detections, labels=labels
            )

        # 3. Dibujo de geofencing poligonal (si existe)
        annotated = self.draw_geofencing(annotated, custom_polygon_points)

        # 4. Dibujo de personas trackeadas con identificador o nombre biométrico y traza
        if (
            hasattr(detections_tracked, "tracker_id")
            and detections_tracked.tracker_id is not None
            and len(detections_tracked.tracker_id) > 0
        ):
            annotated = self.box_annotator.annotate(
                scene=annotated, detections=detections_tracked
            )
            tracked_labels = self.build_tracked_labels(
                detections_tracked, id_map, known_identities
            )
            annotated = self.label_annotator.annotate(
                scene=annotated, detections=detections_tracked, labels=tracked_labels
            )
            annotated = self.trace_annotator.annotate(
                scene=annotated, detections=detections_tracked
            )

        # 5. Banner superior de Apertura Tardía (visible durante 15 segundos)
        if (
            show_late_opening
            and (late_opening_trigger_time > 0.0)
            and (now_ts - late_opening_trigger_time < 15.0)
            and late_opening_time_str
        ):
            annotated = self.draw_late_opening_banner(annotated, late_opening_time_str)

        return annotated
