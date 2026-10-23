"""
queue_analyzer.py — Analizador de Colas, Tiempos, Flujos y Sesiones para IVOA.

Gestiona la inteligencia espacial y temporal de sucursales:
1. Zonas Espaciales Poligonales: Fila de espera (fila_zone) y Módulo de atención (modulo_zone).
2. Conteo Bidireccional de Tránsito: Cruce de línea virtual (LineZone in/out).
3. Detección de Acceso No Autorizado / Tailgating: Entradas múltiples en < 3.5 segundos.
4. Detección de Saturación de Fila: Alerta cuando la fila acumula >= 4 personas.
5. Ciclo de Vida de Sesiones: Registro de tiempos de entrada, espera, atención y servicio completado.
6. Abandono de Clientes: Detección de personas que salen de fila tras > 60 s sin recibir atención.
7. Apertura Tardía: Lectura de reloj en cabecera de video (OCR) o reloj de sistema (> 9:15 AM).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger("queue_analyzer")


class QueueAnalyzer:
    """Motor de análisis espacial de flujos de personas, colas de espera y métricas de servicio."""

    def __init__(
        self,
        width: int,
        height: int,
        fila_polygon: Optional[np.ndarray] = None,
        modulo_polygon: Optional[np.ndarray] = None,
        debounce_seconds: float = 1.0,
        cooldown_seconds: float = 8.0,
        ocr_reader: Any = None,
    ) -> None:
        self.width = width
        self.height = height
        self.DEBOUNCE_SECONDS = debounce_seconds
        self.ALERT_COOLDOWN = cooldown_seconds
        self._ocr_reader = ocr_reader

        # 1. Zonas espaciales por defecto (Mitad superior = Módulo, Mitad inferior = Fila)
        if modulo_polygon is None:
            self.modulo_polygon = np.array(
                [[0, 0], [width, 0], [width, height // 2], [0, height // 2]]
            )
        else:
            self.modulo_polygon = modulo_polygon

        if fila_polygon is None:
            self.fila_polygon = np.array(
                [[0, height // 2], [width, height // 2], [width, height], [0, height]]
            )
        else:
            self.fila_polygon = fila_polygon

        # Inicialización de Zonas Supervision
        self.modulo_zone = sv.PolygonZone(
            polygon=self.modulo_polygon,
            triggering_anchors=[sv.Position.CENTER],
        )
        self.fila_zone = sv.PolygonZone(
            polygon=self.fila_polygon,
            triggering_anchors=[sv.Position.CENTER],
        )
        self.line_zone = sv.LineZone(
            start=sv.Point(0, height // 2),
            end=sv.Point(width, height // 2),
            triggering_anchors=[sv.Position.CENTER],
        )

        # 2. Métricas y contadores expuestos públicamente
        self._fila_count: int = 0
        self._modulo_count: int = 0
        self._in_count: int = 0
        self._out_count: int = 0

        # 3. Control de aperturas tardías
        self.first_client_seen: bool = False
        self.late_opening_trigger_time: float = 0.0
        self.late_opening_time_str: str = ""

        # 4. Estado de tailgating y persistencia de cruces
        self.recent_entries: List[Tuple[float, int, List[float]]] = []  # (ts, tid, bbox)
        self.last_alert_seguridad: float = 0.0

        # 5. Estado de colas y debounce de saturación
        self.saturacion_start_time: Optional[float] = None
        self.triggered_alerts: Set[str] = set()

        # 6. Historial de sesiones de personas
        self.track_history: Dict[int, Dict[str, Any]] = {}

    # Propiedades públicas para compatibilidad con app_api.py
    @property
    def fila_count(self) -> int:
        return self._fila_count

    @property
    def modulo_count(self) -> int:
        return self._modulo_count

    @property
    def in_count(self) -> int:
        return self._in_count

    @property
    def out_count(self) -> int:
        return self._out_count

    def update(
        self,
        frame: np.ndarray,
        detections_tracked: sv.Detections,
        now_ts: Optional[float] = None,
    ) -> Tuple[List[Tuple[str, str, str]], List[Tuple[int, Dict[str, Any]]], List[List[float]]]:
        """
        Procesa el fotograma y las personas trackeadas para actualizar métricas de cola,
        conteo de línea, eventos de tailgating y ciclo de vida de sesiones.
        Retorna: (alertas_generadas, sesiones_a_actualizar, cajas_de_tailgating).
        """
        alerts: List[Tuple[str, str, str]] = []
        sessions_to_update: List[Tuple[int, Dict[str, Any]]] = []
        tailgating_boxes: List[List[float]] = []

        now_ts = now_ts or time.time()
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        has_tracks = (
            hasattr(detections_tracked, "tracker_id")
            and detections_tracked.tracker_id is not None
            and len(detections_tracked.tracker_id) > 0
        )

        # ---------------------------------------------------------------------
        # 1. CRUCE DE LÍNEA Y CONTROL DE ACCESOS (LINE ZONE & TAILGATING)
        # ---------------------------------------------------------------------
        crossed_in, crossed_out = self.line_zone.trigger(detections=detections_tracked)
        self._in_count = self.line_zone.in_count
        self._out_count = self.line_zone.out_count

        # Depuración de entradas viejas (> 10s)
        self.recent_entries = [e for e in self.recent_entries if now_ts - e[0] <= 10.0]

        if crossed_in is not None and len(crossed_in) > 0 and has_tracks:
            for i, (is_in, is_out) in enumerate(zip(crossed_in, crossed_out)):
                if is_in or is_out:
                    tid = int(detections_tracked.tracker_id[i])
                    bbox = detections_tracked.xyxy[i].tolist()

                    # Verificar si otra persona distinta ingresó en los últimos 3.5 segundos
                    for old_ts, old_tid, old_bbox in self.recent_entries:
                        if old_tid != tid and (now_ts - old_ts < 3.5):
                            tailgating_boxes.append(bbox)
                            tailgating_boxes.append(old_bbox)

                            if now_ts - self.last_alert_seguridad >= self.ALERT_COOLDOWN:
                                alerts.append(
                                    (
                                        "incidencia_seguridad",
                                        "ALERTA CRITICA: Acceso multiple no autorizado (Tailgating).",
                                        "critical",
                                    )
                                )
                                self.last_alert_seguridad = now_ts

                    self.recent_entries.append((now_ts, tid, bbox))

        # ---------------------------------------------------------------------
        # 2. EVALUACIÓN DE ZONAS (FILA Y MÓDULO)
        # ---------------------------------------------------------------------
        fila_in = self.fila_zone.trigger(detections=detections_tracked)
        modulo_in = self.modulo_zone.trigger(detections=detections_tracked)

        self._fila_count = int(np.sum(fila_in)) if len(fila_in) > 0 else 0
        self._modulo_count = int(np.sum(modulo_in)) if len(modulo_in) > 0 else 0

        # Alerta de Saturación de Fila (>= 4 personas en espera)
        if self._fila_count >= 4:
            if self.saturacion_start_time is None:
                self.saturacion_start_time = now_ts
            elif now_ts - self.saturacion_start_time >= self.DEBOUNCE_SECONDS:
                if "saturacion" not in self.triggered_alerts:
                    alerts.append(
                        (
                            "incidencia_saturacion",
                            "ALERTA: Fila saturada (4+ clientes en espera). Se requiere abrir módulo de apoyo.",
                            "warning",
                        )
                    )
                    self.triggered_alerts.add("saturacion")
        else:
            self.saturacion_start_time = None
            self.triggered_alerts.discard("saturacion")

        # ---------------------------------------------------------------------
        # 3. CICLO DE VIDA DE SESIONES Y TIEMPOS DE SERVICIO
        # ---------------------------------------------------------------------
        if has_tracks:
            current_ids: Set[int] = set()

            for tid_raw, in_f, in_m in zip(
                detections_tracked.tracker_id, fila_in, modulo_in
            ):
                tid = int(tid_raw)
                current_ids.add(tid)

                if tid not in self.track_history:
                    initial_status = (
                        "en_modulo" if in_m else ("en_fila" if in_f else "detectado")
                    )
                    self.track_history[tid] = {
                        "first_seen": current_time_str,
                        "last_seen_epoch": now_ts,
                        "entered_fila": current_time_str if in_f else None,
                        "entered_fila_epoch": now_ts if in_f else None,
                        "left_fila": None,
                        "entered_modulo": current_time_str if in_m else None,
                        "entered_modulo_epoch": now_ts if in_m else None,
                        "left_modulo": None,
                        "status": initial_status,
                        "total_service_time": None,
                    }

                    # Verificación de apertura tardía con el primer cliente del día
                    if not self.first_client_seen:
                        self.first_client_seen = True
                        late_alert = self._evaluate_late_opening(frame, now_ts)
                        if late_alert:
                            alerts.append(late_alert)

                session = self.track_history[tid]
                session["last_seen_epoch"] = now_ts

                # Transición a Fila
                if in_f and not session["entered_fila"]:
                    session["entered_fila"] = current_time_str
                    session["entered_fila_epoch"] = now_ts
                    session["status"] = "en_fila"
                if not in_f and session["entered_fila"] and not session["left_fila"]:
                    session["left_fila"] = current_time_str

                # Transición a Módulo de Atención
                if in_m and not session["entered_modulo"]:
                    session["entered_modulo"] = current_time_str
                    session["entered_modulo_epoch"] = now_ts
                    session["status"] = "en_modulo"

                # Alerta de Atención Excedida (> 15s en módulo para calibración de demo)
                if in_m and session.get("entered_modulo_epoch"):
                    mod_dur = now_ts - session["entered_modulo_epoch"]
                    if mod_dur >= 15.0:
                        if "atencion_excedida" not in self.triggered_alerts:
                            alerts.append(
                                (
                                    "excessive_service_time",
                                    "ALERTA: Tiempo de atención excedido (>15 min). Posible bloqueo en trámite.",
                                    "warning",
                                )
                            )
                            self.triggered_alerts.add("atencion_excedida")

                # Salida de Módulo -> Servicio Completado
                if not in_m and session["entered_modulo"] and not session["left_modulo"]:
                    session["left_modulo"] = current_time_str
                    session["status"] = "atendido"
                    try:
                        fmt = "%Y-%m-%d %H:%M:%S"
                        t_in = datetime.strptime(session["entered_modulo"], fmt)
                        t_out = datetime.strptime(session["left_modulo"], fmt)
                        session["total_service_time"] = int((t_out - t_in).total_seconds())
                    except Exception:
                        session["total_service_time"] = int(now_ts - session.get("entered_modulo_epoch", now_ts))

                    alerts.append(
                        (
                            "service_completed",
                            f"Servicio completado en {session['total_service_time']}s",
                            "info",
                        )
                    )
                    sessions_to_update.append((tid, session.copy()))

            # Detección de Abandono de Fila para personas desaparecidas de la cámara
            tids_to_remove: List[int] = []
            for tid_history, session_history in self.track_history.items():
                if tid_history not in current_ids:
                    if now_ts - session_history["last_seen_epoch"] > 60.0:
                        if session_history["entered_fila"] and not session_history["entered_modulo"]:
                            session_history["status"] = "abandonado"
                            if not session_history["left_fila"]:
                                session_history["left_fila"] = datetime.fromtimestamp(
                                    session_history["last_seen_epoch"]
                                ).strftime("%Y-%m-%d %H:%M:%S")

                            if "abandono" not in self.triggered_alerts:
                                alerts.append(
                                    (
                                        "customer_abandonment",
                                        "Cliente se retiró sin ser atendido tras 60s",
                                        "warning",
                                    )
                                )
                                self.triggered_alerts.add("abandono")

                        sessions_to_update.append((tid_history, session_history.copy()))
                        tids_to_remove.append(tid_history)

            for tid_rem in tids_to_remove:
                del self.track_history[tid_rem]

        return alerts, sessions_to_update, tailgating_boxes

    # -------------------------------------------------------------------------
    # 4. DETECCIÓN DE APERTURA TARDÍA
    # -------------------------------------------------------------------------
    def _evaluate_late_opening(
        self, frame: np.ndarray, now_ts: float
    ) -> Optional[Tuple[str, str, str]]:
        """Determina si la sucursal abrió tarde inspeccionando la hora en video o del sistema."""
        is_late = False
        cam_time_str = "N/A"

        # 1. Intentar leer la hora grabada en la cabecera del video vía OCR
        if self._ocr_reader is not None:
            try:
                crop = frame[0:150, 0:600]
                if crop.size > 0:
                    big = cv2.resize(crop, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                    gray_crop = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
                    res = self._ocr_reader.readtext(gray_crop, detail=0)
                    for text in res:
                        match = re.search(r"(\d{1,2})\s*:\s*(\d{2})", text)
                        if match:
                            h, m = int(match.group(1)), int(match.group(2))
                            cam_time_str = f"{h:02d}:{m:02d}"
                            is_late = (h > 9) or (h == 9 and m > 15)
                            break
            except Exception as e:
                logger.debug("Fallo en OCR de cabecera de reloj: %s", e)

        # 2. Fallback a reloj del sistema
        now = datetime.now()
        if cam_time_str == "N/A":
            is_late = (now.hour > 9) or (now.hour == 9 and now.minute > 15)

        if is_late:
            source = (
                f"Cámara ({cam_time_str})"
                if cam_time_str != "N/A"
                else f"Sistema ({now.strftime('%H:%M:%S')})"
            )
            self.late_opening_trigger_time = now_ts
            self.late_opening_time_str = source
            return (
                "late_opening",
                f"Primer cliente detectado a las {source}. Apertura tardía.",
                "warning",
            )

        return None
