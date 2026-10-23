"""
alert_manager.py — Gestor Centralizado de Alertas, Debounce y Atribución para IVOA.

Maneja de forma aislada:
1. Máquina de estados temporal (debounce de tiempo y cooldown anti-spam).
2. Atribución espacial de infractores/culpables (_get_culprits) asociando objetos con personas trackeadas.
3. Sincronización en caliente con la configuración de notificaciones (config/notify_config.json).
4. Filtro maestro de supresión de alertas y etiquetas visuales.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

logger = logging.getLogger("alert_manager")


class AlertManager:
    """Administrador desacoplado de reglas de tiempo, debounce y emisión de incidencias."""

    # Mapeo de clases a configuraciones de regla de negocio
    RULE_DEFINITIONS = {
        5: {  # Celular
            "tag": "celular",
            "event_type": "incidencia_celular",
            "debounce": 0.5,  # 0.5 s para reacción ágil en celular
            "severity": "warning",
            "msg_template": "ALERTA: {culprit} detectado(a) usando dispositivo móvil (Celular).",
        },
        7: {  # Audífonos
            "tag": "audifonos",
            "event_type": "incidencia_audifonos",
            "debounce": 1.0,
            "severity": "warning",
            "msg_template": "ALERTA: {culprit} detectado(a) usando audífonos de diadema.",
        },
        2: {  # Billetes / Dinero
            "tag": "dinero",
            "event_type": "incidencia_dinero",
            "debounce": 1.0,
            "severity": "warning",
            "msg_template": "ALERTA: {culprit} detectado con manejo de billetes o efectivo.",
        },
        4: {  # Comida / Alimentos
            "tag": "alimentos",
            "event_type": "incidencia_alimentos",
            "debounce": 1.0,
            "severity": "warning",
            "msg_template": "ALERTA: {culprit} detectado(a) consumiendo alimentos/bebidas.",
        },
        8: {  # Pantallas / Páginas no autorizadas
            "tag": "paginas",
            "event_type": "incidencia_paginas",
            "debounce": 1.0,
            "severity": "warning",
            "msg_template": "ALERTA: {culprit} con ingreso a páginas / redes no autorizadas en monitor.",
        },
        10: {  # Arreglo personal / Maquillaje
            "tag": "arreglo",
            "event_type": "incidencia_arreglo_personal",
            "debounce": 1.5,
            "severity": "warning",
            "msg_template": "ALERTA: {culprit} realizando arreglo personal / maquillaje.",
        },
        9: {  # Uniforme incompleto
            "tag": "uniforme",
            "event_type": "incidencia_uniforme",
            "debounce": 1.0,
            "severity": "warning",
            "msg_template": "ALERTA: {culprit} con código de vestimenta incompleto.",
        },
        6: {  # Llaves en mesa de atención
            "tag": "llaves",
            "event_type": "incidencia_llaves",
            "debounce": 1.0,
            "severity": "warning",
            "msg_template": "ALERTA: {culprit} con llaves detectadas en área de atención.",
        },
        11: {  # Situación atípica (de pie / postura anómala)
            "tag": "atipica",
            "event_type": "incidencia_situacion_atipica",
            "debounce": 1.0,
            "severity": "warning",
            "msg_template": "ALERTA: {culprit} detectado en Situación atípica (Persona de pie / Movimiento anómalo).",
        },
    }

    CLASS_TO_ALERT_CONFIG = {
        2: "incidencia_dinero",
        4: "incidencia_alimentos",
        5: "incidencia_celular",
        6: "incidencia_llaves",
        7: "incidencia_audifonos",
        8: "incidencia_paginas",
        9: "incidencia_uniforme",
        14: "incidencia_uniforme",  # Uniforme completo mapea a la misma bandera
        10: "incidencia_arreglo_personal",
        11: "incidencia_situacion_atipica",
    }

    TAG_TO_CLASS = {
        "celular": 5,
        "audifonos": 7,
        "dinero": 2,
        "alimentos": 4,
        "paginas": 8,
        "arreglo": 10,
        "uniforme": 9,
        "llaves": 6,
        "atipica": 11,
    }

    def __init__(
        self,
        width: int,
        height: int,
        config_path: Optional[str | Path] = None,
        debounce_seconds: float = 1.0,
        cooldown_seconds: float = 8.0,
    ) -> None:
        self.width = width
        self.height = height
        self.DEBOUNCE_SECONDS = debounce_seconds
        self.ALERT_COOLDOWN = cooldown_seconds

        # Resolución de ruta al archivo JSON de configuración
        if config_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.config_path = base_dir / "config" / "notify_config.json"
        else:
            self.config_path = Path(config_path)

        # Estado temporal y control anti-spam
        self.last_alert_times: Dict[str, float] = {}
        self.detection_start_times: Dict[str, float] = {}
        self.last_seen_times: Dict[str, float] = {}
        self.triggered_alerts: Set[str] = set()

        # Configuración por defecto
        self.alert_config: Dict[str, Any] = {
            "incidencia_dinero": False,
            "incidencia_alimentos": False,
            "incidencia_celular": False,
            "incidencia_audifonos": False,
            "incidencia_paginas": False,
            "incidencia_arreglo_personal": False,
            "incidencia_seguridad": False,
            "incidencia_uniforme": False,
            "incidencia_llaves": False,
            "incidencia_situacion_atipica": False,
            "incidencia_saturacion": False,
            "late_opening": False,
            "excessive_service_time": False,
            "customer_abandonment": False,
            "service_completed": False,
            "zona_izquierda": True,
            "zona_derecha": True,
            "enable_privacy_blur": False,
            "privacy_blur_mode": "non_employees",
        }
        self.custom_polygon_points: Optional[np.ndarray] = None

        # Sincronización inicial
        self.sync_config(frame_count=0, force=True)

    def sync_config(self, frame_count: int, force: bool = False) -> None:
        """Lee notify_config.json cada 30 cuadros y actualiza reglas y polígono de geofencing."""
        if not force and frame_count % 30 != 0:
            return

        if not self.config_path.exists():
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

            self.alert_config["incidencia_celular"] = cfg.get("enable_celulares", True)
            self.alert_config["incidencia_alimentos"] = cfg.get("enable_alimentos", True)
            self.alert_config["incidencia_audifonos"] = cfg.get("enable_audifonos", True)
            self.alert_config["incidencia_paginas"] = cfg.get("enable_paginas", True)
            self.alert_config["incidencia_arreglo_personal"] = cfg.get("enable_arreglo", True)
            self.alert_config["incidencia_situacion_atipica"] = cfg.get("enable_atipica", True)
            self.alert_config["incidencia_dinero"] = cfg.get("enable_dinero", True)
            self.alert_config["incidencia_uniforme"] = cfg.get("enable_vestimenta", True)
            self.alert_config["late_opening"] = cfg.get("enable_apertura", True)

            self.alert_config["zona_izquierda"] = cfg.get("zona_izquierda", True)
            self.alert_config["zona_derecha"] = cfg.get("zona_derecha", True)

            poly_data = cfg.get("custom_polygon", None)
            if poly_data and len(poly_data) >= 3:
                pts = [
                    [int(p["x"] * self.width), int(p["y"] * self.height)]
                    for p in poly_data
                ]
                self.custom_polygon_points = np.array(pts, np.int32)
            else:
                self.custom_polygon_points = None

        except Exception as e:
            logger.debug("Error leyendo notify_config.json: %s", e)

    def get_culprits(
        self,
        class_id: int,
        detections: Any,
        detections_tracked: Any,
        id_map: Optional[Dict[int, int]] = None,
        known_identities: Optional[Dict[str, str]] = None,
    ) -> List[str]:
        """
        Determina qué persona (o personas) cometieron la infracción según proximidad geométrica.
        Aplica los filtros de zona_izquierda y zona_derecha.
        """
        culprits: Set[str] = set()
        id_map = id_map or {}
        known_identities = known_identities or {}

        try:
            indices = np.where(detections.class_id == class_id)[0]
            if len(indices) > 0:
                for idx in indices:
                    px1, py1, px2, py2 = detections.xyxy[idx]
                    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0

                    zona_izq = self.alert_config.get("zona_izquierda", True)
                    zona_der = self.alert_config.get("zona_derecha", True)
                    if pcx < self.width / 2.0 and not zona_izq:
                        continue
                    if pcx >= self.width / 2.0 and not zona_der:
                        continue

                    best_tid = None
                    min_dist = float("inf")
                    found_inside = False

                    if (
                        hasattr(detections_tracked, "tracker_id")
                        and detections_tracked.tracker_id is not None
                    ):
                        for tid, bbox in zip(
                            detections_tracked.tracker_id, detections_tracked.xyxy
                        ):
                            hx1, hy1, hx2, hy2 = bbox
                            hcx, hcy = (hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0
                            dist = (pcx - hcx) ** 2 + (pcy - hcy) ** 2

                            bw, bh = hx2 - hx1, hy2 - hy1
                            # Verificación de contención ampliada (30% de margen)
                            if (
                                (hx1 - bw * 0.30) <= pcx <= (hx2 + bw * 0.30)
                                and (hy1 - bh * 0.30) <= pcy <= (hy2 + bh * 0.30)
                            ):
                                mapped_tid = id_map.get(tid, tid)
                                str_mapped = str(mapped_tid)
                                culprits.add(
                                    known_identities.get(
                                        str_mapped, f"persona_id_{mapped_tid}"
                                    )
                                )
                                found_inside = True
                                break

                            if dist < min_dist:
                                min_dist = dist
                                best_tid = tid

                        if not found_inside and best_tid is not None:
                            mapped_tid = id_map.get(best_tid, best_tid)
                            str_mapped = str(mapped_tid)
                            culprits.add(
                                known_identities.get(
                                    str_mapped, f"persona_id_{mapped_tid}"
                                )
                            )
        except Exception as e:
            logger.debug("Error calculando culpables para clase %d: %s", class_id, e)

        if not culprits:
            mapped_tid = 1
            if len(id_map) > 0:
                mapped_tid = list(id_map.values())[0]
            str_mapped = str(mapped_tid)
            culprits.add(
                known_identities.get(str_mapped, f"persona_id_{mapped_tid}")
            )

        return list(culprits)

    def evaluate_detections(
        self,
        detections: Any,
        detections_tracked: Any,
        id_map: Optional[Dict[int, int]] = None,
        known_identities: Optional[Dict[str, str]] = None,
        now_ts: Optional[float] = None,
        extra_active_flags: Optional[Dict[str, bool]] = None,
    ) -> List[Tuple[str, str, str]]:
        """
        Evalúa las detecciones actuales con la máquina de estados de debounce y cooldown.
        Retorna la lista de alertas generadas en este cuadro: [(event_type, mensaje, severidad), ...]
        """
        alerts: List[Tuple[str, str, str]] = []
        now_ts = now_ts or time.time()
        extra_flags = extra_active_flags or {}

        active_classes = set(detections.class_id) if len(detections) > 0 else set()

        for cid, rule in self.RULE_DEFINITIONS.items():
            tag = rule["tag"]
            event_type = rule["event_type"]
            req_debounce = rule["debounce"]
            severity = rule["severity"]
            template = rule["msg_template"]

            # Comprobar si la clase está presente o tiene condición externa activa (ej. situación atípica por persona parada)
            is_present = (cid in active_classes) or bool(extra_flags.get(tag, False))

            if is_present:
                if tag not in self.detection_start_times:
                    self.detection_start_times[tag] = now_ts
                self.last_seen_times[tag] = now_ts

                elapsed = now_ts - self.detection_start_times[tag]
                if elapsed >= req_debounce:
                    last_alert = self.last_alert_times.get(tag, 0.0)
                    if now_ts - last_alert >= self.ALERT_COOLDOWN:
                        culprits = self.get_culprits(
                            cid, detections, detections_tracked, id_map, known_identities
                        )
                        for culprit in culprits:
                            msg = template.format(culprit=culprit)
                            alerts.append((event_type, msg, severity))
                        self.last_alert_times[tag] = now_ts
            else:
                # Período de gracia de 1.0 segundo antes de resetear el cronómetro de debounce
                if tag in self.last_seen_times and (now_ts - self.last_seen_times[tag] > 1.0):
                    self.detection_start_times.pop(tag, None)
                    self.last_seen_times.pop(tag, None)

        return alerts

    def is_alert_enabled(self, alert_id: str) -> bool:
        """Verifica si una alerta específica está habilitada en notify_config.json."""
        return bool(self.alert_config.get(alert_id, True))

    def filter_active_alerts(
        self, alerts: List[Tuple[str, str, str]]
    ) -> List[Tuple[str, str, str]]:
        """Filtra una lista de alertas eliminando aquellas apagadas en alert_config."""
        return [a for a in alerts if self.is_alert_enabled(a[0])]

    def should_display_class(self, class_id: int) -> bool:
        """Determina si un recuadro de clase debe dibujarse o ser suprimido visualmente."""
        if class_id == 0:  # Persona/humano siempre se dibuja
            return True
        config_key = self.CLASS_TO_ALERT_CONFIG.get(class_id)
        if config_key is None:
            return True
        return self.is_alert_enabled(config_key)

    def get_label_tag_status(
        self,
        tag: Optional[str],
        base_name: str,
        confidence: float,
        now_ts: Optional[float] = None,
    ) -> str:
        """Genera el texto de etiqueta enriquecida con [TICKET EMITIDO] o cronómetro de debounce."""
        now_ts = now_ts or time.time()
        if tag and (now_ts - self.last_alert_times.get(tag, 0.0) < 3.0):
            return f"{base_name} [TICKET EMITIDO]"
        elif tag and tag in self.detection_start_times:
            elapsed = now_ts - self.detection_start_times[tag]
            rule = next((r for r in self.RULE_DEFINITIONS.values() if r["tag"] == tag), None)
            debounce = rule["debounce"] if rule else self.DEBOUNCE_SECONDS
            return f"{base_name} ({elapsed:.1f}s / {debounce:.1f}s)"
        return f"{base_name} {confidence:.2f}"
