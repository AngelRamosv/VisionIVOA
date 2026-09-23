import os
import time
from datetime import datetime
import cv2
import numpy as np
import warnings
import supervision as sv
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_BASE_PATH = os.path.join(BASE_DIR, "yolov8n.pt")
MODEL_CUSTOM_PATH = os.path.join(BASE_DIR, "yolov8_ivoa_v2.pt")

def extract_color_signature(frame, bbox):
    """Extrae una huella digital matemática basada en los colores de la ropa (torso superior)"""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    if pw < 10 or ph < 10:
        return None
    # Recorte del torso para no incluir el piso, sillas o sombras
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

    def _get_culprits(self, class_id, detections, detections_tracked):
        culprits = set()
        try:
            import numpy as np
            indices = np.where(detections.class_id == class_id)[0]
            
            if len(indices) > 0:
                for idx in indices:
                    px1, py1, px2, py2 = detections.xyxy[idx]
                    pcx, pcy = (px1 + px2)/2, (py1 + py2)/2
                    
                    best_tid = None
                    min_dist = float('inf')
                    found_inside = False
                    
                    if hasattr(detections_tracked, 'tracker_id') and detections_tracked.tracker_id is not None:
                        for tid, bbox in zip(detections_tracked.tracker_id, detections_tracked.xyxy):
                            hx1, hy1, hx2, hy2 = bbox
                            hcx, hcy = (hx1 + hx2)/2, (hy1 + hy2)/2
                            dist = (pcx - hcx)**2 + (pcy - hcy)**2
                            
                            bw, bh = hx2-hx1, hy2-hy1
                            if (hx1 - bw*0.30) <= pcx <= (hx2 + bw*0.30) and (hy1 - bh*0.30) <= pcy <= (hy2 + bh*0.30):
                                mapped_tid = self.id_map.get(tid, tid)
                                str_mapped = str(mapped_tid)
                                culprits.add(getattr(self, 'known_identities', {}).get(str_mapped, f"persona_id_{mapped_tid}"))
                                found_inside = True
                                break
                                
                            if dist < min_dist:
                                min_dist = dist
                                best_tid = tid
                                
                        if not found_inside and best_tid is not None:
                            mapped_tid = self.id_map.get(best_tid, best_tid)
                            str_mapped = str(mapped_tid)
                            culprits.add(getattr(self, 'known_identities', {}).get(str_mapped, f"persona_id_{mapped_tid}"))
        except: pass
        
        if not culprits:
            mapped_tid = 1
            if hasattr(self, 'id_map') and len(self.id_map) > 0:
                mapped_tid = list(self.id_map.values())[0]
            str_mapped = str(mapped_tid)
            culprits.add(getattr(self, 'known_identities', {}).get(str_mapped, f"persona_id_{mapped_tid}"))
            
        return list(culprits)
    """Motor de IA Central: Maneja Modelos, Tracking, Zonas y Reglas de Negocio"""
    def __init__(self, width, height, frame_skip=5):
        self.width = width
        self.height = height
        self.FRAME_SKIP = frame_skip
        
        self.model_base = YOLO(MODEL_BASE_PATH if os.path.exists(MODEL_BASE_PATH) else "yolov8n.pt") # Rastreador universal
        self.model_custom = YOLO(os.path.join(BASE_DIR, "yolov8_ivoa_v5.pt")) # Especialista en Audífonos, Celular, Uniforme, Comida
        self.model_screens = YOLO(os.path.join(BASE_DIR, "yolov8_ivoa_v7.pt")) # Especialista en Pantallas y Páginas No Autorizadas
        self.model_grooming_cash = YOLO(os.path.join(BASE_DIR, "yolov8_ivoa_v8.pt") if os.path.exists(os.path.join(BASE_DIR, "yolov8_ivoa_v8.pt")) else self.model_custom) # Especialista en Arreglo Personal y Billetes
        self.model_v9 = YOLO(os.path.join(BASE_DIR, "yolov8_ivoa_v9.pt")) # Nuevo Especialista en Billetes, Llaves y Situación Atípica
        
        # 2. Configuración Espacial
        self.modulo_polygon = np.array([[0, 0], [width, 0], [width, height//2], [0, height//2]])
        self.fila_polygon = np.array([[0, height//2], [width, height//2], [width, height], [0, height]])
        
        self.fila_zone = sv.PolygonZone(polygon=self.fila_polygon, triggering_anchors=[sv.Position.CENTER])
        self.modulo_zone = sv.PolygonZone(polygon=self.modulo_polygon, triggering_anchors=[sv.Position.CENTER])
        self.line_zone = sv.LineZone(
            start=sv.Point(0, height // 2), 
            end=sv.Point(width, height // 2), 
            triggering_anchors=[sv.Position.CENTER]
        )
        
        # 3. Módulos de Procesamiento y Anotación
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.55,
                lost_track_buffer=180,
                minimum_matching_threshold=0.50,
                frame_rate=30
            )

        # Paleta de colores personalizada: Negro (#0f0f0f) para Arreglo Personal (clase 10), Naranja para Uniforme Incompleto (clase 9)
        palette_colors = list(sv.ColorPalette.DEFAULT.colors)
        palette_colors[10] = sv.Color(255, 165, 0) # Naranja brillante para Arreglo Personal
        palette_colors[9] = sv.Color(255, 87, 34) # Naranja alerta uniforme
        custom_palette = sv.ColorPalette(colors=palette_colors)

        self.box_annotator = sv.BoxAnnotator(color=custom_palette, thickness=2)
        self.label_annotator = sv.LabelAnnotator(color=custom_palette, text_color=sv.Color(255, 255, 255))
        self.trace_annotator = sv.TraceAnnotator(trace_length=20)
        self.line_zone_annotator = sv.LineZoneAnnotator()
        self.blur_annotator = sv.BlurAnnotator(kernel_size=15)
        
        # 4. Estado Interno (Memoria y Métricas)
        self.id_map = {}
        self.persistent_identities = {}
        self.track_history = {}
        self.known_identities = {"999": "persona_id_999"} # Ejemplo de diccionario para caras conocidas
        self.first_client_seen = False
        self.late_opening_trigger_time = 0
        self.late_opening_time_str = ""
        self.in_count = 0
        self.out_count = 0
        self.fila_count = 0
        self.modulo_count = 0
        self.alerta_saturacion = False

        # --- INICIO ASINCRONO OCR ---
        def load_ocr():
            try:
                import easyocr
                self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            except Exception:
                pass
        import threading
        threading.Thread(target=load_ocr, daemon=True).start()
        # ----------------------------
        self.saturacion_frames = 0
        
        self.last_alert_time = 0
        self.last_alert_uniforme = 0
        self.last_alert_times = {} # Registro del último timestamp de emisión por tipo de alerta
        self.detection_start_times = {} # Registro de tiempo de inicio de detección
        self.triggered_alerts = set() # Evitar alertas duplicadas
        self.last_seen_times = {} # Última vez que se vio cada objeto (da 1.0s de gracia)
        self.DEBOUNCE_SECONDS = 1.0 # 1.0s continuo para respuesta rápida
        self.ALERT_COOLDOWN = 8.0 # Evitar spam
        self.audifonos_frames = 0
        self.celular_frames = 0
        
        # --- CONFIGURACIÓN DE ALERTAS (ENCENDER / APAGAR FÁCILMENTE) ---
        # Si no quieres que suene alguna alerta, simplemente cambia True por False.
        self.alert_config = {
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
            "service_completed":  False
        }

        
        self.last_detections = sv.Detections.empty()
        self.last_detections_tracked = sv.Detections.empty()
        
        # --- Validador de Billetes en 3 Capas (Color + Textura + OCR) ---
        self._ocr_reader = None
        self.BILL_DENOMINATIONS = {"20", "50", "100", "200", "500", "1000"}
        self._ocr_cache = {}  # cache: evitar OCR en cada frame
        
        # Cargar OCR en segundo plano sin bloquear el video (sin impresiones para evitar crash)
        import threading
        try:
            import easyocr
            self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except Exception as e:
            self._ocr_reader = None



    def _sync_alert_config(self, frame_count):
        if frame_count % 30 != 0: return
        try:
            import json
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            cfg_path = os.path.join(base_dir, "notify_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                    # Mapear el JSON del dashboard hacia el alert_config del Edge Sensor
                    self.alert_config["incidencia_celular"] = cfg.get("enable_celulares", True)
                    self.alert_config["incidencia_alimentos"] = cfg.get("enable_alimentos", True)
                    self.alert_config["incidencia_audifonos"] = cfg.get("enable_audifonos", True)
                    self.alert_config["incidencia_paginas"] = cfg.get("enable_paginas", True)
                    self.alert_config["incidencia_arreglo_personal"] = cfg.get("enable_arreglo", True)
                    self.alert_config["incidencia_situacion_atipica"] = cfg.get("enable_atipica", True)
                    self.alert_config["incidencia_dinero"] = cfg.get("enable_dinero", True)
                    self.alert_config["incidencia_uniforme"] = cfg.get("enable_vestimenta", True)
                    self.alert_config["late_opening"] = cfg.get("enable_apertura", True)
        except Exception as e:
            pass

    def process_frame(self, frame, frame_count):
        self._sync_alert_config(frame_count)
        """Procesa un fotograma y devuelve la imagen dibujada junto con las alertas disparadas"""
        alerts = []
        sessions_to_update = []
        annotated = frame.copy()
        now_ts = time.time()
        
        # Solo procesar detecciones pesadas cada N frames (Frame Skip)
        if frame_count % self.FRAME_SKIP == 0:
            
            # --- FASE 1: DETECCIÓN EN CASCADA ---
            result_base = self.model_base(frame, verbose=False, imgsz=640, conf=0.08)[0]
            detections_base_all = sv.Detections.from_ultralytics(result_base)
            
            # Filtrar solo Humanos reales: confianza >= 0.55 (evita sillas/sombras) y tamaño mínimo
            is_human = (detections_base_all.class_id == 0) & (detections_base_all.confidence >= 0.55)
            if np.any(is_human):
                valid_humans = []
                for i, (cid, conf, b) in enumerate(zip(detections_base_all.class_id, detections_base_all.confidence, detections_base_all.xyxy)):
                    if cid == 0 and conf >= 0.55:
                        bw = b[2] - b[0]
                        bh = b[3] - b[1]
                        if bh >= 60:
                            valid_humans.append(i)
                detections_base = detections_base_all[valid_humans] if len(valid_humans) > 0 else sv.Detections.empty()
            else:
                detections_base = sv.Detections.empty()
            
            # Capturar celulares con máxima precisión (COCO ID 67 -> remapeado a 5)
            # Ajuste: Si el celular está muy de frente/cerca (ocupa más del 12% de la pantalla), bajar el umbral para reacción rápida
            phones_idx = []
            for i, (cid, conf, bbox) in enumerate(zip(detections_base_all.class_id, detections_base_all.confidence, detections_base_all.xyxy)):
                if cid == 67:
                    bw = bbox[2] - bbox[0]
                    if bw > self.width * 0.12: # Celular muy cerca de la cámara
                        if conf >= 0.04:
                            phones_idx.append(i)
                    else: # Celular lejos (perfil / escritorio)
                        if conf >= 0.12:
                            phones_idx.append(i)
                            
            phones_base = detections_base_all[phones_idx] if len(phones_idx) > 0 else sv.Detections.empty()
            if len(phones_base) > 0:
                phones_base.class_id = np.array([5] * len(phones_base))
            
            # Identificar periféricos de computadora (Mouse ID 64, Teclado ID 66, Laptop ID 63) para suprimir falsos positivos
            peripherals = detections_base_all[np.isin(detections_base_all.class_id, [62, 63, 64, 66])]
            
            # Capturar alimentos y bebidas reales con COCO (39: bottle, 40: glass, 41: cup, 46-55: comida)
            food_coco_ids = [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
            food_mask = np.isin(detections_base_all.class_id, food_coco_ids) & (detections_base_all.confidence >= 0.18)
            food_base = detections_base_all[food_mask]
            
            # Filtrar si colisiona con mouse/periféricos o si tiene tamaño de torso/espalda humana
            if len(food_base) > 0:
                valid_food = []
                for i, fbox in enumerate(food_base.xyxy):
                    fw = fbox[2] - fbox[0]
                    fh = fbox[3] - fbox[1]
                    # Descartar si es muy grande (pantalla completa) o si está en la esquina del mouse
                    if (fbox[0] < 220 and fbox[1] > self.height * 0.5) or fw > self.width * 0.70 or fh > self.height * 0.70:
                        continue
                    is_peripheral = False
                    for pbox in peripherals.xyxy:
                        if max(fbox[0], pbox[0]) < min(fbox[2], pbox[2]) and max(fbox[1], pbox[1]) < min(fbox[3], pbox[3]):
                            is_peripheral = True
                            break
                    if not is_peripheral:
                        valid_food.append(i)
                food_base = food_base[valid_food] if len(valid_food) > 0 else sv.Detections.empty()
                
            if len(food_base) > 0:
                food_base.class_id = np.array([4] * len(food_base))
            
            # Inferencia con modelo especialista en vestimenta y accesorios (v5)
            result_custom = self.model_custom(frame, verbose=False, imgsz=640, conf=0.03)[0]
            detections_custom_raw = sv.Detections.from_ultralytics(result_custom)

            # Inferencia con modelo especialista en pantallas y billetes (v7)
            result_screens = self.model_screens(frame, verbose=False, imgsz=640, conf=0.005)[0]
            detections_screens_raw = sv.Detections.from_ultralytics(result_screens)
            v7_boxes = []
            v7_classes = []
            v7_confs = []
            
            # Detección de Billetes v7 — APAGADA. Ahora usa v9.


            # Detección de Pantallas No Autorizadas
            # Usar modelo base (COCO) para saber exactamente dónde hay monitores reales y evitar cajas de cartón
            monitores_reales = detections_base_all[np.isin(detections_base_all.class_id, [62, 63])].xyxy
            
            for b in monitores_reales:
                if b[0] < self.width * 0.65:
                    bx1, by1, bx2, by2 = map(int, b)
                    # Filtro de tamaño: los monitores reales no ocupan la mitad de la cámara
                    bw, bh = bx2 - bx1, by2 - by1
                    if bw > self.width * 0.45 or bh > self.height * 0.45:
                        continue
                        
                    crop_s = frame[max(0, by1):min(self.height, by2), max(0, bx1):min(self.width, bx2)]
                    if crop_s.size > 0:
                        hsv_s = cv2.cvtColor(crop_s, cv2.COLOR_BGR2HSV)
                        dark_ratio = np.sum(hsv_s[:, :, 2] < 70) / float(crop_s.shape[0] * crop_s.shape[1])
                        mean_v = np.mean(hsv_s[:, :, 2])
                        # Validacion 1: Matemática (Oscuridad)
                        if dark_ratio >= 0.14 and mean_v < 145:
                            v7_boxes.append([bx1, by1, bx2, by2])
                            v7_classes.append(8)
                            v7_confs.append(0.92)

            for cid, conf, bbox in zip(detections_screens_raw.class_id, detections_screens_raw.confidence, detections_screens_raw.xyxy):
                if cid == 8 and bbox[0] < self.width * 0.65:
                    # Filtro de tamaño para v7 también
                    bw = bbox[2] - bbox[0]
                    bh = bbox[3] - bbox[1]
                    if bw > self.width * 0.45 or bh > self.height * 0.45:
                        continue
                        
                    # Validacion 2: Si el modelo v7 tiene confianza alta (>30%), lo aceptamos.
                    # Si tiene confianza baja (8% - 30%), SOLO lo aceptamos si está dentro de un monitor real de COCO.
                    if conf >= 0.30:
                        v7_boxes.append(bbox)
                        v7_classes.append(8)
                        v7_confs.append(conf)
                    elif conf >= 0.08:
                        for mb in monitores_reales:
                            # Checar intersección
                            if max(bbox[0], mb[0]) < min(bbox[2], mb[2]) and max(bbox[1], mb[1]) < min(bbox[3], mb[3]):
                                v7_boxes.append(bbox)
                                v7_classes.append(8)
                                v7_confs.append(conf)
                                break

            if len(v7_boxes) > 0:
                detections_screens = sv.Detections(
                    xyxy=np.array(v7_boxes),
                    class_id=np.array(v7_classes),
                    confidence=np.array(v7_confs)
                )
            else:
                detections_screens = sv.Detections.empty()
            
            # Inferencia con modelo especialista en Arreglo Personal y Billetes (v8)
            result_v8 = self.model_grooming_cash(frame, verbose=False, imgsz=640, conf=0.05)[0]
            detections_v8_raw = sv.Detections.from_ultralytics(result_v8)
            v8_boxes = []
            v8_classes = []
            v8_confs = []
            if len(detections_v8_raw) > 0:
                for cid, conf, bbox in zip(detections_v8_raw.class_id, detections_v8_raw.confidence, detections_v8_raw.xyxy):
                    # Clase 0: arreglo_personal
                    if cid == 0 and conf >= 0.25:
                        # CREAMOS UN RECUADRO PEQUEÑO (TIPO CELULAR) CENTRADO EN EL ROSTRO
                        # Y lo desplazamos un poco hacia abajo para que la etiqueta NO se empalme con la etiqueta morada del tracker.
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                        nx1 = x1 + (w * 0.20)
                        nx2 = x2 - (w * 0.20)
                        ny1 = y1 + (h * 0.08) # Bajamos el techo un 8% para separar la etiqueta
                        ny2 = y1 + (h * 0.35) # El recuadro termina a la altura del pecho
                        
                        v8_boxes.append([nx1, ny1, nx2, ny2])
                        v8_classes.append(10)
                        v8_confs.append(conf)
                    # Billetes en v8 APAGADO. Ahora usa v9.
                            
            if len(v8_boxes) > 0:
                detections_v8 = sv.Detections(
                    xyxy=np.array(v8_boxes),
                    class_id=np.array(v8_classes),
                    confidence=np.array(v8_confs)
                )
            else:
                detections_v8 = sv.Detections.empty()
                
            # --- NUEVO MODELO v9: Billetes, Llaves y Situación Atípica ---
            result_v9 = self.model_v9(frame, verbose=False, imgsz=640, conf=0.02)[0]
            detections_v9_raw = sv.Detections.from_ultralytics(result_v9)
            v9_boxes = []
            v9_classes = []
            v9_confs = []
            if len(detections_v9_raw) > 0:
                for cid, conf, bbox in zip(detections_v9_raw.class_id, detections_v9_raw.confidence, detections_v9_raw.xyxy):
                    # Clase 0: llaves -> sistema clase 6
                    if cid == 0 and conf >= 0.15:
                        v9_boxes.append(bbox)
                        v9_classes.append(6)
                        v9_confs.append(conf)
                    # Clase 1: billetes -> sistema clase 2
                    elif cid == 1:
                        req_conf = 0.06 # 6% de base, equilibrio perfecto.
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        
                        # Evitar el "falso positivo gigante de la pared/escritorio" (donde la caja ocupa casi toda la pantalla)
                        # Pero permitir que se acerque a la cámara (hasta un 40% del ancho de la pantalla, que es muchísimo)
                        if bw > self.width * 0.40 or bh > self.height * 0.40:
                            continue
                            
                        # Filtro Anti-Monitor: Si el centro del supuesto billete está adentro de la pantalla, es alucinación del color negro
                        bcx = bbox[0] + bw / 2.0
                        bcy = bbox[1] + bh / 2.0
                        is_on_monitor = False
                        for mb in monitores_reales:
                            # Expandimos un poco el área del monitor para atrapar recuadros sueltos
                            mx1, my1, mx2, my2 = mb[0] - 20, mb[1] - 20, mb[2] + 20, mb[3] + 20
                            if mx1 < bcx < mx2 and my1 < bcy < my2:
                                is_on_monitor = True
                                break
                        if is_on_monitor:
                            continue
                            
                        # Filtro Anti-Pared/Cajas: Los billetes deben estar siendo sostenidos por la persona.
                        # Verificamos si el billete se superpone con la caja delimitadora de alguna persona (clase 0 en COCO)
                        personas = detections_base_all[detections_base_all.class_id == 0].xyxy
                        is_held_by_person = False
                        for pb in personas:
                            # Expandimos la caja de la persona un poco para incluir las manos estiradas
                            px1, py1, px2, py2 = pb[0] - 80, pb[1] - 80, pb[2] + 80, pb[3] + 80
                            if max(bbox[0], px1) < min(bbox[2], px2) and max(bbox[1], py1) < min(bbox[3], py2):
                                is_held_by_person = True
                                break
                                
                        if not is_held_by_person:
                            continue # Si el billete está en la pared o caja, ignorarlo
                            
                        if conf >= req_conf:
                            v9_boxes.append(bbox)
                            v9_classes.append(2)
                            v9_confs.append(conf)
                    # Clase 2: situacion_atipica -> sistema clase 11
                    elif cid == 2 and conf >= 0.15: # Umbral equilibrado (15%) para evitar que dispare por error al estar sentado
                        v9_boxes.append(bbox)
                        v9_classes.append(11)
                        v9_confs.append(conf)
                        
            if len(v9_boxes) > 0:
                detections_v9 = sv.Detections(
                    xyxy=np.array(v9_boxes),
                    class_id=np.array(v9_classes),
                    confidence=np.array(v9_confs)
                )
            else:
                detections_v9 = sv.Detections.empty()
            # Umbrales calibrados para máxima sensibilidad en comida, billetes, celular, audífonos y pantallas
            CONF_MAP = {
                2: 0.05, # billetes (captura instantánea y ultra sensible)
                4: 0.02, # comida: ULTRA SENSIBLE (2%) para detectar cualquier objeto que se lleve a la boca
                5: 0.12, # celular (alta sensibilidad)
                6: 0.99, # llaves (deshabilitado a petición del usuario)
                7: 0.08, # audifonos_diadema (captura instantánea y ultra sensible)
                8: 0.01, # pagina_no_autorizada (YouTube, Facebook, TikTok, Instagram)
            }
            
            if len(detections_custom_raw) > 0:
                valid_idx = []
                for i, (cid, conf, bbox) in enumerate(zip(detections_custom_raw.class_id, detections_custom_raw.confidence, detections_custom_raw.xyxy)):
                    # Omitir clases de uniforme crudas del modelo v5 (0: pantalon, 1: playera, 3: zapatos)
                    # Y omitir Billetes (2) porque ahora lo hace v9.
                    if cid in [0, 1, 2, 3]:
                        continue
                    req_conf = CONF_MAP.get(cid, 0.20)
                    # Comida: chetos, botellas, paletas (descartar si es el mouse en esquina inf izq, techo o tamaño excesivo)
                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        # Relajamos el tamaño al 35% para que detecte botellas o comida grande cerca de la cámara
                        if bw > self.width * 0.35 or bh > self.height * 0.35:
                            continue 
                            
                        # "Sólo a la boca": El centro de la comida debe estar en el 35% superior del cuerpo
                        is_at_mouth = False
                        fcx = bbox[0] + (bw / 2.0)
                        fcy = bbox[1] + (bh / 2.0)
                        personas = detections_base_all[detections_base_all.class_id == 0].xyxy
                        for pb in personas:
                            # 50% superior y margen X expandido para asegurar que detecta "CUALQUIER COSA" llevada a la cara
                            head_y_limit = pb[1] + (pb[3] - pb[1]) * 0.50
                            if (pb[0] - 50) < fcx < (pb[2] + 50) and pb[1] < fcy < head_y_limit:
                                is_at_mouth = True
                                break
                        if not is_at_mouth:
                            continue
                    # Billetes: descartar si está en el monitor izquierdo o tamaño gigante (> 75% pantalla)
                    if cid == 2:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        bcx = (bbox[0] + bbox[2]) / 2.0
                        if bw > self.width * 0.75 or bh > self.height * 0.75:
                            continue
                    # Celular: reaccionar más rápido si está muy de frente/cerca a la cámara
                    if cid == 5:
                        bw = bbox[2] - bbox[0]
                        if bw > self.width * 0.12:
                            req_conf = 0.04 # Mucho más sensible de cerca
                            
                    if conf >= req_conf:
                        valid_idx.append(i)
                detections_custom = detections_custom_raw[valid_idx] if len(valid_idx) > 0 else sv.Detections.empty()
            else:
                detections_custom = sv.Detections.empty()
            
            # Inferencia Multi-Escala con Zoom Óptico y Validador de 3 Capas de Billetes en Manos/Torso
            zoom_boxes = []
            zoom_classes = []
            zoom_confs = []
            if len(detections_base) > 0:
                for pbox in detections_base.xyxy:
                    px1, py1, px2, py2 = map(int, pbox)
                    pw, ph = px2 - px1, py2 - py1
                    
                    # Extraer recuadro frontal (cabeza para audífonos, manos para billetes/celular)
                    zx1 = max(0, px1 - int(pw * 0.15))
                    zy1 = max(0, py1 - int(ph * 0.05))
                    zx2 = min(self.width, px1 + int(pw * 0.85))
                    zy2 = min(self.height, py1 + int(ph * 0.85))
                    
                    crop_zoom = frame[zy1:zy2, zx1:zx2]
                    if crop_zoom.size > 0 and crop_zoom.shape[0] >= 50 and crop_zoom.shape[1] >= 50:
                        res_zoom = self.model_custom(crop_zoom, verbose=False, imgsz=640, conf=0.03)[0]
                        if len(res_zoom.boxes) > 0:
                            for zbox in res_zoom.boxes:
                                zcid = int(zbox.cls[0])
                                zconf = float(zbox.conf[0])
                                # Audífonos de diadema en zoom (ultra sensible con v5)
                                if zcid == 7 and zconf >= 0.15:
                                    zb = zbox.xyxy[0].cpu().numpy()
                                    gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]
                                    if (gx2 - gx1) < self.width * 0.40 and (gy2 - gy1) < self.height * 0.40:
                                        zoom_boxes.append([gx1, gy1, gx2, gy2])
                                        zoom_classes.append(7)
                                        zoom_confs.append(zconf)
                                # Celular en manos/torso al estar sentado (v5)
                                elif zcid == 5 and zconf >= 0.25:
                                    zb = zbox.xyxy[0].cpu().numpy()
                                    gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]
                                    # Filtro de tamaño: un celular no debería ser enorme
                                    if (gx2 - gx1) < self.width * 0.30 and (gy2 - gy1) < self.height * 0.30:
                                        zoom_boxes.append([gx1, gy1, gx2, gy2])
                                        zoom_classes.append(5)
                                        zoom_confs.append(zconf)
                                # Comida / Bebida en manos o boca al estar sentado (v5)
                                elif zcid == 4 and zconf >= 0.15:
                                    zb = zbox.xyxy[0].cpu().numpy()
                                    gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]
                                    zoom_boxes.append([gx1, gy1, gx2, gy2])
                                    zoom_classes.append(4)
                                    zoom_confs.append(zconf)
                                                
                        # Detección complementaria de celular y COMIDA/BEBIDA en manos/torso con modelo base (COCO)
                        res_zoom_base = self.model_base(crop_zoom, verbose=False, imgsz=640, conf=0.08)[0]
                        if len(res_zoom_base.boxes) > 0:
                            for zbox in res_zoom_base.boxes:
                                zcid = int(zbox.cls[0])
                                zconf = float(zbox.conf[0])
                                # Celular base
                                if zcid == 67 and zconf >= 0.25:
                                    zb = zbox.xyxy[0].cpu().numpy()
                                    gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]
                                    # Filtro de tamaño: un celular no debería ocupar más del 30% del recuadro
                                    if (gx2 - gx1) < self.width * 0.30 and (gy2 - gy1) < self.height * 0.30:
                                        zoom_boxes.append([gx1, gy1, gx2, gy2])
                                        zoom_classes.append(5)
                                        zoom_confs.append(zconf)
                                # Bebidas y comida base (39: bottle, 40: glass, 41: cup, 46-55: food)
                                elif zcid in [39, 40, 41, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55] and zconf >= 0.02:
                                    zb = zbox.xyxy[0].cpu().numpy()
                                    gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]
                                    zoom_boxes.append([gx1, gy1, gx2, gy2])
                                    zoom_classes.append(4)
                                    zoom_confs.append(zconf)

                                    

                                            

            if len(zoom_boxes) > 0:
                detections_zoom = sv.Detections(
                    xyxy=np.array(zoom_boxes),
                    class_id=np.array(zoom_classes),
                    confidence=np.array(zoom_confs)
                )
            else:
                detections_zoom = sv.Detections.empty()

            # Inferencia y Análisis de Código de Vestimenta (Playera blanca/negra y Pantalón negro)
            uniform_boxes = []
            uniform_classes = []
            uniform_confs = []
            
            if len(detections_base) > 0:
                for pbox in detections_base.xyxy:
                    px1, py1, px2, py2 = map(int, pbox)
                    pw, ph = px2 - px1, py2 - py1
                    
                    # Evaluar cuando la persona está de pie o visible en el módulo
                    is_standing = (ph / max(1, pw)) >= 1.45 and ph > self.height * 0.28
                    
                    # Torso (Playera)
                    is_standing = ph > (pw * 1.6)
                    
                    # CROP MÁS ESTRICTO AL CENTRO DEL CUERPO (evitar que agarre la pared blanca del fondo)
                    ty1, ty2 = max(0, py1 + int(ph * 0.15)), min(self.height, py1 + int(ph * 0.45))
                    tx1, tx2 = max(0, px1 + int(pw * 0.30)), min(self.width, px2 - int(pw * 0.30))
                    t_crop = frame[ty1:ty2, tx1:tx2]
                    
                    ly1, ly2 = max(0, py1 + int(ph * 0.55)), min(self.height, py1 + int(ph * 0.90))
                    lx1, lx2 = max(0, px1 + int(pw * 0.25)), min(self.width, px2 - int(pw * 0.25))
                    l_crop = frame[ly1:ly2, lx1:lx2]
                    
                    if t_crop.size > 0 and l_crop.size > 0:
                        t_hsv = cv2.cvtColor(t_crop, cv2.COLOR_BGR2HSV)
                        l_hsv = cv2.cvtColor(l_crop, cv2.COLOR_BGR2HSV)
                        
                        # Para evitar falsos positivos con la pared blanca, analizamos sólo el 50% central del torso
                        th, tw = t_crop.shape[:2]
                        center_t_hsv = t_hsv[int(th*0.25):int(th*0.75), int(tw*0.25):int(tw*0.75)]
                        if center_t_hsv.size > 0:
                            t_white = np.sum((center_t_hsv[:, :, 1] < 40) & (center_t_hsv[:, :, 2] > 180)) / float(center_t_hsv.shape[0] * center_t_hsv.shape[1])
                            t_black = np.sum(center_t_hsv[:, :, 2] < 85) / float(center_t_hsv.shape[0] * center_t_hsv.shape[1])
                        else:
                            t_white, t_black = 0, 0
                        
                        l_black = np.sum(l_hsv[:, :, 2] < 70) / float(l_crop.shape[0] * l_crop.shape[1])
                        l_blue = np.sum((l_hsv[:, :, 0] >= 90) & (l_hsv[:, :, 0] <= 130) & (l_hsv[:, :, 1] > 35)) / float(l_crop.shape[0] * l_crop.shape[1])
                        
                        # Si es negra, t_white será muy bajo. Si es blanca, t_white será muy alto (>55%).
                        has_white_shirt = (t_white >= 0.55)
                        has_black_pants = (l_black >= 0.45 and l_blue < 0.15)
                        
                        is_valid = has_white_shirt and has_black_pants
                            
                        # Si falta la prenda reglamentaria -> Se emite UNA SOLA etiqueta clara
                        if not is_valid:
                            uniform_boxes.append([px1, py1, px2, py2])
                            uniform_classes.append(9) # uniforme_incompleto
                            uniform_confs.append(0.92)
                        else:
                            uniform_boxes.append([px1, py1, px2, py2])
                            uniform_classes.append(14) # uniforme_completo
                            uniform_confs.append(0.92)

            if len(uniform_boxes) > 0:
                detections_uniform = sv.Detections(
                    xyxy=np.array(uniform_boxes),
                    class_id=np.array(uniform_classes),
                    confidence=np.array(uniform_confs)
                )
            else:
                detections_uniform = sv.Detections.empty()

            # Fusión de detecciones (Base + Custom v5 + Screens/Cash v7 + Grooming v8 + v9 + Zoom Óptico + Uniforme)
            to_merge = [d for d in [detections_custom, detections_screens, detections_v8, detections_v9, detections_zoom, phones_base, food_base, detections_uniform] if len(d) > 0]
            clean_merge = []
            for d in to_merge:
                if len(d) > 0:
                    clean_merge.append(sv.Detections(
                        xyxy=d.xyxy,
                        confidence=d.confidence if d.confidence is not None else np.ones(len(d.xyxy)),
                        class_id=d.class_id
                    ))
            if len(clean_merge) > 1:
                detections = sv.Detections.merge(clean_merge)
            elif len(clean_merge) == 1:
                detections = clean_merge[0]
            else:
                detections = sv.Detections.empty()
            
            # Desambiguación Estricta:
            # 1. Si colisionan Billetes y Comida en la misma zona, los Billetes tienen prioridad absoluta
            if len(detections) > 0 and 2 in detections.class_id and 4 in detections.class_id:
                keep_idx = []
                for idx, (cid, conf, b) in enumerate(zip(detections.class_id, detections.confidence, detections.xyxy)):
                    if cid == 2:
                        # Los billetes nunca se suprimen por comida, ya que el modelo v9 está muy entrenado para esto.
                        keep_idx.append(idx)
                    elif cid == 4:
                        # Suprimir la comida SIEMPRE si choca con billetes, sin importar la confianza
                        suppressed = any(
                            (ocid == 2 and max(b[0], ob[0]) < min(b[2], ob[2]) and max(b[1], ob[1]) < min(b[3], ob[3]))
                            for ocid, ob in zip(detections.class_id, detections.xyxy)
                        )
                        if not suppressed:
                            keep_idx.append(idx)
                    elif cid == 10:
                        # Suprimir maquillaje si colisiona con billetes
                        overlap = any(
                            (ocid == 2 and max(b[0], ob[0]) < min(b[2], ob[2]) and max(b[1], ob[1]) < min(b[3], ob[3]))
                            for ocid, ob in zip(detections.class_id, detections.xyxy)
                        )
                        if not overlap:
                            keep_idx.append(idx)
                    else:
                        keep_idx.append(idx)
                detections = detections[keep_idx] if len(keep_idx) > 0 else sv.Detections.empty()
            elif len(detections) > 0 and 2 in detections.class_id:
                # Si solo hay billetes, suprimir maquillaje que se solape
                keep_idx = []
                for idx, (cid, b) in enumerate(zip(detections.class_id, detections.xyxy)):
                    if cid == 10:
                        overlap = any(
                            (ocid == 2 and max(b[0], ob[0]) < min(b[2], ob[2]) and max(b[1], ob[1]) < min(b[3], ob[3]))
                            for ocid, ob in zip(detections.class_id, detections.xyxy)
                        )
                        if not overlap:
                            keep_idx.append(idx)
                    else:
                        keep_idx.append(idx)
                detections = detections[keep_idx] if len(keep_idx) > 0 else sv.Detections.empty()

            # 2. Si se detectó Comida/Bebida en mano/boca, suprimir Arreglo Personal para no confundir comer con maquillar
            if len(detections) > 0 and 4 in detections.class_id and 10 in detections.class_id:
                food_boxes = [b for cid, b in zip(detections.class_id, detections.xyxy) if cid == 4]
                keep_idx = []
                for idx, (cid, b) in enumerate(zip(detections.class_id, detections.xyxy)):
                    if cid == 10:
                        overlap = any(max(b[0], fb[0]) < min(b[2], fb[2]) and max(b[1], fb[1]) < min(b[3], fb[3]) for fb in food_boxes)
                        if not overlap:
                            keep_idx.append(idx)
                    else:
                        keep_idx.append(idx)
                detections = detections[keep_idx] if len(keep_idx) > 0 else sv.Detections.empty()
            
            # --- FILTRO MAESTRO VISUAL ---
            # Si una alerta está apagada (False) en alert_config, quitamos su recuadro para que no salga en pantalla
            MAPA_CLASES_ALERTAS = {
                2: "incidencia_dinero",
                4: "incidencia_alimentos",
                5: "incidencia_celular",
                6: "incidencia_llaves",
                7: "incidencia_audifonos",
                8: "incidencia_paginas",
                9: "incidencia_uniforme",
                14: "incidencia_uniforme", # Mapeamos uniforme completo a la misma bandera para que se oculte
                10: "incidencia_arreglo_personal",
                11: "incidencia_situacion_atipica"
            }
            if len(detections) > 0:
                keep_visual = []
                for idx, cid in enumerate(detections.class_id):
                    # El humano (0) siempre se queda, las demás según config
                    if cid in MAPA_CLASES_ALERTAS and not self.alert_config.get(MAPA_CLASES_ALERTAS[cid], True):
                        continue
                    keep_visual.append(idx)
                detections = detections[keep_visual] if len(keep_visual) > 0 else sv.Detections.empty()


            # --- FASE 3: TRACKING Y RE-IDENTIFICACIÓN ---
            detections_tracked = self.tracker.update_with_detections(detections_base)
            
            if detections_tracked.tracker_id is not None:
                new_tracker_ids = []
                current_time_epoch = time.time()
                active_mapped_ids = set()
                
                for tid, bbox in zip(detections_tracked.tracker_id, detections_tracked.xyxy):
                    signature = extract_color_signature(frame, bbox)
                    mapped_tid = self.id_map.get(tid, tid)
                    
                    cur_cx = (bbox[0] + bbox[2]) / 2.0
                    cur_cy = (bbox[1] + bbox[3]) / 2.0
                    cur_w = bbox[2] - bbox[0]
                    cur_h = bbox[3] - bbox[1]
                    
                    if mapped_tid not in self.persistent_identities:
                        best_match_id = None
                        best_match_score = -1
                        for old_tid, old_data in self.persistent_identities.items():
                            if old_tid in active_mapped_ids: continue
                            time_diff = current_time_epoch - old_data['last_seen']
                            if time_diff > 300: continue
                            
                            ocx, ocy = old_data.get('last_center', (cur_cx, cur_cy))
                            old_bbox = old_data.get('last_bbox', bbox)
                            old_w = old_bbox[2] - old_bbox[0]
                            
                            dist = np.hypot(cur_cx - ocx, cur_cy - ocy)
                            dx = abs(cur_cx - ocx)
                            
                            # Solapamiento horizontal en X (mismo escritorio o columna de atención)
                            x_inter = max(0, min(bbox[2], old_bbox[2]) - max(bbox[0], old_bbox[0]))
                            x_overlap = x_inter / float(max(1, min(cur_w, old_w)))
                            
                            color_score = 0
                            if signature is not None and old_data.get('signature') is not None:
                                color_score = cv2.compareHist(signature, old_data['signature'], cv2.HISTCMP_CORREL)
                            
                            # Criterio 1: Mismo puesto de trabajo al pararse/sentarse (columna X coincidente y tiempo < 60s)
                            is_same_workstation = (x_overlap >= 0.40 and dx < 140 and time_diff < 60.0)
                            
                            # Criterio 2: Movimiento cercano continuo en la escena (< 220 px y tiempo < 10s)
                            is_nearby_continuation = (dist < 220 and time_diff < 10.0)
                            
                            # Criterio 3: Re-identificación por vestimenta (> 0.60 de correlación)
                            is_color_match = (color_score > 0.60)
                            
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
                        'signature': signature,
                        'last_center': (cur_cx, cur_cy),
                        'last_bbox': bbox,
                        'last_seen': current_time_epoch
                    }
                    active_mapped_ids.add(mapped_tid)
                    new_tracker_ids.append(mapped_tid)
                    
                detections_tracked.tracker_id = np.array(new_tracker_ids)
                
            # --- FASE 2: MOTOR DE REGLAS CON BUFFER DE TOLERANCIA ---
            
            # 1. Detección de Celular
            if 5 in detections.class_id:
                if "celular" not in self.detection_start_times:
                    self.detection_start_times["celular"] = now_ts
                self.last_seen_times["celular"] = now_ts
                if now_ts - self.detection_start_times["celular"] >= self.DEBOUNCE_SECONDS:
                    if now_ts - self.last_alert_times.get("celular", 0) >= self.ALERT_COOLDOWN:
                        for c_id in self._get_culprits(5, detections, detections_tracked):
                            alerts.append(("incidencia_celular", f"ALERTA: {c_id} detectado(a) usando dispositivo móvil (Celular).", "warning"))
                        self.last_alert_times["celular"] = now_ts
            else:
                if "celular" in self.last_seen_times and now_ts - self.last_seen_times["celular"] > 1.0:
                    self.detection_start_times.pop("celular", None)
                    self.last_seen_times.pop("celular", None)

            # 2. Uso de Audífonos de Diadema (Alerta cuando los traen puestos)
            if 7 in detections.class_id:
                if "audifonos" not in self.detection_start_times:
                    self.detection_start_times["audifonos"] = now_ts
                self.last_seen_times["audifonos"] = now_ts
                if now_ts - self.detection_start_times["audifonos"] >= self.DEBOUNCE_SECONDS:
                    if now_ts - self.last_alert_times.get("audifonos", 0) >= self.ALERT_COOLDOWN:
                        for c_id in self._get_culprits(7, detections, detections_tracked):
                            alerts.append(("incidencia_audifonos", f"ALERTA: {c_id} detectado(a) usando audífonos de diadema.", "warning"))
                        self.last_alert_times["audifonos"] = now_ts
            else:
                if "audifonos" in self.last_seen_times and now_ts - self.last_seen_times["audifonos"] > 1.0:
                    self.detection_start_times.pop("audifonos", None)
                    self.last_seen_times.pop("audifonos", None)
                        
            # 3. Dinero / Billetes (v9)
            if 2 in detections.class_id:
                if "dinero" not in self.detection_start_times:
                    self.detection_start_times["dinero"] = now_ts
                self.last_seen_times["dinero"] = now_ts
                if now_ts - self.detection_start_times["dinero"] >= self.DEBOUNCE_SECONDS:
                    if now_ts - self.last_alert_times.get("dinero", 0) >= self.ALERT_COOLDOWN:
                        for c_id in self._get_culprits(2, detections, detections_tracked):
                            alerts.append(("incidencia_dinero", f"ALERTA: {c_id} detectado con manejo de billetes o efectivo.", "warning"))
                        self.last_alert_times["dinero"] = now_ts
            else:
                if "dinero" in self.last_seen_times and now_ts - self.last_seen_times["dinero"] > 1.0:
                    self.detection_start_times.pop("dinero", None)
                    self.last_seen_times.pop("dinero", None)

            # 4. Comida/Bebidas (Base + Custom)
            if 4 in detections.class_id:
                if "alimentos" not in self.detection_start_times:
                    self.detection_start_times["alimentos"] = now_ts
                self.last_seen_times["alimentos"] = now_ts
                if now_ts - self.detection_start_times["alimentos"] >= self.DEBOUNCE_SECONDS:
                    if now_ts - self.last_alert_times.get("alimentos", 0) >= self.ALERT_COOLDOWN:
                        for c_id in self._get_culprits(4, detections, detections_tracked):
                            alerts.append(("incidencia_alimentos", f"ALERTA: {c_id} detectado(a) consumiendo alimentos/bebidas.", "warning"))
                        self.last_alert_times["alimentos"] = now_ts
            else:
                if "alimentos" in self.last_seen_times and now_ts - self.last_seen_times["alimentos"] > 1.0:
                    self.detection_start_times.pop("alimentos", None)
                    self.last_seen_times.pop("alimentos", None)

            # 5. Páginas no autorizadas (Screens v7)
            if 8 in detections.class_id:
                if "paginas" not in self.detection_start_times:
                    self.detection_start_times["paginas"] = now_ts
                self.last_seen_times["paginas"] = now_ts
                if now_ts - self.detection_start_times["paginas"] >= self.DEBOUNCE_SECONDS:
                    if now_ts - self.last_alert_times.get("paginas", 0) >= self.ALERT_COOLDOWN:
                        for c_id in self._get_culprits(8, detections, detections_tracked):
                            alerts.append(("incidencia_paginas", f"ALERTA: {c_id} con ingreso a páginas / redes no autorizadas en monitor.", "warning"))
                        self.last_alert_times["paginas"] = now_ts
            else:
                if "paginas" in self.last_seen_times and now_ts - self.last_seen_times["paginas"] > 1.0:
                    self.detection_start_times.pop("paginas", None)
                    self.last_seen_times.pop("paginas", None)
            
            # 6. Arreglo Personal (Grooming v8)
            if 10 in detections.class_id:
                if "arreglo" not in self.detection_start_times:
                    self.detection_start_times["arreglo"] = now_ts
                self.last_seen_times["arreglo"] = now_ts
                if now_ts - self.detection_start_times["arreglo"] >= 1.5:
                    if now_ts - self.last_alert_times.get("arreglo", 0) >= self.ALERT_COOLDOWN:
                        for c_id in self._get_culprits(10, detections, detections_tracked):
                            alerts.append(("incidencia_arreglo_personal", f"ALERTA: {c_id} realizando arreglo personal / maquillaje.", "warning"))
                        self.last_alert_times["arreglo"] = now_ts
            else:
                if "arreglo" in self.last_seen_times and now_ts - self.last_seen_times["arreglo"] > 1.0:
                    self.detection_start_times.pop("arreglo", None)
                    self.last_seen_times.pop("arreglo", None)
            
            # 7. Uniforme (Heurística)
            if 9 in detections.class_id:
                if "uniforme" not in self.detection_start_times:
                    self.detection_start_times["uniforme"] = now_ts
                self.last_seen_times["uniforme"] = now_ts
                if now_ts - self.detection_start_times["uniforme"] >= self.DEBOUNCE_SECONDS:
                    if now_ts - self.last_alert_times.get("uniforme", 0) >= self.ALERT_COOLDOWN:
                        for c_id in self._get_culprits(9, detections, detections_tracked):
                            alerts.append(("incidencia_uniforme", f"ALERTA: {c_id} con código de vestimenta incompleto.", "warning"))
                        self.last_alert_times["uniforme"] = now_ts
            else:
                if "uniforme" in self.last_seen_times and now_ts - self.last_seen_times["uniforme"] > 1.0:
                    self.detection_start_times.pop("uniforme", None)
                    self.last_seen_times.pop("uniforme", None)
                    
            # 8. Llaves (v9)
            if 6 in detections.class_id:
                if "llaves" not in self.detection_start_times:
                    self.detection_start_times["llaves"] = now_ts
                self.last_seen_times["llaves"] = now_ts
                if now_ts - self.detection_start_times["llaves"] >= self.DEBOUNCE_SECONDS:
                    if now_ts - self.last_alert_times.get("llaves", 0) >= self.ALERT_COOLDOWN:
                        for c_id in self._get_culprits(6, detections, detections_tracked):
                            alerts.append(("incidencia_llaves", f"ALERTA: {c_id} con llaves detectadas en área de atención.", "warning"))
                        self.last_alert_times["llaves"] = now_ts
            else:
                if "llaves" in self.last_seen_times and now_ts - self.last_seen_times["llaves"] > 1.0:
                    self.detection_start_times.pop("llaves", None)
                    self.last_seen_times.pop("llaves", None)

            # 9. Situación Atípica (v9 o persona de pie)
            # Evaluamos si hay clase 11 (v9) o si alguna persona detectada está parada
            alguien_parado = False
            if len(detections_base) > 0:
                atipica_boxes = []
                for pbox in detections_base.xyxy:
                    px1, py1, px2, py2 = map(int, pbox)
                    pw, ph = px2 - px1, py2 - py1
                    # Ajuste geométrico ESTRICTO para Situación Atípica (Completamente Parado):
                    # 1. Postura muy vertical (ratio >= 1.65)
                    # 2. Cabeza tocando el techo de la cámara (py1 < 10%)
                    if (py1 < self.height * 0.10) or ((ph / max(1, pw)) >= 1.65 and ph > self.height * 0.40):
                        alguien_parado = True
                        atipica_boxes.append(pbox)
                
                # Inyectar caja visual si la alerta está activa
                if alguien_parado and self.alert_config.get("incidencia_situacion_atipica", True):
                    if len(detections) > 0:
                        new_xyxy = np.vstack([detections.xyxy, np.array(atipica_boxes)])
                        new_class_id = np.concatenate([detections.class_id, np.array([11] * len(atipica_boxes))])
                        new_confidence = np.concatenate([detections.confidence, np.array([0.99] * len(atipica_boxes))])
                        detections = sv.Detections(xyxy=new_xyxy, class_id=new_class_id, confidence=new_confidence)
                    else:
                        detections = sv.Detections(
                            xyxy=np.array(atipica_boxes),
                            class_id=np.array([11] * len(atipica_boxes)),
                            confidence=np.array([0.99] * len(atipica_boxes))
                        )

            if 11 in detections.class_id or alguien_parado:
                if "atipica" not in self.detection_start_times:
                    self.detection_start_times["atipica"] = now_ts
                self.last_seen_times["atipica"] = now_ts
                if now_ts - self.detection_start_times["atipica"] >= self.DEBOUNCE_SECONDS:
                    if now_ts - self.last_alert_times.get("atipica", 0) >= self.ALERT_COOLDOWN:
                        for c_id in self._get_culprits(11, detections, detections_tracked):
                            alerts.append(("incidencia_situacion_atipica", f"ALERTA: {c_id} detectado en Situación atípica (Persona de pie / Movimiento anómalo).", "warning"))
                        self.last_alert_times["atipica"] = now_ts
            else:
                if "atipica" in self.last_seen_times and now_ts - self.last_seen_times["atipica"] > 1.0:
                    self.detection_start_times.pop("atipica", None)
                    self.last_seen_times.pop("atipica", None)
            
            # --- FASE 4: ANALISIS ESPACIAL Y FLUJOS (SITUACIONES ATIPICAS) ---
            crossed_in, crossed_out = self.line_zone.trigger(detections=detections_tracked)
            self.in_count = self.line_zone.in_count
            self.out_count = self.line_zone.out_count
            
            # --- NUEVA ALERTA: SEGURIDAD (TAILGATING) ---
            if not hasattr(self, 'recent_entries'):
                self.recent_entries = [] # lista de (timestamp, tracker_id, bbox)
                
            # Limpiar entradas viejas (> 3.5 segundos) para evitar acumular basura
            self.recent_entries = [entry for entry in self.recent_entries if now_ts - entry[0] <= 10.0]
            
            tailgating_boxes = []
            if crossed_in is not None and len(crossed_in) > 0:
                for i, (is_in, is_out) in enumerate(zip(crossed_in, crossed_out)):
                    if is_in or is_out:
                        tid = detections_tracked.tracker_id[i]
                        bbox = detections_tracked.xyxy[i]
                        
                        # Checar si ya hay otra persona diferente que entro hace menos de 3.5s
                        for old_ts, old_tid, old_bbox in self.recent_entries:
                            if old_tid != tid:
                                # Alarma de Tailgating! Dos IDs distintos entraron en < 3.5s
                                tailgating_boxes.append(bbox)
                                tailgating_boxes.append(old_bbox)
                                
                                if now_ts - self.last_alert_times.get("seguridad", 0) >= self.ALERT_COOLDOWN:
                                    alerts.append(("incidencia_seguridad", "ALERTA CRITICA: Acceso multiple no autorizado (Tailgating).", "critical"))
                                    self.last_alert_times["seguridad"] = now_ts
                        
                        # Agregar este nuevo cruce
                        self.recent_entries.append((now_ts, tid, bbox))
                        
            if len(tailgating_boxes) > 0:
                if len(detections) > 0:
                    detections = sv.Detections(
                        xyxy=np.vstack([detections.xyxy, tailgating_boxes]),
                        class_id=np.append(detections.class_id, [15]*len(tailgating_boxes)),
                        confidence=np.append(detections.confidence, [1.0]*len(tailgating_boxes))
                    )
                else:
                    detections = sv.Detections(
                        xyxy=np.array(tailgating_boxes),
                        class_id=np.array([15]*len(tailgating_boxes)),
                        confidence=np.array([1.0]*len(tailgating_boxes))
                    )
            
            fila_in = self.fila_zone.trigger(detections=detections_tracked)
            modulo_in = self.modulo_zone.trigger(detections=detections_tracked)
            
            self.fila_count = np.sum(fila_in)
            self.modulo_count = np.sum(modulo_in)
            
            # 8. Alerta de Saturación de Fila >= 4 personas
            if self.fila_count >= 4:
                if "saturacion" not in self.detection_start_times:
                    self.detection_start_times["saturacion"] = now_ts
                elif now_ts - self.detection_start_times["saturacion"] >= self.DEBOUNCE_SECONDS:
                    if "saturacion" not in self.triggered_alerts:
                        alerts.append(("incidencia_saturacion", "ALERTA: Fila saturada (4+ clientes en espera). Se requiere abrir módulo de apoyo.", "warning"))
                        self.triggered_alerts.add("saturacion")
            else:
                self.detection_start_times.pop("saturacion", None)
                if "saturacion" in self.triggered_alerts:
                    self.triggered_alerts.remove("saturacion")
                
            # Mantenimiento de Sesiones y Tracking
            if detections_tracked.tracker_id is not None:
                current_ids = set()
                current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                current_time_epoch = time.time()
                
                for tid, in_f, in_m in zip(detections_tracked.tracker_id, fila_in, modulo_in):
                    current_ids.add(tid)
                    
                    if tid not in self.track_history:
                        self.track_history[tid] = {
                            'first_seen': current_time_str,
                            'last_seen_epoch': current_time_epoch,
                            'entered_fila': current_time_str if in_f else None,
                            'entered_fila_epoch': current_time_epoch if in_f else None,
                            'left_fila': None,
                            'entered_modulo': current_time_str if in_m else None,
                            'entered_modulo_epoch': current_time_epoch if in_m else None,
                            'left_modulo': None,
                            'status': 'detectado',
                            'total_service_time': None
                        }
                        if not self.first_client_seen:
                            self.first_client_seen = True
                            is_late = False
                            cam_time_str = "N/A"
                            
                            # 1. Leer OCR de cámara (si está disponible y cargado)
                            if getattr(self, '_ocr_reader', None) is not None:
                                try:
                                    import re
                                    crop = frame[0:150, 0:600] 
                                    big = cv2.resize(crop, (0,0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
                                    gray_crop = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
                                    
                                    # Leer imagen procesada (permitiendo letras y números para que lea Wed y demás)
                                    res = self._ocr_reader.readtext(gray_crop, detail=0)
                                    for text in res:
                                        match = re.search(r'(\d{1,2})\s*:\s*(\d{2})', text)
                                        if match:
                                            h, m = int(match.group(1)), int(match.group(2))
                                            cam_time_str = f"{h:02d}:{m:02d}"
                                            if h > 9 or (h == 9 and m > 15):
                                                is_late = True
                                            else:
                                                is_late = False
                                            break
                                except:
                                    pass
                            
                            # 2. Leer hora local (solo si el OCR falló)
                            now = datetime.now()
                            if cam_time_str == "N/A":
                                if now.hour > 9 or (now.hour == 9 and now.minute > 15):
                                    is_late = True
                                    
                            if is_late:
                                source = f"Cámara ({cam_time_str})" if cam_time_str != "N/A" else f"Sistema ({now.strftime('%H:%M:%S')})"
                                alerts.append(("late_opening", f"Primer cliente detectado a las {source}. Apertura tardía.", "warning"))
                                self.late_opening_trigger_time = now_ts
                                self.late_opening_time_str = source
                                
                    session = self.track_history[tid]
                    session['last_seen_epoch'] = current_time_epoch
                    
                    if in_f and not session['entered_fila']:
                        session['entered_fila'] = current_time_str
                        session['entered_fila_epoch'] = current_time_epoch
                        session['status'] = 'en_fila'
                    if not in_f and session['entered_fila'] and not session['left_fila']:
                        session['left_fila'] = current_time_str

                    if in_m and not session['entered_modulo']:
                        session['entered_modulo'] = current_time_str
                        session['entered_modulo_epoch'] = current_time_epoch
                        session['status'] = 'en_modulo'
                    
                    # 10. Alerta de Tiempo de Atención Excedido en Módulo (>15s calibrados para demo)
                    if in_m and session.get('entered_modulo_epoch'):
                        mod_dur = current_time_epoch - session['entered_modulo_epoch']
                        if mod_dur >= 15:
                            if "atencion_excedida" not in self.triggered_alerts:
                                alerts.append(("excessive_service_time", "ALERTA: Tiempo de atención excedido (>15 min). Posible bloqueo en trámite.", "warning"))
                                self.triggered_alerts.add("atencion_excedida")

                    if not in_m and session['entered_modulo'] and not session['left_modulo']:
                        session['left_modulo'] = current_time_str
                        session['status'] = 'atendido'
                        fmt = "%Y-%m-%d %H:%M:%S"
                        t_in = datetime.strptime(session['entered_modulo'], fmt)
                        t_out = datetime.strptime(session['left_modulo'], fmt)
                        session['total_service_time'] = int((t_out - t_in).total_seconds())
                        alerts.append(("service_completed", f"Servicio completado en {session['total_service_time']}s", "info"))
                        sessions_to_update.append((int(tid), session.copy()))
                        
                tids_to_remove = []
                for tid_history, session_history in self.track_history.items():
                    if tid_history not in current_ids:
                        if current_time_epoch - session_history['last_seen_epoch'] > 60:
                            if session_history['entered_fila'] and not session_history['entered_modulo']:
                                session_history['status'] = 'abandonado'
                                if not session_history['left_fila']:
                                    session_history['left_fila'] = datetime.fromtimestamp(session_history['last_seen_epoch']).strftime("%Y-%m-%d %H:%M:%S")
                                if "abandono" not in self.triggered_alerts:
                                    alerts.append(("customer_abandonment", "Cliente se retiró sin ser atendido tras 60s", "warning"))
                                    self.triggered_alerts.add("abandono")
                            sessions_to_update.append((int(tid_history), session_history.copy()))
                            tids_to_remove.append(tid_history)
                for tid in tids_to_remove:
                    del self.track_history[tid]
                    
            self.last_detections = detections
            self.last_detections_tracked = detections_tracked
            
        else:
            # Reutilizar el cálculo del frame anterior para mantener fluidez
            detections = self.last_detections
            detections_tracked = self.last_detections_tracked
            
        # --- FASE 5: DIBUJADO VISUAL ---
        # (Difuminado de rostro comentado temporalmente para prueba)
        # if hasattr(detections_tracked, 'xyxy') and len(detections_tracked) > 0:
        #     head_xyxy = detections_tracked.xyxy.copy()
        #     head_xyxy[:, 3] = head_xyxy[:, 1] + (head_xyxy[:, 3] - head_xyxy[:, 1]) * 0.35
        #     head_detections = sv.Detections(xyxy=head_xyxy)
        #     annotated = self.blur_annotator.annotate(scene=annotated, detections=head_detections)
            
        CLASS_NAMES = {
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
            11: "situacion_atipica"
        }
        labels = []
        for class_id, confidence in zip(detections.class_id, detections.confidence):
            name = CLASS_NAMES.get(class_id, f"obj_{class_id}")
            tag = None
            if class_id == 5: tag = "celular"
            elif class_id == 7: tag = "audifonos"
            elif class_id == 2: tag = "dinero"
            elif class_id == 6: tag = "llaves"
            elif class_id == 4: tag = "alimentos"
            elif class_id == 8: tag = "paginas"
            elif class_id == 9: tag = "uniforme"
            elif class_id == 10: tag = "arreglo"
            elif class_id == 11: tag = "atipica"
            elif class_id == 13: tag = "audifonos"
            elif class_id == 14: tag = None
            elif class_id == 15: tag = "seguridad" 
            
            if class_id == 13: name = "audifonos_no_detectados"
            elif class_id == 7: name = "audifonos_diadema"
            elif class_id == 14: name = "uniforme_completo"
            
            if tag and (now_ts - self.last_alert_times.get(tag, 0) < 3.0):
                labels.append(f"{name} [TICKET EMITIDO]")
            elif tag and tag in self.detection_start_times:
                elapsed = now_ts - self.detection_start_times[tag]
                labels.append(f"{name} ({elapsed:.1f}s / {self.DEBOUNCE_SECONDS:.1f}s)")
            else:
                labels.append(f"{name} {confidence:.2f}")
            
        annotated = self.box_annotator.annotate(scene=annotated, detections=detections)
        annotated = self.label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
        
        if hasattr(detections_tracked, 'tracker_id') and detections_tracked.tracker_id is not None:
            annotated = self.box_annotator.annotate(scene=annotated, detections=detections_tracked)
            
            tracked_labels = []
            for tid, bbox in zip(detections_tracked.tracker_id, detections_tracked.xyxy):
                mapped_tid = self.id_map.get(tid, tid)
                str_mapped = str(mapped_tid)
                
                # Check if it's already a known face in this session memory
                if str_mapped in getattr(self, 'known_identities', {}):
                    tracked_labels.append(f"{self.known_identities[str_mapped]}")
                else:
                    # Attempt ArcFace biometric detection every few frames for unknown tracking IDs
                    if frame_count % 15 == 0:
                        try:
                            from biometric_engine import find_face_and_embed, identify_person_in_db, register_new_face
                            face_emb = find_face_and_embed(frame, bbox)
                            if face_emb:
                                db_id, db_name = identify_person_in_db(face_emb)
                                if db_id:
                                    # Match found in database!
                                    self.known_identities[str_mapped] = db_name if db_name else f"persona_id_{db_id}"
                                    tracked_labels.append(f"{self.known_identities[str_mapped]}")
                                    # We could also remap the ID here, but for now just show label
                                else:
                                    # New face! Save it to the database
                                    nuevo_id_db = register_new_face(embedding=face_emb, nombre=f"persona_id_{mapped_tid}")
                                    self.known_identities[str_mapped] = f"persona_id_{nuevo_id_db}"
                                    tracked_labels.append(f"persona_id_{nuevo_id_db}")
                                continue
                        except Exception as e:
                            pass
                    
                    # Fallback to random ID if not known and not detected in this frame
                    tracked_labels.append(f"persona_id_{mapped_tid}")
                    
            annotated = self.label_annotator.annotate(scene=annotated, detections=detections_tracked, labels=tracked_labels)
            annotated = self.trace_annotator.annotate(scene=annotated, detections=detections_tracked)
            
        # (Difuminado removido por petición del usuario)
        
        # Etiqueta flotante roja de Apertura Tardía (solo por 5 segundos)
        if (self.late_opening_trigger_time > 0) and (now_ts - self.late_opening_trigger_time < 15.0) and self.alert_config.get("late_opening", True):
            text = f"ALERTA: APERTURA TARDIA [{self.late_opening_time_str}]"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # Dibujar centrado en la parte superior
            tx = (self.width - tw) // 2
            ty = 35
            cv2.rectangle(annotated, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (0, 0, 255), -1) # Fondo Rojo BGR
            cv2.rectangle(annotated, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 10), (255, 255, 255), 1) # Borde Blanco
            cv2.putText(annotated, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Filtro maestro de alertas activas/inactivas
        final_alerts = []
        for alert in alerts:
            alert_id = alert[0]
            # Solo deja pasar la alerta si está en el diccionario y su valor es True
            if self.alert_config.get(alert_id, True):
                final_alerts.append(alert)
        
        return annotated, final_alerts, sessions_to_update
