import os
import warnings
import time
import threading
import requests
import cv2
from flask import Flask, Response, render_template, jsonify
from datetime import datetime

# Configuración avanzada de FFmpeg y supresión de mensajes
os.environ["OPENCV_LOG_LEVEL"] = "0"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import sys
# --- Importar la nueva Arquitectura Modular IVOA ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ivoa_engine import IVOAVisionEngine

app = Flask(__name__)

API_BASE_URL = "http://localhost:8001/v1/events"

def api_log_event(event_type, details, severity="info"):
    def _send():
        try:
            import json
            try:
                BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
                config_path = os.path.join(BASE_DIR, "notify_config.json")
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        cfg = json.load(f)
                        if isinstance(details, dict):
                            details["sucursal"] = cfg.get("sucursal", "Centro de Atención")
                            details["gerente"] = cfg.get("gerente", "Gerente en Turno")
                            
                        # Mapeo de eventos a las opciones de privacidad
                        map_config = {
                            "incidencia_apertura_tardia": "enable_apertura",
                            "incidencia_uniforme": "enable_vestimenta",
                            "incidencia_celular": "enable_celulares",
                            "incidencia_alimentos": "enable_alimentos",
                            "incidencia_paginas": "enable_paginas",
                            "incidencia_arreglo_personal": "enable_arreglo",
                            "incidencia_situacion_atipica": "enable_atipica",
                            "incidencia_dinero": "enable_dinero",
                            "incidencia_audifonos": "enable_audifonos"
                        }
                        cfg_key = map_config.get(event_type)
                        if cfg_key and cfg.get(cfg_key, True) is False:
                            return # Cancelar el envio de la alerta si fue apagada en Ajustes
                            
            except Exception as conf_err:
                pass
                
            if isinstance(details, dict) and "sucursal" not in details:
                details["sucursal"] = "Centro de Atención"
                details["gerente"] = "Gerente en Turno"
                
            requests.post(f"{API_BASE_URL}/log", json={
                "event_type": event_type,
                "details": details,
                "severity": severity
            }, timeout=2)
        except Exception as e:
            print(f"Error logging event to API: {e}")
    threading.Thread(target=_send, daemon=True).start()

def api_upsert_session(tracker_id, session_data):
    def _send():
        try:
            requests.post(f"{API_BASE_URL}/session", json={
                "tracker_id": tracker_id,
                "session_data": session_data
            }, timeout=2)
        except Exception as e:
            print(f"Error upserting session to API: {e}")
    threading.Thread(target=_send, daemon=True).start()

api_log_event("system_started", {"message": "Motor IVOA iniciado"}, "info")

# --- Estado global compartido entre el hilo de detección y Flask ---
frame_lock = threading.Lock()
latest_frame = None
camera_connected = False
camera_logged = False
running = True

# Instancia global del Motor IA
engine = None

# Conexión a la cámara IP Hikvision
CAMERA_URL = "rtsp://admin:QWERTY123@192.168.51.125:554/Streaming/Channels/302"

class CameraReader:
    """Hilo dedicado EXCLUSIVAMENTE a extraer imágenes de la cámara a máxima velocidad con reconexión automática"""
    def __init__(self, src=1):
        self.src = src
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        
        self.ret = False
        self.frame = None
        self.new_frame = False
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        consecutive_failures = 0
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.ret = True
                    self.frame = frame
                    self.new_frame = True
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 15:
                        # Reconectar RTSP
                        time.sleep(1.0)
                        try:
                            self.cap.release()
                            self.cap = cv2.VideoCapture(self.src)
                        except Exception:
                            pass
                        consecutive_failures = 0
                    time.sleep(0.02)
            except Exception as e:
                time.sleep(0.05)

    def read(self):
        wait_count = 0
        while not self.new_frame and self.running and wait_count < 40:
            time.sleep(0.005)
            wait_count += 1
        self.new_frame = False
        return self.ret, self.frame
        
    def release(self):
        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

def detection_thread():
    """Hilo Integrador: Lee la cámara y la pasa al Motor IVOA de Inteligencia Artificial"""
    global latest_frame, camera_connected, camera_logged, engine
    
    cap = CameraReader(CAMERA_URL)
    
    # 1. Instanciamos el Motor (Cerebro) aislando la lógica pesada
    engine = IVOAVisionEngine(width=cap.width, height=cap.height, frame_skip=5)
    
    frame_count = -1

    while running:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            frame = frame.copy()
            
            camera_connected = True
            if not camera_logged:
                api_log_event("camera_online", {"message": "Conexión con Cámara Establecida. Procesando video."}, "info")
                camera_logged = True

            frame_count += 1

            # 2. Delegar TODO el pensamiento (Detección, Reglas, Tracking) al motor
            annotated, alerts, sessions_to_update = engine.process_frame(frame, frame_count)
            
            # 3. Procesar las Alertas devueltas por el motor
            for event_type, msg, severity in alerts:
                if event_type == "finding_created":
                    api_log_event("finding_created", {"queue_length": int(engine.fila_count), "action_required": msg}, severity)
                elif event_type == "queue_threshold_exceeded":
                    api_log_event("queue_threshold_exceeded", {"queue_size": int(engine.fila_count), "message": msg}, severity)
                elif event_type == "late_opening":
                    api_log_event("late_opening", {"message": msg}, severity)
                elif event_type == "customer_abandonment":
                    api_log_event("customer_abandonment", {"tracker_id": 0, "message": msg}, severity)
                elif event_type == "service_completed":
                    api_log_event("service_completed", {"message": msg}, severity)
                elif event_type == "reid_success":
                    api_log_event("reid_success", {"message": msg}, severity)
                else:
                    api_log_event(event_type, {"message": msg}, severity)
                    
            # 4. Actualizar sesiones en Backend
            for tid, session in sessions_to_update:
                api_upsert_session(tid, session)

            with frame_lock:
                latest_frame = annotated
        except Exception as err:
            import traceback
            traceback.print_exc()
            time.sleep(0.02)

    cap.release()

def generate_frames():
    """Generador MJPEG para transmitir el video al navegador."""
    while True:
        frame_to_encode = None
        
        with frame_lock:
            if latest_frame is not None:
                frame_to_encode = latest_frame.copy()
                
        if frame_to_encode is None:
            time.sleep(0.05)
            continue
            
        ret, buffer = cv2.imencode('.jpg', frame_to_encode, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
        time.sleep(0.033)  # ~30 fps

# --- Rutas Flask ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/stats')
def stats():
    if not engine:
        return jsonify({})
        
    total = engine.in_count + engine.out_count
    inside = max(0, engine.in_count - engine.out_count)
    personas_en_toma = engine.fila_count + engine.modulo_count
    
    return jsonify({
        'in_count': engine.in_count,
        'out_count': engine.out_count,
        'total': total,
        'inside': inside,
        'camera_connected': camera_connected,
        'fila_count': int(engine.fila_count),
        'modulo_count': int(engine.modulo_count),
        'personas_en_toma': int(personas_en_toma),
        'alerta_saturacion': engine.alerta_saturacion
    })

@app.route('/api/events')
def api_events():
    import json
    try:
        res = requests.get(f"{API_BASE_URL}/log?limit=5", timeout=2)
        events = res.json()
    except Exception as e:
        print(f"Error fetching events from API: {e}")
        events = []
    
    # Formatear el JSON para que sea legible por humanos y respete los acentos (UTF-8)
    json_str = json.dumps(events, indent=4, ensure_ascii=False)
    return Response(json_str, mimetype='application/json')

if __name__ == '__main__':
    t = threading.Thread(target=detection_thread, daemon=True)
    t.start()
    print("\n✅ Dashboard Edge Sensor listo en: http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

# celular (color verde limon podrido)
# apertura_tardia (color rojo)

