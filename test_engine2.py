import cv2
import numpy as np
from src.edge_sensor.ivoa_engine import IVOAVisionEngine

print("Starting test...")
engine = IVOAVisionEngine(width=1920, height=1080)
frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

# Force a condition that triggers an alert (e.g. Situacion Atipica)
# Situacion atipica triggers if someone is standing.
# We need to bypass the model and just trigger the geometric rule.
engine.alert_config["incidencia_situacion_atipica"] = True

# We can't easily mock detections, but let's just see if the engine crashes or what it returns.
annotated, alerts, sessions = engine.process_frame(frame, 1)
print("ALERTS:", alerts)
