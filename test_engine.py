import cv2
import numpy as np
from src.edge_sensor.ivoa_engine import IVOAVisionEngine

print("Starting test...")
engine = IVOAVisionEngine(width=1920, height=1080)
frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
print("Processing frame...")
annotated, alerts, sessions = engine.process_frame(frame, 1)
print("Success!", alerts)
