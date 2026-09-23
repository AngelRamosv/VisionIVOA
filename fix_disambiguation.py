import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

disambiguation_orig = '''            # 1. Si colisionan Billetes y Comida en la misma zona, gana la detección con mayor confianza
            if len(detections) > 0 and 2 in detections.class_id and 4 in detections.class_id:
                keep_idx = []
                for idx, (cid, conf, b) in enumerate(zip(detections.class_id, detections.confidence, detections.xyxy)):
                    if cid == 2:
                        # Buscar si choca con comida de mayor confianza
                        suppressed = any(
                            (ocid == 4 and oconf > conf and max(b[0], ob[0]) < min(b[2], ob[2]) and max(b[1], ob[1]) < min(b[3], ob[3]))
                            for ocid, oconf, ob in zip(detections.class_id, detections.confidence, detections.xyxy)
                        )
                        if not suppressed:
                            keep_idx.append(idx)
                    elif cid == 4:
                        # Buscar si choca con billetes de mayor o igual confianza
                        suppressed = any(
                            (ocid == 2 and oconf >= conf and max(b[0], ob[0]) < min(b[2], ob[2]) and max(b[1], ob[1]) < min(b[3], ob[3]))
                            for ocid, oconf, ob in zip(detections.class_id, detections.confidence, detections.xyxy)
                        )
                        if not suppressed:
                            keep_idx.append(idx)
                    elif cid == 10:'''

disambiguation_new = '''            # 1. Si colisionan Billetes y Comida en la misma zona, los Billetes tienen prioridad absoluta
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
                    elif cid == 10:'''
                        
code = code.replace(disambiguation_orig, disambiguation_new)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied for disambiguation")
