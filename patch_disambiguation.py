import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update the Disambiguation logic
old_disambiguation = '''            # 1. Si colisionan Billetes y Comida en la misma zona, gana la detección con mayor confianza
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

new_disambiguation = '''            # 1. Si colisionan Billetes y Comida en la misma zona, los Billetes tienen prioridad absoluta
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

code = code.replace(old_disambiguation, new_disambiguation)

# 2. Update the Food logic to be ultra strict for "llevar a la boca"
old_food = '''                        # El usuario solicita: "debe asegurarse de llevarme un alimento a la boca"
                        # Filtro espacial estricto: La comida DEBE estar en el tercio superior de una persona (cerca de la cara/boca).
                        is_near_mouth = False
                        personas = detections_base_all[detections_base_all.class_id == 0].xyxy
                        for pb in personas:
                            # Calculamos el límite de la cabeza/boca (top 40% del cuerpo)
                            head_y_limit = pb[1] + (pb[3] - pb[1]) * 0.40
                            if max(bbox[0], pb[0]) < min(bbox[2], pb[2]) and max(bbox[1], pb[1]) < min(bbox[3], head_y_limit):
                                is_near_mouth = True
                                break
                                
                        if not is_near_mouth:
                            continue'''

new_food = '''                        # El usuario solicita: "debe asegurarse de llevarme un alimento a la boca"
                        # Filtro espacial ULTRA estricto: El centro de la comida DEBE estar en el top 20% de una persona (exactamente en la boca/cara).
                        is_near_mouth = False
                        fcx = bbox[0] + (bw / 2.0)
                        fcy = bbox[1] + (bh / 2.0)
                        personas = detections_base_all[detections_base_all.class_id == 0].xyxy
                        for pb in personas:
                            # Calculamos el límite de la cabeza/boca (top 20% del cuerpo)
                            head_y_limit = pb[1] + (pb[3] - pb[1]) * 0.20
                            if pb[0] < fcx < pb[2] and pb[1] < fcy < head_y_limit:
                                is_near_mouth = True
                                break
                                
                        if not is_near_mouth:
                            continue'''

code = code.replace(old_food, new_food)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied for disambiguation and mouth detection")
