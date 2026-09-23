import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Revert CONF_MAP
code = code.replace("4: 0.40, # comida", "4: 0.15, # comida")
code = code.replace("2: 0.45, # billetes", "2: 0.05, # billetes")

# 2. Revert V9 Money Threshold and Size Limit
v9_bad = '''                    elif cid == 1:
                        req_conf = 0.08 # 8% de base (el modelo v9 de billetes tiene confianzas naturalmente bajas).
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        
                        # Filtro de tamaño: Los billetes no pueden ser más grandes que el 20% de la pantalla.
                        if bw > self.width * 0.20 or bh > self.height * 0.20:
                            continue'''

v9_orig = '''                    elif cid == 1:
                        req_conf = 0.02 # 2% de base. El filtro geográfico ya es súper estricto.
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]'''
                        
code = code.replace(v9_bad, v9_orig)

# 3. Revert V5 Food Filter
v5_bad = '''                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        if bw > self.width * 0.30 or bh > self.height * 0.30:
                            continue
                            
                        # El usuario solicita: "debe asegurarse de llevarme un alimento a la boca"
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

v5_orig = '''                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        if (bbox[0] < 220 and bbox[1] > self.height * 0.5) or (bbox[1] < self.height * 0.25 and bbox[0] > self.width * 0.60) or bw > self.width * 0.60 or bh > self.height * 0.60:
                            continue'''

code = code.replace(v5_bad, v5_orig)

# 4. Revert Disambiguation
disambiguation_bad = '''            # 1. Si colisionan Billetes y Comida en la misma zona, los Billetes tienen prioridad absoluta
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
                        
code = code.replace(disambiguation_bad, disambiguation_orig)

# 5. Revert Zoom Food
zoom_bad = '''                                # Comida / Bebida en manos o boca al estar sentado (v5)
                                elif zcid == 4 and zconf >= 0.35:
                                    zb = zbox.xyxy[0].cpu().numpy()
                                    # Filtro estricto: El centro de la comida en el recorte (que es cabeza+pecho)
                                    # debe estar en la mitad superior de ese recorte (que corresponde a la cara)
                                    zcy = zb[1] + (zb[3] - zb[1]) / 2.0
                                    crop_h = int((pbox[3] - pbox[1]) * 0.45)
                                    if zcy < crop_h * 0.60: # 60% superior del recorte (cubre la boca perfectamente)
                                        gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]
                                        zoom_boxes.append([gx1, gy1, gx2, gy2])
                                        zoom_classes.append(4)
                                        zoom_confs.append(zconf)'''

zoom_orig = '''                                # Comida / Bebida en manos o boca al estar sentado (v5)
                                elif zcid == 4 and zconf >= 0.035:
                                    zb = zbox.xyxy[0].cpu().numpy()
                                    gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]
                                    zoom_boxes.append([gx1, gy1, gx2, gy2])
                                    zoom_classes.append(4)
                                    zoom_confs.append(zconf)'''

code = code.replace(zoom_bad, zoom_orig)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Reverted to original behavior for food and money")
