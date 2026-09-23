import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

v5_old = '''                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        if (bbox[0] < 220 and bbox[1] > self.height * 0.5) or (bbox[1] < self.height * 0.25 and bbox[0] > self.width * 0.60) or bw > self.width * 0.30 or bh > self.height * 0.30:
                            continue'''

v5_new = '''                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        if bw > self.width * 0.30 or bh > self.height * 0.30:
                            continue
                            
                        # El usuario solicita: "debe asegurarse de llevarme un alimento a la boca"
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

code = code.replace(v5_old, v5_new)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied for food mouth detection")
