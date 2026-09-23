import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update CONF_MAP for food to 0.45
code = code.replace("4: 0.15, # comida", "4: 0.45, # comida")

# 2. Update Zoom model for food to 0.35 (to stop back wrinkles from triggering it)
code = code.replace("elif zcid == 4 and zconf >= 0.035:", "elif zcid == 4 and zconf >= 0.35:")

# 3. Update Base V5 Food Logic (Strict Top 20% + 20% size limit)
old_food = '''                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        if bw > self.width * 0.35 or bh > self.height * 0.35:
                            continue # Evita la confusión con la espalda
                            
                        # Asegurar que la comida está en la mitad superior de una persona (torso/boca)
                        is_held_high = False
                        personas = detections_base_all[detections_base_all.class_id == 0].xyxy
                        for pb in personas:
                            mid_y = pb[1] + (pb[3] - pb[1]) * 0.50
                            if max(bbox[0], pb[0]) < min(bbox[2], pb[2]) and max(bbox[1], pb[1]) < min(bbox[3], mid_y):
                                is_held_high = True
                                break
                        if not is_held_high:
                            continue'''

new_food = '''                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        # Una comida (sándwich, botella) jamás ocupará más del 20% del ancho de toda la cámara. La espalda sí.
                        if bw > self.width * 0.20 or bh > self.height * 0.20:
                            continue 
                            
                        # Exigencia ABSOLUTA: "Sólo a la boca"
                        # El centro de la comida DEBE estar en el 20% superior de la persona (la cabeza/cara).
                        is_at_mouth = False
                        fcx = bbox[0] + (bw / 2.0)
                        fcy = bbox[1] + (bh / 2.0)
                        personas = detections_base_all[detections_base_all.class_id == 0].xyxy
                        for pb in personas:
                            # head_y_limit marca dónde termina el 20% superior (el cuello)
                            head_y_limit = pb[1] + (pb[3] - pb[1]) * 0.20
                            if pb[0] < fcx < pb[2] and pb[1] < fcy < head_y_limit:
                                is_at_mouth = True
                                break
                        if not is_at_mouth:
                            continue'''

code = code.replace(old_food, new_food)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied for strict food mouth logic")
