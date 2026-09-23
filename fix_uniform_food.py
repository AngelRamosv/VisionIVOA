import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update CONF_MAP for food to 0.25
code = code.replace("4: 0.45, # comida", "4: 0.25, # comida")

# 2. Relax Base V5 Food Logic (Size to 0.35, mouth to 0.35)
old_food = '''                    if cid == 4:
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

new_food = '''                    if cid == 4:
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
                            # 35% cubre la cabeza y el nivel de la boca perfectamente sin ser restrictivo
                            head_y_limit = pb[1] + (pb[3] - pb[1]) * 0.35
                            if pb[0] < fcx < pb[2] and pb[1] < fcy < head_y_limit:
                                is_at_mouth = True
                                break
                        if not is_at_mouth:
                            continue'''

code = code.replace(old_food, new_food)

# 3. Update Zoom Food to 0.15
code = code.replace("elif zcid == 4 and zconf >= 0.35:", "elif zcid == 4 and zconf >= 0.15:")

# 4. Update Uniform Logic to be much stricter about white shirts
old_uniform = '''                            has_white_shirt = (t_white >= 0.35 and t_black < 0.25)'''
new_uniform = '''                            # Si es negra, t_white será muy bajo. Si es blanca, t_white será muy alto (>60%).
                            has_white_shirt = (t_white >= 0.60)'''
code = code.replace(old_uniform, new_uniform)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied for uniform and food relaxation")
