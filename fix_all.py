import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Fix Uniform: Use the center of the torso crop to avoid the white wall
old_uniform = '''                        if t_crop.size > 0 and l_crop.size > 0:
                            t_hsv = cv2.cvtColor(t_crop, cv2.COLOR_BGR2HSV)
                            l_hsv = cv2.cvtColor(l_crop, cv2.COLOR_BGR2HSV)
                            
                            # Ajustes finos: El blanco debe ser casi sin color (Sat < 40) y muy brillante (Val > 180) para ignorar paredes
                            t_white = np.sum((t_hsv[:, :, 1] < 40) & (t_hsv[:, :, 2] > 180)) / float(t_crop.shape[0] * t_crop.shape[1])
                            # El negro en tela bajo luces puede verse gris oscuro, subimos a Val < 85
                            t_black = np.sum(t_hsv[:, :, 2] < 85) / float(t_crop.shape[0] * t_crop.shape[1])
                            
                            l_black = np.sum(l_hsv[:, :, 2] < 70) / float(l_crop.shape[0] * l_crop.shape[1])
                            l_blue = np.sum((l_hsv[:, :, 0] >= 90) & (l_hsv[:, :, 0] <= 130) & (l_hsv[:, :, 1] > 35)) / float(l_crop.shape[0] * l_crop.shape[1])
                            
                            # Exigimos al menos 35% del torso blanco y que no tenga más del 20% negro/oscuro encima (chamarra)
                            has_white_shirt = (t_white >= 0.35 and t_black < 0.20)'''

new_uniform = '''                        if t_crop.size > 0 and l_crop.size > 0:
                            t_hsv = cv2.cvtColor(t_crop, cv2.COLOR_BGR2HSV)
                            l_hsv = cv2.cvtColor(l_crop, cv2.COLOR_BGR2HSV)
                            
                            # Para evitar falsos positivos con la pared blanca (como se vio en la prueba),
                            # extraemos el CENTRO del torso (50% interior) donde está la camisa, ignorando los bordes
                            ch_y, ch_x = t_crop.shape[0], t_crop.shape[1]
                            center_t_hsv = t_hsv[int(ch_y*0.25):int(ch_y*0.75), int(ch_x*0.25):int(ch_x*0.75)]
                            if center_t_hsv.size > 0:
                                t_white = np.sum((center_t_hsv[:, :, 1] < 40) & (center_t_hsv[:, :, 2] > 160)) / float(center_t_hsv.shape[0] * center_t_hsv.shape[1])
                                t_black = np.sum(center_t_hsv[:, :, 2] < 90) / float(center_t_hsv.shape[0] * center_t_hsv.shape[1])
                            else:
                                t_white, t_black = 0, 0
                            
                            l_black = np.sum(l_hsv[:, :, 2] < 70) / float(l_crop.shape[0] * l_crop.shape[1])
                            l_blue = np.sum((l_hsv[:, :, 0] >= 90) & (l_hsv[:, :, 0] <= 130) & (l_hsv[:, :, 1] > 35)) / float(l_crop.shape[0] * l_crop.shape[1])
                            
                            # Ahora exigimos que el CENTRO del pecho sea al menos 40% blanco
                            has_white_shirt = (t_white >= 0.40 and t_black < 0.25)'''

code = code.replace(old_uniform, new_uniform)

# 2. Fix Money: Re-introduce a SAFER size constraint to block the giant wall false positive, but allow close-ups.
old_money = '''                    elif cid == 1:
                        req_conf = 0.02 # 2% de base. El filtro geográfico ya es súper estricto.
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        
                        # Filtro Anti-Monitor: Si el centro del supuesto billete está adentro de la pantalla, es alucinación del color negro'''

new_money = '''                    elif cid == 1:
                        req_conf = 0.06 # 6% de base, equilibrio perfecto.
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        
                        # Evitar el "falso positivo gigante de la pared/escritorio" (donde la caja ocupa casi toda la pantalla)
                        # Pero permitir que se acerque a la cámara (hasta un 40% del ancho de la pantalla, que es muchísimo)
                        if bw > self.width * 0.40 or bh > self.height * 0.40:
                            continue
                            
                        # Filtro Anti-Monitor: Si el centro del supuesto billete está adentro de la pantalla, es alucinación del color negro'''

code = code.replace(old_money, new_money)

# 3. Fix Food: Base model threshold + strict overlap with upper body (but not tiny 20% center)
old_food = '''                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        if (bbox[0] < 220 and bbox[1] > self.height * 0.5) or (bbox[1] < self.height * 0.25 and bbox[0] > self.width * 0.60) or bw > self.width * 0.60 or bh > self.height * 0.60:
                            continue'''

new_food = '''                    if cid == 4:
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

code = code.replace(old_food, new_food)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied for strict but fair uniform, money, and food fixes")
