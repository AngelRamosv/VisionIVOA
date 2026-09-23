import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

old_zoom = '''                                # Comida / Bebida en manos o boca al estar sentado (v5)
                                elif zcid == 4 and zconf >= 0.35:
                                    zb = zbox.xyxy[0].cpu().numpy()
                                    gx1, gy1, gx2, gy2 = zx1 + zb[0], zy1 + zb[1], zx1 + zb[2], zy1 + zb[3]
                                    zoom_boxes.append([gx1, gy1, gx2, gy2])
                                    zoom_classes.append(4)
                                    zoom_confs.append(zconf)'''

new_zoom = '''                                # Comida / Bebida en manos o boca al estar sentado (v5)
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

code = code.replace(old_zoom, new_zoom)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied for zoom food logic")
