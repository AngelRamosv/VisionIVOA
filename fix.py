with open('src/edge_sensor/ivoa_engine.py', 'r', encoding='utf-8') as f: lines = f.readlines()

# Extract F2 and F3 based on exact lines
f2_start, f3_start, f4_start = 678, 833, 906

f1 = lines[:f2_start]
f2 = lines[f2_start:f3_start]
f3 = lines[f3_start:f4_start]
f4 = lines[f4_start:]

new_lines = f1 + f3 + f2 + f4

class_def_idx = next(i for i, l in enumerate(new_lines) if 'class IVOAVisionEngine:' in l)
helper_code = '''
    def _get_culprit(self, class_id, detections, detections_tracked):
        try:
            import numpy as np
            indices = np.where(detections.class_id == class_id)[0]
            if len(indices) == 0: return "Persona"
            px1, py1, px2, py2 = detections.xyxy[indices[0]]
            pcx, pcy = (px1 + px2)/2, (py1 + py2)/2
            if hasattr(detections_tracked, 'tracker_id') and detections_tracked.tracker_id is not None:
                for tid, bbox in zip(detections_tracked.tracker_id, detections_tracked.xyxy):
                    hx1, hy1, hx2, hy2 = bbox
                    # Ampliamos el bounding box un 15% para agarrar objetos en la mano/borde
                    bw, bh = hx2-hx1, hy2-hy1
                    if (hx1 - bw*0.15) <= pcx <= (hx2 + bw*0.15) and (hy1 - bh*0.15) <= pcy <= (hy2 + bh*0.15):
                        mapped_tid = self.id_map.get(tid, tid)
                        str_mapped = str(mapped_tid)
                        return getattr(self, 'known_identities', {}).get(str_mapped, f"id_aleatorio_{mapped_tid}")
        except: pass
        return "Persona"
'''
new_lines.insert(class_def_idx + 1, helper_code)

# Now, we need to modify the alert strings in new_lines to use this helper.
# Since this is tricky via regex, we'll do it cleanly:
for i, line in enumerate(new_lines):
    if 'alerts.append(("incidencia_celular"' in line:
        new_lines[i] = line.replace('"ALERTA: Uso de dispositivo móvil (Celular) detectado en horario laboral."', 'f"ALERTA: {self._get_culprit(5, detections, detections_tracked)} detectado(a) usando dispositivo móvil (Celular)."')
    elif 'alerts.append(("incidencia_arreglo_personal"' in line:
        new_lines[i] = line.replace('"ALERTA: Actividad de arreglo personal / maquillaje / peinado detectada."', 'f"ALERTA: {self._get_culprit(10, detections, detections_tracked)} realizando arreglo personal / maquillaje."')
    elif 'alerts.append(("incidencia_alimentos"' in line:
        new_lines[i] = line.replace('"ALERTA: Consumo de alimentos o bebidas detectado en área de atención."', 'f"ALERTA: {self._get_culprit(4, detections, detections_tracked)} detectado(a) consumiendo alimentos/bebidas."')
    elif 'alerts.append(("incidencia_uniforme"' in line:
        new_lines[i] = line.replace('"ALERTA: Código de vestimenta incompleto (Falta playera reglamentaria o pantalón negro)."', 'f"ALERTA: {self._get_culprit(9, detections_uniform, detections_tracked)} con código de vestimenta incompleto."')

with open('src/edge_sensor/ivoa_engine.py', 'w', encoding='utf-8') as f: f.writelines(new_lines)
