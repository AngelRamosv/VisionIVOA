import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Rename _get_culprit to _get_culprits and rewrite it to return a list of unique strings
new_get_culprits = '''    def _get_culprits(self, class_id, detections, detections_tracked):
        culprits = set()
        try:
            import numpy as np
            indices = np.where(detections.class_id == class_id)[0]
            
            if len(indices) == 0:
                pass
            else:
                for idx in indices:
                    px1, py1, px2, py2 = detections.xyxy[idx]
                    pcx, pcy = (px1 + px2)/2, (py1 + py2)/2
                    
                    best_tid = None
                    min_dist = float('inf')
                    found_inside = False
                    
                    if hasattr(detections_tracked, 'tracker_id') and detections_tracked.tracker_id is not None:
                        for tid, bbox in zip(detections_tracked.tracker_id, detections_tracked.xyxy):
                            hx1, hy1, hx2, hy2 = bbox
                            hcx, hcy = (hx1 + hx2)/2, (hy1 + hy2)/2
                            dist = (pcx - hcx)**2 + (pcy - hcy)**2
                            
                            bw, bh = hx2-hx1, hy2-hy1
                            if (hx1 - bw*0.30) <= pcx <= (hx2 + bw*0.30) and (hy1 - bh*0.30) <= pcy <= (hy2 + bh*0.30):
                                mapped_tid = self.id_map.get(tid, tid)
                                str_mapped = str(mapped_tid)
                                culprits.add(getattr(self, 'known_identities', {}).get(str_mapped, f"id_aleatorio_{mapped_tid}"))
                                found_inside = True
                                break
                                
                            if dist < min_dist:
                                min_dist = dist
                                best_tid = tid
                                
                        if not found_inside and best_tid is not None:
                            mapped_tid = self.id_map.get(best_tid, best_tid)
                            str_mapped = str(mapped_tid)
                            culprits.add(getattr(self, 'known_identities', {}).get(str_mapped, f"id_aleatorio_{mapped_tid}"))
        except: pass
        
        if not culprits:
            mapped_tid = 1
            if hasattr(self, 'id_map') and len(self.id_map) > 0:
                mapped_tid = list(self.id_map.values())[0]
            str_mapped = str(mapped_tid)
            culprits.add(getattr(self, 'known_identities', {}).get(str_mapped, f"id_aleatorio_{mapped_tid}"))
            
        return list(culprits)'''

# Replace the old method block
import textwrap
# We use regex to find the def _get_culprit block
code = re.sub(
    r"    def _get_culprit\(self, class_id, detections, detections_tracked\):.*?return getattr\(self, 'known_identities', \{\}\)\.get\(str_mapped, f\"id_aleatorio_\{mapped_tid\}\"\)",
    new_get_culprits,
    code,
    flags=re.DOTALL
)

# 2. Replace the alert appends
def replace_alert(match):
    indent = match.group(1)
    rule_name = match.group(2)
    culprit_args = match.group(3)
    rest_of_msg = match.group(4)
    severity = match.group(5)
    
    return f"{indent}for c_id in self._get_culprits({culprit_args}):\n{indent}    alerts.append((\"{rule_name}\", f\"ALERTA: {{c_id}} {rest_of_msg}\", \"{severity}\"))"

# Example: alerts.append(("incidencia_uniforme", f"ALERTA: {self._get_culprit(9, detections, detections_tracked)} con código...", "warning"))
pattern = r"([ \t]*)alerts\.append\(\(\"([^\"]+)\", f\"ALERTA: \{self\._get_culprit\(([^)]+)\)\} ([^\"]+)\", \"([^\"]+)\"\)\)"
code = re.sub(pattern, replace_alert, code)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print('Success')
