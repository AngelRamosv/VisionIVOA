import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Update CONF_MAP
code = code.replace("4: 0.15, # comida", "4: 0.40, # comida")
code = code.replace("2: 0.05, # billetes", "2: 0.45, # billetes")

# 2. Update Zoom model for food
code = code.replace("elif zcid == 4 and zconf >= 0.035:", "elif zcid == 4 and zconf >= 0.35:")

# 3. Update V9 money threshold and size limit
v9_old = '''                    elif cid == 1:
                        req_conf = 0.02 # 2% de base. El filtro geográfico ya es súper estricto.
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]'''

v9_new = '''                    elif cid == 1:
                        req_conf = 0.45 # 45% de base para evitar falsos en brazos y espaldas.
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        
                        # Filtro de tamaño: Los billetes no pueden ser más grandes que el 25% de la pantalla.
                        if bw > self.width * 0.25 or bh > self.height * 0.25:
                            continue'''

code = code.replace(v9_old, v9_new)

# 4. Update V5 food size limit
v5_old = '''                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        if (bbox[0] < 220 and bbox[1] > self.height * 0.5) or (bbox[1] < self.height * 0.25 and bbox[0] > self.width * 0.60) or bw > self.width * 0.60 or bh > self.height * 0.60:
                            continue'''

v5_new = '''                    if cid == 4:
                        bw = bbox[2] - bbox[0]
                        bh = bbox[3] - bbox[1]
                        if (bbox[0] < 220 and bbox[1] > self.height * 0.5) or (bbox[1] < self.height * 0.25 and bbox[0] > self.width * 0.60) or bw > self.width * 0.30 or bh > self.height * 0.30:
                            continue'''
                            
code = code.replace(v5_old, v5_new)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied")
