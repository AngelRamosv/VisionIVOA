import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

config_block = '''        self.alert_config = {
            "incidencia_dinero": False,
            "incidencia_alimentos": False,
            "incidencia_celular": False,
            "incidencia_audifonos": False,
            "incidencia_paginas": False,
            "incidencia_arreglo_personal": False,
            "incidencia_seguridad": False,
            "incidencia_uniforme": False,
            "incidencia_llaves": False,
            "incidencia_situacion_atipica": True,
            "incidencia_saturacion": False,
            "late_opening": False,
            "excessive_service_time": False,
            "customer_abandonment": False,
            "service_completed": False
        }'''

config_block_new = '''        self.alert_config = {
            "incidencia_dinero": True,
            "incidencia_alimentos": True,
            "incidencia_celular": True,
            "incidencia_audifonos": True,
            "incidencia_paginas": True,
            "incidencia_arreglo_personal": True,
            "incidencia_seguridad": True,
            "incidencia_uniforme": True,
            "incidencia_llaves": True,
            "incidencia_situacion_atipica": True,
            "incidencia_saturacion": True,
            "late_opening": True,
            "excessive_service_time": True,
            "customer_abandonment": True,
            "service_completed": True
        }'''

code = code.replace(config_block, config_block_new)

old_geom = '''                    # Ajuste geométrico INFALIBLE para Situación Atípica (Pararse):
                    # 1. Si la cabeza toca el 20% superior de la cámara (py1 muy bajo)
                    # 2. Si la altura de la caja ocupa casi toda la pantalla (> 85%)
                    # 3. Si la postura es estrictamente vertical (ratio >= 1.50)
                    if (py1 < self.height * 0.20) or (ph > self.height * 0.85) or ((ph / max(1, pw)) >= 1.50 and ph > self.height * 0.40):
                        alguien_parado = True'''

new_geom = '''                    # Ajuste geométrico BALANCEADO para Situación Atípica (Pararse):
                    # Relajamos el ratio a 1.30 para que capture el momento en que se pone de pie sin fallar.
                    # Reducimos la exigencia de altura a 75% o que la cabeza suba al 25% superior.
                    if (py1 < self.height * 0.25) or (ph > self.height * 0.75) or ((ph / max(1, pw)) >= 1.30 and ph > self.height * 0.45):
                        alguien_parado = True'''

code = code.replace(old_geom, new_geom)

old_v9_atipica = '''                    # Clase 2: situacion_atipica -> sistema clase 11
                    elif cid == 2 and conf >= 0.05: # Umbral súper sensible (5%) para que la IA dispare nativamente'''

new_v9_atipica = '''                    # Clase 2: situacion_atipica -> sistema clase 11
                    elif cid == 2 and conf >= 0.01: # Umbral súper sensible (1%) garantizado para que la IA dispare nativamente'''

code = code.replace(old_v9_atipica, new_v9_atipica)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied")
