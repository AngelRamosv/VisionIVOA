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
        }'''

code = code.replace(config_block, config_block_new)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied")
