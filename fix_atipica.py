import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Fix V9 Threshold for Situacion Atipica
old_v9_atipica = '''                    # Clase 2: situacion_atipica -> sistema clase 11
                    elif cid == 2 and conf >= 0.50:'''

new_v9_atipica = '''                    # Clase 2: situacion_atipica -> sistema clase 11
                    elif cid == 2 and conf >= 0.15: # Bajamos de 0.50 a 0.15 porque V9 da confianzas bajas nativamente'''

code = code.replace(old_v9_atipica, new_v9_atipica)

# 2. Fix Geometric Fallback (alguien_parado)
old_geom = '''                    # Ajuste geométrico: una persona parada tiene la cabeza muy arriba en la toma y ocupa buen tamaño vertical.
                    # El ratio de altura/anchura falla si extienden los brazos (se vuelven anchos).
                    # Solución: Si el inicio de su cabeza (py1) está en el tercio superior de la cámara, está de pie.
                    if (py1 < self.height * 0.33) and (ph > self.height * 0.40):
                        alguien_parado = True'''

new_geom = '''                    # Ajuste geométrico: Si la altura de la caja es muy grande en proporción a la pantalla,
                    # o si el ratio altura/anchura es moderadamente alto (>= 1.40, que tolera brazos extendidos).
                    if (ph > self.height * 0.65) or ((ph / max(1, pw)) >= 1.40 and ph > self.height * 0.40):
                        alguien_parado = True'''

code = code.replace(old_geom, new_geom)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied for situacion atipica")
