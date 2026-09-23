import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

old_v9_atipica = '''                    # Clase 2: situacion_atipica -> sistema clase 11
                    elif cid == 2 and conf >= 0.15: # Bajamos de 0.50 a 0.15 porque V9 da confianzas bajas nativamente'''

new_v9_atipica = '''                    # Clase 2: situacion_atipica -> sistema clase 11
                    elif cid == 2 and conf >= 0.05: # Umbral súper sensible (5%) para que la IA dispare nativamente'''

code = code.replace(old_v9_atipica, new_v9_atipica)

old_geom = '''                    # Ajuste geométrico: Si la altura de la caja es muy grande en proporción a la pantalla,
                    # o si el ratio altura/anchura es moderadamente alto (>= 1.40, que tolera brazos extendidos).
                    if (ph > self.height * 0.65) or ((ph / max(1, pw)) >= 1.40 and ph > self.height * 0.40):
                        alguien_parado = True'''

new_geom = '''                    # Ajuste geométrico: La forma más segura de saber si está parado sin que la cámara importe,
                    # es ver si la proporción alto/ancho es grande (postura vertical) Y la caja es grande.
                    # Si extiende los brazos, pw crece, por lo que respaldaremos fuertemente con la IA nativa (v9).
                    if ((ph / max(1, pw)) >= 1.45 and ph > self.height * 0.50):
                        alguien_parado = True'''

code = code.replace(old_geom, new_geom)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied")
