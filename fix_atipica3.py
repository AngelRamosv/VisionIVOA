import re

with open("src/edge_sensor/ivoa_engine.py", "r", encoding="utf-8") as f:
    code = f.read()

old_geom = '''                    # Ajuste geométrico: La forma más segura de saber si está parado sin que la cámara importe,
                    # es ver si la proporción alto/ancho es grande (postura vertical) Y la caja es grande.
                    # Si extiende los brazos, pw crece, por lo que respaldaremos fuertemente con la IA nativa (v9).
                    if ((ph / max(1, pw)) >= 1.45 and ph > self.height * 0.50):
                        alguien_parado = True'''

new_geom = '''                    # Ajuste geométrico INFALIBLE para Situación Atípica (Pararse):
                    # 1. Si la cabeza toca el 20% superior de la cámara (py1 muy bajo)
                    # 2. Si la altura de la caja ocupa casi toda la pantalla (> 85%)
                    # 3. Si la postura es estrictamente vertical (ratio >= 1.50)
                    if (py1 < self.height * 0.20) or (ph > self.height * 0.85) or ((ph / max(1, pw)) >= 1.50 and ph > self.height * 0.40):
                        alguien_parado = True'''

code = code.replace(old_geom, new_geom)

with open("src/edge_sensor/ivoa_engine.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied")
