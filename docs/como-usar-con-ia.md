# Cómo usar este repositorio con una IA

## Archivos clave que acabamos de crear

| Archivo / Carpeta              | Propósito                                      |
|--------------------------------|------------------------------------------------|
| `AGENTS.md`                    | Instrucciones principales para cualquier IA    |
| `docs/ai-architecture.md`      | Visión de la capa de skills + evaluación       |
| `skills/_template/SKILL.md`    | Plantilla para crear nuevas skills             |
| `prompts/`                     | Biblioteca de prompts listos para copiar/pegar |
| `eval/README.md`               | Base del harness de evaluación                 |
| `.cursor/rules/ivoa.mdc`       | Reglas automáticas si usas Cursor              |

## Flujo de trabajo recomendado (paso a paso)

### Paso 0 — Preparación (ya hecho)
Copia toda la carpeta `artifacts` (o los archivos nuevos) a la raíz de tu repo local `supervision`.

### Paso 1 — Empezar cualquier tarea
Abre el chat de la IA y pega primero el contenido de:

```
prompts/00-system-ivoa.md
```

### Paso 2 — Crear una skill nueva
Usa el prompt:

```
prompts/01-crear-skill.md
```

### Paso 3 — Implementar Prioridad 1 (Hardware Optimizer)
Usa el prompt:

```
prompts/02-implementar-p1-optimizer.md
```

### Paso 4 — Desacoplar el motor
Usa el prompt:

```
prompts/03-desacoplar-engine.md
```

### Paso 5 — Evaluación
Cuando tengas modelos nuevos, sigue las instrucciones de `eval/README.md`.

---

## Próximos prompts que podemos crear juntos

- Prompt para migrar a WebRTC (P2)
- Prompt para crear el Frame Governor
- Prompt para el VLM forense
- Prompt para modo privacidad / blurring
- Prompt de revisión de código (code review)

Dime cuál quieres que preparemos a continuación.
