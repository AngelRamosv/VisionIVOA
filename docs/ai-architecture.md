# Arquitectura AI-Ready de IVOA

Este documento describe cómo el repositorio está preparado para ser trabajado por agentes de IA, skills y sistemas de evaluación.

## 1. Filosofía

IVOA pasa de ser un monolito orientado a humanos a un sistema **modular + evaluable + agent-friendly**, inspirado en las mejores prácticas de DeepCamera/Aegis pero sin su deuda técnica.

Principios:
- Skills autocontenidas
- Evaluación automatizada antes de desplegar modelos
- Instrucciones claras y versionadas para las IAs (`AGENTS.md`)
- Prompts reutilizables
- Soporte opcional a MCP

## 2. Capas

```
┌─────────────────────────────────────────────────────────────┐
│  Capa de Agentes / IA Externa (Cursor, Claude, Grok, etc.) │
│  - Lee AGENTS.md                                           │
│  - Usa prompts/                                            │
│  - Puede invocar skills vía MCP o directamente             │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Skills Layer (skills/)                                     │
│  - Cada skill tiene su SKILL.md                             │
│  - Interfaz uniforme de entrada/salida                      │
│  - Pueden ser edge o cloud                                  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Core Edge / Cloud                                          │
│  - ivoa_engine.py (Orquestador ligero desacoplado)          │
│  - detection_pipeline.py + model_optimizer.py               │
│  - bill_validator.py + business_rules.py                    │
│  - queue_analyzer.py + alert_manager.py                     │
│  - visual_annotator.py                                      │
│  - biometric_engine.py                                      │
│  - app_api.py + FastAPI                                     │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  Evaluation Harness (eval/)                                 │
│  - Benchmarks de modelos YOLO                               │
│  - Tests de biometría                                       │
│  - Regresión de reglas de negocio                           │
└─────────────────────────────────────────────────────────────┘
```

## 3. Skills prioritarias a implementar

| Skill ID                    | Descripción                              | Prioridad | Estado       |
|-----------------------------|------------------------------------------|-----------|--------------|
| `hardware-optimizer`        | Auto-export a TensorRT / ONNX / OpenVINO | P1        | ✅ Listo     |
| `webrtc-streaming`          | Reemplazo de MJPEG por WebRTC            | P2        | ✅ Listo     |
| `frame-governor`            | Backpressure y descarte de frames        | P2        | ✅ Listo     |
| `business-rules`            | Reglas de uniforme, celular, efectivo... | P3        | ✅ Listo     |
| `bill-validator`            | Validación OCR + color + textura         | P3        | ✅ Listo     |
| `benchmark-harness`         | Suite de evaluación automatizada         | P4        | ✅ Listo     |
| `privacy-blur`              | Anonimización selectiva de rostros       | P5        | ✅ Listo     |
| `vlm-forensics`             | Consultas en lenguaje natural            | P6        | ⏳ Pendiente |

## 4. Flujo de trabajo recomendado para una IA

1. Leer `AGENTS.md`
2. Leer el `SKILL.md` de la skill que se va a modificar/crear
3. Ejecutar el evaluation harness si se tocan modelos
4. Implementar cambios manteniendo contratos OpenAPI
5. Actualizar el `SKILL.md` y este documento si es necesario

## 5. MCP (Model Context Protocol)

La carpeta `mcp/` contendrá configuraciones de servidores MCP cuando se necesiten (por ejemplo para acceder a la base de datos local o a los streams de video de forma segura).

Por ahora se deja preparada la estructura.
