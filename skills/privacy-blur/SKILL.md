---
name: privacy-blur
description: Ofuscación selectiva de rostros para cumplimiento de normativas de privacidad (LFPDPPP / GDPR) en transmisiones de video.
---

# Modo de Privacidad y Anonimización Facial (P5)

Módulo de protección de datos personales que difumina de forma irreversible los rostros de clientes y terceras personas en el stream visual, preservando la visibilidad de los colaboradores para fines de auditoría operativa.

## Modos de Operación

1. **`non_employees` (Por Defecto):**
   - Evalúa si el identificador biométrico tiene un nombre asignado en BD (`known_identities`).
   - Los empleados (`Angel`, `Cajero_1`, etc.) se mantienen visibles.
   - Los clientes (`persona_id_X` / no identificados) reciben difuminado mosaico irreversible.
2. **`all`:**
   - Difumina el 100% de los rostros en pantalla sin excepción.

## Configuración (`notify_config.json`)

Se sincroniza en caliente cada 30 fotogramas sin necesidad de reiniciar el servicio:

```json
{
  "enable_privacy_blur": true,
  "privacy_blur_mode": "non_employees"
}
```

## Rendimiento

- **Técnica:** Mosaico bicúbico mediante downsampling a 8x8 px y reescalado con interpolación nearest-neighbor.
- **Latencia:** $\sim 1.1\text{ ms}$ en CPU para una escena con 4 personas en resolución 1080p/720p ($< 0.3\text{ ms}$ por persona).
- **Inferencia intacta:** El difuminado se aplica exclusivamente en la capa de renderizado visual (`visual_annotator.py`), permitiendo que el pipeline de detección y reglas de negocio operen con resolución y fidelidad nativas.
