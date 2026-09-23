"""
Script independiente para enviar mensajes de WhatsApp a través de la API de OpenWA.

Este script es 100% autónomo:
- No depende de los archivos del proyecto OpenWA ni de variables de entorno locales.
- No requiere librerías externas (utiliza únicamente la librería estándar de Python: urllib, json, sys).
- Puede copiarse y ejecutarse en cualquier otra máquina o proyecto que tenga acceso de red al servidor.
"""

import sys
import json
import argparse
import urllib.request
import urllib.error

# Soporte UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ==============================================================================
# CONFIGURACIÓN DEL SERVIDOR REMOTO
# ==============================================================================
OPENWA_BASE_URL = "http://192.168.49.93:2785"
API_MASTER_KEY = "mAGD3357$"
DEFAULT_PHONE = "+5215584073442"
# ==============================================================================

def normalize_chat_id(phone_or_chat_id: str) -> str:
    """
    Normaliza el número de teléfono o ID de chat a formato JID de WhatsApp.
    Ejemplos:
      '+5215584073442'        -> '5215584073442@c.us'
      '52 1 55 8407-3442'     -> '5215584073442@c.us'
      '120363399851611827@g.us' -> '120363399851611827@g.us' (grupo)
    """
    cleaned = phone_or_chat_id.strip()
    if "@" in cleaned:
        return cleaned

    # Conservar solo dígitos
    digits = "".join(ch for ch in cleaned if ch.isdigit())
    return f"{digits}@c.us"

def get_active_session_id(base_url: str, api_key: str, preferred_name: str = None) -> tuple[str | None, str | None]:
    """
    Consulta la API remota para obtener el ID de una sesión activa (ready).
    Retorna: (session_id, session_name)
    """
    url = f"{base_url.rstrip('/')}/api/sessions"
    req = urllib.request.Request(
        url,
        headers={
            "X-API-Key": api_key,
            "User-Agent": "Standalone-OpenWA-Client/1.0",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            if resp.status == 200:
                sessions = json.loads(resp.read().decode("utf-8"))
                if not sessions:
                    return None, None

                # 1. Si especificó nombre o ID de sesión
                if preferred_name:
                    for s in sessions:
                        if s.get("id") == preferred_name or s.get("name") == preferred_name:
                            return s.get("id"), s.get("name")

                # 2. Buscar la primera en estado 'ready'
                for s in sessions:
                    if s.get("status") == "ready":
                        return s.get("id"), s.get("name")

                # 3. Fallback a la primera sesión disponible
                first = sessions[0]
                return first.get("id"), first.get("name")
    except Exception as e:
        print(f"[!] Error al conectar con OpenWA en {base_url}: {e}")

    return None, None

def send_whatsapp_message(
    message: str,
    phone: str = DEFAULT_PHONE,
    base_url: str = OPENWA_BASE_URL,
    api_key: str = API_MASTER_KEY,
    session_id: str = None,
) -> dict | None:
    """
    Función reutilizable para enviar un mensaje de WhatsApp.
    Puede importarse en cualquier script:
        from send_message import send_whatsapp_message
        send_whatsapp_message("Hola!", "+5215584073442")
    """
    # Si no se pasó sesión, buscar la activa automáticamente
    if not session_id:
        session_id, session_name = get_active_session_id(base_url, api_key)
        if not session_id:
            raise RuntimeError(f"No se encontró ninguna sesión activa en {base_url}")
    else:
        session_name = session_id

    chat_id = normalize_chat_id(phone)
    url = f"{base_url.rstrip('/')}/api/sessions/{session_id}/messages/send-text"
    payload = {
        "chatId": chat_id,
        "text": message,
    }
    data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-API-Key": api_key,
            "User-Agent": "Standalone-OpenWA-Client/1.0",
        },
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=30.0) as resp:
        if resp.status in (200, 201):
            return json.loads(resp.read().decode("utf-8"))

    return None

def main():
    parser = argparse.ArgumentParser(
        description="Cliente independiente para enviar mensajes de WhatsApp mediante OpenWA API"
    )
    parser.add_argument(
        "-m", "--message",
        default="¡Hola! Mensaje enviado desde un script independiente remoto 🚀",
        help="Texto del mensaje a enviar"
    )
    parser.add_argument(
        "-p", "--phone",
        default=DEFAULT_PHONE,
        help=f"Número de teléfono o Chat ID (ej. +5215584073442 o grupo JID). Por defecto: {DEFAULT_PHONE}"
    )
    parser.add_argument(
        "-u", "--url",
        default=OPENWA_BASE_URL,
        help=f"URL del servidor OpenWA. Por defecto: {OPENWA_BASE_URL}"
    )
    parser.add_argument(
        "-k", "--key",
        default=API_MASTER_KEY,
        help="API Key de autenticación"
    )
    parser.add_argument(
        "-s", "--session",
        default=None,
        help="ID o nombre de sesión específica (opcional, por defecto detecta la activa)"
    )

    args = parser.parse_args()

    print("=" * 65)
    print("  [>] OpenWA - Cliente Remoto Independiente")
    print("=" * 65)
    print(f"[*] Servidor:     {args.url}")
    print(f"[*] Destinatario: {normalize_chat_id(args.phone)}")
    print(f"[*] Mensaje:      \"{args.message}\"")
    print("-" * 65)

    try:
        print("[*] Conectando y enviando...", end="", flush=True)
        result = send_whatsapp_message(
            message=args.message,
            phone=args.phone,
            base_url=args.url,
            api_key=args.key,
            session_id=args.session,
        )

        if result:
            msg_id = result.get("id") or result.get("messageId") or "OK"
            print("\n[+] ¡Mensaje enviado exitosamente!")
            print(f"    - ID del mensaje: {msg_id}")
            print(f"    - Destino:       {normalize_chat_id(args.phone)}")
        else:
            print("\n[-] Respuesta vacía del servidor.")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        print(f"\n[!] Error HTTP {e.code}: {body or e.reason}")
    except Exception as e:
        print(f"\n[!] Error: {e}")

    print("=" * 65)

if __name__ == "__main__":
    main()
