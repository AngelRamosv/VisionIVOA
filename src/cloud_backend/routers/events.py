import json
from fastapi import APIRouter, Depends
from fastapi.responses import Response
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import csv
import io

from ..database import get_db, SesionPersona, Evento, Hallazgo
from ..schemas import PersonSessionPayload, EventoPayload

router = APIRouter()

@router.get("/report/excel")
def download_excel_report(db: Session = Depends(get_db)):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    except ImportError:
        return {"error": "openpyxl not installed"}
        
    events = db.query(Evento).order_by(Evento.timestamp.desc()).all()
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Reporte Incidencias"
    
    # Headers
    headers = ['ID', 'Fecha', 'Hora', 'Etiqueta (Categoría)', 'Detalles', 'Severidad']
    ws.append(headers)
    
    # Styling Headers
    header_fill = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    header_font = Font(color="FFFFFF", bold=True)
    for col in range(1, 7):
        cell = ws.cell(row=1, column=col)
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")
        
    # Styling Rows
    thin_border = Border(left=Side(style='thin', color='E2E8F0'), 
                         right=Side(style='thin', color='E2E8F0'), 
                         top=Side(style='thin', color='E2E8F0'), 
                         bottom=Side(style='thin', color='E2E8F0'))
                         
    row_num = 2
    for ev in events:
        fecha = ev.timestamp.strftime('%Y-%m-%d') if ev.timestamp else ""
        hora = ev.timestamp.strftime('%H:%M:%S') if ev.timestamp else ""
        try:
            parsed = json.loads(ev.details)
            detalle = parsed.get('message', ev.details)
        except:
            detalle = ev.details
            
        ws.append([ev.id, fecha, hora, ev.event_type, detalle, ev.severity])
        
        # Color specific categories
        cat = ev.event_type
        row_color = "FFFFFF"
        if cat == "late_opening": row_color = "F3E8FF" # Purple tint
        elif "celular" in cat: row_color = "FEE2E2" # Red tint
        elif "uniforme" in cat: row_color = "FFE4E6" # Rose tint
        elif "alimentos" in cat: row_color = "FFEDD5" # Orange tint
        
        fill = PatternFill(start_color=row_color, end_color=row_color, fill_type="solid")
        
        for col in range(1, 7):
            c = ws.cell(row=row_num, column=col)
            c.fill = fill
            c.border = thin_border
            if col in [1, 2, 3, 6]:
                c.alignment = Alignment(horizontal="center")
        row_num += 1
        
    # Auto-adjust column widths
    dims = {}
    for row in ws.rows:
        for cell in row:
            if cell.value:
                dims[cell.column_letter] = max((dims.get(cell.column_letter, 0), len(str(cell.value))))    
    for col, value in dims.items():
        ws.column_dimensions[col].width = min(value + 2, 60)
        
    # Save to memory
    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    
    return Response(
        content=output.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=Reporte_IVOAVISION_{datetime.now().strftime('%Y%m%d')}.xlsx"}
    )

@router.post("/session")
def update_session(payload: PersonSessionPayload, db: Session = Depends(get_db)):
    db_session = db.query(SesionPersona).filter(SesionPersona.id_persona == payload.tracker_id).first()
    
    def parse_dt(dt_str):
        if not dt_str:
            return None
        return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

    data = payload.session_data
    if not db_session:
        db_session = SesionPersona(
            id_persona=payload.tracker_id,
            primera_vez_visto=parse_dt(data.first_seen),
            entro_a_fila=parse_dt(data.entered_fila),
            salio_de_fila=parse_dt(data.left_fila),
            entro_a_modulo=parse_dt(data.entered_modulo),
            salio_de_modulo=parse_dt(data.left_modulo),
            estado=data.status,
            tiempo_total_atencion_segundos=data.total_service_time
        )
        db.add(db_session)
    else:
        if data.first_seen: db_session.primera_vez_visto = parse_dt(data.first_seen)
        if data.entered_fila: db_session.entro_a_fila = parse_dt(data.entered_fila)
        if data.left_fila: db_session.salio_de_fila = parse_dt(data.left_fila)
        if data.entered_modulo: db_session.entro_a_modulo = parse_dt(data.entered_modulo)
        if data.left_modulo: db_session.salio_de_modulo = parse_dt(data.left_modulo)
        if data.status: db_session.estado = data.status
        if data.total_service_time is not None: db_session.tiempo_total_atencion_segundos = data.total_service_time
        
    db.commit()
    return {"status": "success"}

@router.post("/log")
def log_event(payload: EventoPayload, db: Session = Depends(get_db)):
    nuevo_evento = Evento(
        event_type=payload.event_type,
        timestamp=datetime.now(),
        details=json.dumps(payload.details),
        severity=payload.severity
    )
    db.add(nuevo_evento)
    db.commit()
    db.refresh(nuevo_evento)

    critical_events = [
        "late_opening", 
        "incidencia_uniforme", 
        "incidencia_celular", 
        "incidencia_audifonos",
        "incidencia_dinero",
        "incidencia_billetes",
        "incidencia_llaves",
        "incidencia_alimentos",
        "incidencia_comida",
        "incidencia_arreglo_personal",
        "incidencia_maquillaje",
        "incidencia_paginas",
        "paginas_no_autorizadas",
        "incidencia_saturacion",
        "queue_saturation",
        "customer_abandonment",
        "excessive_service_time",
        "incidencia_atencion",
        "incidencia_situacion_atipica",
        "incidencia_seguridad"
    ]
    if payload.event_type in critical_events:
        if payload.event_type == "late_opening":
            cat = "Apertura tardia"
        elif payload.event_type == "incidencia_uniforme":
            cat = "Codigo de vestimenta"
        elif payload.event_type == "incidencia_celular":
            cat = "Uso de celulares"
        elif payload.event_type == "incidencia_audifonos":
            cat = "Uso de audifonos"
        elif payload.event_type in ["incidencia_dinero", "incidencia_billetes"]:
            cat = "Billetes dinero"
        elif payload.event_type in ["incidencia_llaves"]:
            cat = "Llaves"
        elif payload.event_type in ["incidencia_alimentos", "incidencia_comida"]:
            cat = "Consumo de alimentos"
        elif payload.event_type in ["incidencia_arreglo_personal", "incidencia_maquillaje"]:
            cat = "Arreglo personal"
        elif payload.event_type in ["incidencia_paginas", "paginas_no_autorizadas"]:
            cat = "Paginas no autorizadas"
        elif payload.event_type in ["incidencia_saturacion", "queue_saturation"]:
            cat = "Saturación de fila"
        elif payload.event_type in ["customer_abandonment"]:
            cat = "Abandono de fila"
        elif payload.event_type in ["excessive_service_time", "incidencia_atencion"]:
            cat = "Atención excedida"
        elif payload.event_type == "incidencia_situacion_atipica":
            cat = "Situacion atipica"
        elif payload.event_type == "incidencia_seguridad":
            cat = "Seguridad"
        else:
            cat = "Acción Requerida"
            
        # Regla de negocio: "Apertura tardia", "Codigo de vestimenta", "Seguridad/Situacion atipica", etc. solo 1 vez.
        # "Uso de celulares", "Consumo de alimentos", "Paginas no autorizadas", "Uso de audifonos" permiten múltiples alertas.
        categorias_multiples = ["Uso de celulares", "Consumo de alimentos", "Paginas no autorizadas", "Uso de audifonos"]
        
        if cat not in categorias_multiples:
            existe = db.query(Hallazgo).filter(Hallazgo.categoria == cat).first()
            if existe:
                return {"status": "success", "message": "Alerta ignorada (regla de única vez)"}

        # Registramos el hallazgo
        nuevo_hallazgo = Hallazgo(
            evento_id=nuevo_evento.id,
            categoria=cat,
            estado="DETECTADO",
            severidad=payload.severity,
            fecha_creacion=datetime.now(),
            fecha_vencimiento=datetime.now() + timedelta(minutes=15)
        )
        db.add(nuevo_hallazgo)
        db.commit()
        db.refresh(nuevo_hallazgo)
        
        # (En la nueva arquitectura de Benchmark, las notificaciones en tiempo real estan apagadas.
        # Solo se enviaran los resumenes cuando se presione el boton de Enviar Alertas despues del Batch).
        pass

    return {"status": "success"}

def trigger_alerts(cat, severity, details):
    import json
    from ..whatsapp_service import send_whatsapp_message
    
    config = {"wa_phone": "+5215584073442", "email": "angel.ramos@izzi.mx", "enable_wa": True, "enable_email": False}
    try:
        with open(r"config/notify_config.json", "r") as f:
            config = json.load(f)
    except: pass
    
    if isinstance(details, dict):
        det_str = details.get('message', str(details))
    else:
        det_str = str(details)
        
    # Mejorar el formato de severidad
    sev_map = {
        "warning": "🔴 ALTA (Infracción Detectada)",
        "critical": "🚨 CRÍTICA (Alerta de Seguridad)",
        "info": "🟡 MEDIA (Observación)"
    }
    sev_es = sev_map.get(str(severity).lower(), "🔴 ALTA")
    
    from datetime import datetime
    now_str = datetime.now().strftime("%d/%m/%Y a las %H:%M:%S")
    
    msg = f"🚨 *Alerta Operativa (IVOA)* 🚨\n\n*Fecha y Hora:* {now_str}\n*Incidencia:* {cat}\n*Detalle:* {det_str}\n\n📊 *Monitoreo en vivo:*\nhttp://192.168.51.231:8001/dashboard"
    
    if config.get("enable_wa") and config.get("wa_phone"):
        send_whatsapp_message(msg, phone=config.get("wa_phone"))
        
    if config.get("enable_email") and config.get("email"):
        print(f"[Simulador Email] Enviando correo de alerta a {config.get('email')}...\nContenido:\n{msg}")
        # Aquí se puede conectar smtplib o SendGrid en el futuro

@router.get("/log")
def get_events(limit: int = 5, db: Session = Depends(get_db)):
    eventos = db.query(Evento).order_by(Evento.timestamp.desc()).limit(limit).all()
    resultado = []
    for ev in eventos:
        try:
            detalles_dict = json.loads(ev.details)
        except:
            detalles_dict = {}
        resultado.append({
            "id": ev.id,
            "event_type": ev.event_type,
            "timestamp": ev.timestamp.strftime("%Y-%m-%d %H:%M:%S") if ev.timestamp else None,
            "details": detalles_dict,
            "severity": ev.severity
        })
    return resultado

@router.get("/session")
def get_sessions(limit: int = 10, db: Session = Depends(get_db)):
    sesiones = db.query(SesionPersona).order_by(SesionPersona.primera_vez_visto.desc()).limit(limit).all()
    resultado = []
    for s in sesiones:
        resultado.append({
            "id_rastreo": s.id_persona,
            "primera_vez_visto": s.primera_vez_visto.strftime("%Y-%m-%d %H:%M:%S") if s.primera_vez_visto else None,
            "estado": s.estado,
            "tiempo_total_servicio_segundos": s.tiempo_total_atencion_segundos,
            "entro_a_fila": s.entro_a_fila.strftime("%Y-%m-%d %H:%M:%S") if s.entro_a_fila else None,
            "salio_de_fila": s.salio_de_fila.strftime("%Y-%m-%d %H:%M:%S") if s.salio_de_fila else None,
            "entro_a_modulo": s.entro_a_modulo.strftime("%Y-%m-%d %H:%M:%S") if s.entro_a_modulo else None,
            "salio_de_modulo": s.salio_de_modulo.strftime("%Y-%m-%d %H:%M:%S") if s.salio_de_modulo else None
        })
    return resultado

@router.post("/start_batch_proxy")
def start_batch_proxy():
    import requests
    try:
        res = requests.post("http://localhost:5000/api/start_batch", timeout=5)
        return res.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@router.post("/trigger_batch")
def trigger_batch(db: Session = Depends(get_db)):
    import json
    import os
    from datetime import datetime, timedelta
    try:
        from ..whatsapp_service import send_whatsapp_message
    except:
        pass
    
    # 1. Quitar modo batch
    try:
        with open(r"config/notify_config.json", "r") as f:
            cfg = json.load(f)
        cfg["batch_mode"] = False
        with open(r"config/notify_config.json", "w") as f:
            json.dump(cfg, f)
    except: pass
    
    # 2. Borrar video
    video_path = "C:/Users/Angel/Desktop/prueba/incidencias.mp4"
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except: pass

    # 3. Recopilar alertas de los ultimos 3 minutos
    hace_15_min = datetime.now() - timedelta(minutes=15)
    recientes = db.query(Hallazgo).filter(Hallazgo.fecha_creacion >= hace_15_min).all()
    
    resumen_texto = ""
    if len(recientes) > 0:
        resumen_texto = chr(10).join([f"• {r.categoria} ({r.fecha_creacion.strftime('%H:%M:%S')})" for r in recientes])
    else:
        resumen_texto = "Ninguna incidencia grave detectada."

    # 4. Enviar WA
    phone = cfg.get("wa_phone", "+5215584073442") if 'cfg' in locals() else "+5215584073442"
    
    lineas = [
        "📊 *RESUMEN DE GRABACIÓN LOTE IVOA*",
        "",
        "Las siguientes incidencias fueron detectadas durante el periodo de monitoreo:",
        "",
        resumen_texto,
        "",
        "✅ Memoria de video liberada."
    ]
    msg = chr(10).join(lineas)
    
    try:
        send_whatsapp_message(msg, phone=phone)
    except Exception as e:
        print("Error mandando wa:", e)
        pass
    
    return {"status": "success", "alertas": len(recientes)}

@router.get("/batch_progress_proxy")
def batch_progress_proxy():
    import requests
    try:
        res = requests.get("http://localhost:5000/api/batch_progress", timeout=2)
        return res.json()
    except Exception as e:
        return {"status": "error", "progress": 0, "elapsed_seconds": 0}
