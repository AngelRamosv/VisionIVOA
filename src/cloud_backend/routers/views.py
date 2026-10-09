import os
from fastapi import APIRouter, Depends, HTTPException, Request, Form
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime

from ..database import get_db, Usuario, Hallazgo
from ..schemas import StatusUpdatePayload

router = APIRouter()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

@router.post("/v1/notify_config")
async def save_notify_config(request: Request):
    data = await request.json()
    import json
    with open(r"config/notify_config.json", "w") as f:
        json.dump(data, f)
    return {"status": "ok"}

@router.get("/v1/notify_config")
def get_notify_config():
    import json
    try:
        with open(r"config/notify_config.json", "r") as f:
            return json.load(f)
    except:
        return {"wa_phone": "", "email": "angel.ramos@izzi.mx", "enable_wa": True, "enable_email": False}

@router.post("/v1/notify_manual")
async def manual_notify_test(request: Request):
    data = await request.json()
    from ..whatsapp_service import send_whatsapp_message
    import threading
    def send():
        send_whatsapp_message(data.get("mensaje", ""), phone=data.get("telefono", ""))
    threading.Thread(target=send).start()
    return {"status": "ok"}

def get_current_user(request: Request, db: Session = Depends(get_db)):
    session_token = request.cookies.get("ivoa_session")
    if not session_token:
        raise HTTPException(status_code=401, detail="No autenticado")
    user = db.query(Usuario).filter(Usuario.username == session_token).first()
    if not user:
        raise HTTPException(status_code=401, detail="Token inválido")
    return user

@router.get("/")
def read_root():
    return RedirectResponse(url="/dashboard")

@router.get("/login")
@router.get("/dashboard/login")
def login_page(request: Request):
    return templates.TemplateResponse(request=request, name="login.html")

@router.post("/login")
def login_post(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(Usuario).filter(Usuario.username == username, Usuario.password == password).first()
    if not user:
        return RedirectResponse(url="/login?error=1", status_code=303)
    
    response = RedirectResponse(url="/dashboard", status_code=303)
    response.set_cookie(key="ivoa_session", value=user.username, httponly=True)
    return response

@router.get("/logout")
@router.post("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("ivoa_session")
    return response

@router.get("/dashboard")
def get_dashboard(request: Request, db: Session = Depends(get_db)):
    session_token = request.cookies.get("ivoa_session")
    if not session_token:
        return RedirectResponse(url="/login", status_code=303)
        
    user = db.query(Usuario).filter(Usuario.username == session_token).first()
    if not user:
         return RedirectResponse(url="/login", status_code=303)
         
    return templates.TemplateResponse(request=request, name="dashboard.html", context={"username": user.username})

@router.get("/v1/findings")
def get_findings(limit: int = 5000, db: Session = Depends(get_db)):
    from ..database import Evento
    hallazgos = db.query(Hallazgo).order_by(Hallazgo.id.desc()).limit(limit).all()
    resultado = []
    # Optimizacion para evitar problema N+1
    import json
    evento_ids = [h.evento_id for h in hallazgos if h.evento_id]
    eventos = db.query(Evento).filter(Evento.id.in_(evento_ids)).all() if evento_ids else []
    evento_map = {e.id: e for e in eventos}
    
    for h in hallazgos:
        sucursal = "Centro de Atención"
        gerente = "Gerente en Turno"
        
        if h.evento_id:
            evento = evento_map.get(h.evento_id)
            if evento:
                try:
                    det = json.loads(evento.details)
                    if isinstance(det, dict):
                        sucursal = det.get("sucursal", "Centro de Atención")
                        gerente = det.get("gerente", "Gerente en Turno")
                except:
                    pass
                    
        resultado.append({
            "id": h.id,
            "evento_id": h.evento_id,
            "categoria": h.categoria,
            "estado": h.estado,
            "severidad": h.severidad,
            "sucursal": sucursal,
            "gerente": gerente,
            "detalles": evento.details if evento else "",
            "fecha_creacion": h.fecha_creacion.strftime("%Y-%m-%d %H:%M:%S") if h.fecha_creacion else None,
            "fecha_vencimiento": h.fecha_vencimiento.strftime("%Y-%m-%d %H:%M:%S") if h.fecha_vencimiento else None,
            "fecha_cierre": h.fecha_cierre.strftime("%Y-%m-%d %H:%M:%S") if h.fecha_cierre else None,
            "asignado_a": h.asignado_a
        })
    return resultado

@router.delete("/v1/findings")
def delete_all_findings(db: Session = Depends(get_db)):
    from ..database import Evento, SesionPersona
    db.query(Hallazgo).delete()
    db.query(Evento).delete()
    db.query(SesionPersona).delete()
    db.commit()
    return {"status": "success", "message": "Todos los tickets han sido eliminados."}

@router.put("/v1/findings/{finding_id}/status")
def update_finding_status(finding_id: int, payload: StatusUpdatePayload, db: Session = Depends(get_db)):
    hallazgo = db.query(Hallazgo).filter(Hallazgo.id == finding_id).first()
    if not hallazgo:
        raise HTTPException(status_code=404, detail="Hallazgo no encontrado")
    
    hallazgo.estado = payload.status
    if payload.status == "CERRADO":
        hallazgo.fecha_cierre = datetime.now()
        
    db.commit()
    return {"status": "success", "nuevo_estado": hallazgo.estado}
