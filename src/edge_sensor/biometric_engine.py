import cv2
import numpy as np
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'ivoa.db')}")

engine_args = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
db_engine = create_engine(DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

def find_face_and_embed(frame, bbox):
    """
    Extrae tanto el embedding facial (si está de frente) como la firma del torso (si está de espaldas).
    """
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
        
    result = {"cara": None, "ropa": None}
    
    # 1. Intentar extraer ropa/torso siempre (funciona de espaldas y de frente)
    try:
        import sys
        if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
        from src.edge_sensor.ivoa_engine import extract_color_signature
        hist = extract_color_signature(frame, bbox)
        if hist is not None:
            result["ropa"] = hist.flatten().tolist()
    except: pass

    # 2. Intentar extraer cara (solo funciona de frente)
    try:
        from deepface import DeepFace
        res = DeepFace.represent(img_path=crop, model_name="ArcFace", detector_backend="retinaface", enforce_detection=False)
        if len(res) > 0:
            result["cara"] = res[0]["embedding"]
    except: pass
    
    # Si no sacó ninguno de los dos, regresa None
    if result["cara"] is None and result["ropa"] is None:
        return None
        
    return result

def cosine_similarity(a, b):
    if a is None or b is None: return 0.0
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def identify_person_in_db(embedding_dict, threshold_cara=0.60, threshold_ropa=0.85):
    """
    Compara cara y ropa con la base de datos. Da prioridad a la cara si está disponible.
    """
    db = SessionLocal()
    try:
        import sys
        if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
        from src.cloud_backend.database import SesionPersona
        personas = db.query(SesionPersona).all()
        
        best_match = None
        best_score = -1
        
        for p in personas:
            score = 0.0
            
            # Si trajimos cara y la BD tiene cara, esa manda.
            if embedding_dict.get("cara") and p.cara_embedding:
                try:
                    db_cara = json.loads(p.cara_embedding)
                    c_score = cosine_similarity(embedding_dict["cara"], db_cara)
                    if c_score > threshold_cara and c_score > score:
                        score = c_score
                except: pass
            
            # Si no hay cara (está de espaldas), comparamos por ropa
            elif embedding_dict.get("ropa") and p.ropa_embedding:
                try:
                    db_ropa = json.loads(p.ropa_embedding)
                    r_score = cosine_similarity(embedding_dict["ropa"], db_ropa)
                    if r_score > threshold_ropa and r_score > score:
                        score = r_score
                except: pass

            if score > best_score:
                best_score = score
                best_match = p
                
        # Si pasó alguno de los dos umbrales
        if best_match and best_score > 0:
            return best_match.id_persona, best_match.nombre_reconocido
            
        return None, None
    finally:
        db.close()

def register_new_face(embedding_dict, nombre=None):
    """
    Guarda los embeddings de cara y ropa en la base de datos para la sesión actual y devuelve el ID asignado.
    """
    db = SessionLocal()
    try:
        import sys
        if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)
        from src.cloud_backend.database import SesionPersona
        
        persona = SesionPersona(estado="detectado")
        
        if embedding_dict.get("cara"):
            persona.cara_embedding = json.dumps(embedding_dict["cara"])
        if embedding_dict.get("ropa"):
            persona.ropa_embedding = json.dumps(embedding_dict["ropa"])
            
        if nombre:
            persona.nombre_reconocido = nombre
            
        db.add(persona)
        db.commit()
        db.refresh(persona)
        return persona.id_persona
    finally:
        db.close()
