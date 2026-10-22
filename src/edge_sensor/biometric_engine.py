import os
import sys
import json
import cv2
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Importaciones de UniFace (Backend ONNX Runtime ultra-eficiente)
from uniface.detection import SCRFD
from uniface.recognition import ArcFace
from uniface.constants import ArcFaceWeights

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'data', 'ivoa.db')}")

engine_args = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
db_engine = create_engine(DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

# Modelos en memoria compartida (Lazy Singleton)
_face_detector = None
_face_recognizer = None
_db_cache = None  # Cache de identidades en memoria para evitar I/O repetitivo

def get_face_models():
    """Inicializa SCRFD y ArcFace de UniFace una sola vez."""
    global _face_detector, _face_recognizer
    if _face_detector is None:
        _face_detector = SCRFD()
    if _face_recognizer is None:
        _face_recognizer = ArcFace(model_name=ArcFaceWeights.MNET)
    return _face_detector, _face_recognizer

def find_face_and_embed(frame, bbox):
    """
    Recibe el frame completo y el bbox de la PERSONA (detectada por YOLO).
    Detecta rostro y 5 landmarks faciales dentro de la persona y extrae su embedding ArcFace normalizado.
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1

    # Filtro: descartar detecciones demasiado pequeñas o lejanas
    if pw < 30 or ph < 60:
        return None

    # Recortar la parte superior de la persona (zona de cabeza: primer 50% vertical)
    head_y2 = min(h, y1 + int(ph * 0.50))
    crop_head = frame[max(0, y1):head_y2, max(0, x1):min(w, x2)]
    if crop_head.size == 0:
        return None

    try:
        detector, recognizer = get_face_models()
        faces = detector.detect(crop_head)

        # Si no detectó en el 50%, intentar con el cuerpo completo si es mediano
        if not faces and ph < 250:
            crop_full = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop_full.size > 0:
                faces = detector.detect(crop_full)
                crop_head = crop_full

        if not faces:
            return None

        # Seleccionar el rostro con mayor confianza
        best_face = max(faces, key=lambda f: f.confidence)

        # Extraer embedding ArcFace con alineación facial por landmarks (vector de 512-dim)
        emb = recognizer.get_normalized_embedding(crop_head, best_face.landmarks)
        if emb is not None:
            return {
                "cara": emb.tolist(),
                "ropa": None
            }
    except Exception as e:
        print(f"[BiometricEngine] Error en inferencia facial UniFace: {e}")

    return None

def cosine_similarity(a, b):
    """Calcula la similitud de coseno entre dos embeddings."""
    if a is None or b is None:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))

def _reload_cache():
    """Carga los embeddings de la base de datos a memoria RAM para comparaciones instantáneas."""
    global _db_cache
    db = SessionLocal()
    try:
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from src.cloud_backend.database import SesionPersona

        records = db.query(SesionPersona).all()
        cache = []
        for p in records:
            cara_vec = None
            ropa_vec = None
            if p.cara_embedding:
                try:
                    cara_vec = np.array(json.loads(p.cara_embedding), dtype=np.float32)
                except Exception:
                    pass
            if p.ropa_embedding:
                try:
                    ropa_vec = np.array(json.loads(p.ropa_embedding), dtype=np.float32)
                except Exception:
                    pass
            if cara_vec is not None or ropa_vec is not None:
                cache.append({
                    "id_persona": p.id_persona,
                    "nombre": p.nombre_reconocido,
                    "cara": cara_vec,
                    "ropa": ropa_vec
                })
        _db_cache = cache
    finally:
        db.close()

def identify_person_in_db(embedding_dict, threshold_cara=0.50, threshold_ropa=0.90):
    """
    Compara el embedding extraído con las identidades conocidas.
    Prioriza el rostro (ArcFace) con umbral estándar de 0.50.
    """
    global _db_cache
    if _db_cache is None:
        _reload_cache()

    if not _db_cache:
        return None, None

    query_cara = np.array(embedding_dict["cara"], dtype=np.float32) if embedding_dict.get("cara") else None
    query_ropa = np.array(embedding_dict["ropa"], dtype=np.float32) if embedding_dict.get("ropa") else None

    best_match_id = None
    best_match_name = None
    best_score = -1.0

    for item in _db_cache:
        score = 0.0
        # 1. Comparación por cara (ArcFace)
        if query_cara is not None and item["cara"] is not None:
            c_score = float(np.dot(query_cara, item["cara"]) / (np.linalg.norm(query_cara) * np.linalg.norm(item["cara"])))
            if c_score > threshold_cara and c_score > score:
                score = c_score
        # 2. Respaldo por ropa (si aplica)
        elif query_ropa is not None and item["ropa"] is not None:
            r_score = float(np.dot(query_ropa, item["ropa"]) / (np.linalg.norm(query_ropa) * np.linalg.norm(item["ropa"])))
            if r_score > threshold_ropa and r_score > score:
                score = r_score

        if score > best_score:
            best_score = score
            best_match_id = item["id_persona"]
            best_match_name = item["nombre"]

    if best_match_id is not None and best_score > 0:
        return best_match_id, best_match_name

    return None, None

def register_new_face(embedding_dict=None, nombre=None, embedding=None):
    """
    Registra una nueva identidad facial en la base de datos y actualiza el caché en memoria.
    Soporta tanto 'embedding_dict' como el alias 'embedding' para compatibilidad total con ivoa_engine.
    """
    global _db_cache
    target_dict = embedding_dict if embedding_dict is not None else embedding
    if not target_dict:
        return None

    db = SessionLocal()
    try:
        if BASE_DIR not in sys.path:
            sys.path.insert(0, BASE_DIR)
        from src.cloud_backend.database import SesionPersona

        persona = SesionPersona(estado="detectado")
        if target_dict.get("cara"):
            persona.cara_embedding = json.dumps(target_dict["cara"])
        if target_dict.get("ropa"):
            persona.ropa_embedding = json.dumps(target_dict["ropa"])
        if nombre:
            persona.nombre_reconocido = nombre

        db.add(persona)
        db.commit()
        db.refresh(persona)
        
        # Invalidar caché para que la nueva identidad esté disponible de inmediato
        _reload_cache()
        return persona.id_persona
    except Exception as e:
        db.rollback()
        print(f"[BiometricEngine] Error al registrar nueva cara: {e}")
        return None
    finally:
        db.close()
