import cv2
import numpy as np
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'data', 'ivoa.db')}")

engine_args = {"connect_args": {"check_same_thread": False}} if DATABASE_URL.startswith("sqlite") else {}
db_engine = create_engine(DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

class DeepReID:
    def __init__(self):
        self.device = torch.device('cpu')
        # Load pre-trained ResNet18 and remove the classification head to get the 512-dim embedding
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*(list(resnet.children())[:-1])).to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, cv2_img):
        rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
        return feat.flatten().numpy().tolist()

_reid_extractor = None

def get_reid():
    global _reid_extractor
    if _reid_extractor is None:
        _reid_extractor = DeepReID()
    return _reid_extractor

def find_face_and_embed(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    pw, ph = x2 - x1, y2 - y1
    if pw < 10 or ph < 10:
        return None
        
    # Extraer el cuerpo de la persona
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return None
        
    result = {"cara": None, "ropa": None}
    
    # Extraer firma profunda con PyTorch (Mucho más robusto que colores)
    try:
        reid = get_reid()
        emb = reid.extract(crop)
        result["ropa"] = emb
    except Exception as e:
        print(f"Error DeepReID: {e}")
        pass
        
    if result["ropa"] is None:
        return None
        
    return result

def cosine_similarity(a, b):
    if a is None or b is None: return 0.0
    a, b = np.array(a), np.array(b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0: return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def identify_person_in_db(embedding_dict, threshold_cara=0.60, threshold_ropa=0.92):
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
