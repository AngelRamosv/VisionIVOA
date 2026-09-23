import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SQLITE_PATH = os.path.join(BASE_DIR, "ivoa.db")
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_PATH}")

engine_args = {}
if DATABASE_URL.startswith("sqlite"):
    engine_args["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class SesionPersona(Base):
    __tablename__ = "sesiones_personas"

    id_persona = Column(Integer, primary_key=True, index=True)
    primera_vez_visto = Column(DateTime, nullable=True)
    entro_a_fila = Column(DateTime, nullable=True)
    salio_de_fila = Column(DateTime, nullable=True)
    entro_a_modulo = Column(DateTime, nullable=True)
    salio_de_modulo = Column(DateTime, nullable=True)
    estado = Column(String, default="detectado")
    tiempo_total_atencion_segundos = Column(Integer, nullable=True)
    ropa_embedding = Column(String, nullable=True) # Histograma de ropa JSON
    cara_embedding = Column(String, nullable=True) # ArcFace embedding JSON
    nombre_reconocido = Column(String, nullable=True)

class Evento(Base):
    __tablename__ = "eventos"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    event_type = Column(String, index=True)
    timestamp = Column(DateTime)
    details = Column(String)  # Se guardará como JSON string
    severity = Column(String)

class Hallazgo(Base):
    __tablename__ = "hallazgos"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    evento_id = Column(Integer, ForeignKey('eventos.id'))
    categoria = Column(String, index=True)
    estado = Column(String, default="DETECTADO", index=True)
    severidad = Column(String)
    fecha_creacion = Column(DateTime)
    fecha_vencimiento = Column(DateTime)
    asignado_a = Column(String, nullable=True)
    fecha_cierre = Column(DateTime, nullable=True)

class Usuario(Base):
    __tablename__ = "usuarios"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Función para sembrar el admin
def seed_admin():
    db = SessionLocal()
    try:
        admin = db.query(Usuario).filter(Usuario.username == "admin").first()
        if not admin:
            nuevo_admin = Usuario(username="admin", password="admin123")
            db.add(nuevo_admin)
            db.commit()
    finally:
        db.close()

# Sembrar admin al cargar el módulo
seed_admin()
