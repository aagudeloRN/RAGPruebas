# app/core/config.py
import os
from functools import lru_cache
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# --- Carga de Variables de Entorno Globales ---
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

class GlobalSettings(BaseSettings):
    """Configuraciones globales de la aplicación que no cambian entre KBs."""
    DATABASE_URL: str
    DATABASE_URL_PGVECTOR: str
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    COHERE_API_KEY: str
    GOOGLE_API_KEY: str
    SECRET_KEY: str # Nueva clave secreta para sesiones

    # --- Variables con valores por defecto ---
    PINECONE_INDEX_NAME: str = "rag-factory-index" # Nuevo nombre estandarizado
    PINECONE_BATCH_SIZE: int = 100
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_LLM_MODEL: str = "gpt-4o"

    # Variable para seleccionar la KB activa en el despliegue
    ACTIVE_KB_ID: str = "default"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# --- Configuración de la Base de Conocimiento Activa ---

class ActiveKBConfig(BaseModel):
    """Contiene la configuración de la KB activa, cargada desde la BD."""
    id: str | None = None
    name: str | None = None
    description: str | None = None
    pinecone_namespace: str | None = None

# Objeto global que contendrá la configuración de la KB activa
# Se llena al iniciar la aplicación llamando a load_active_kb_config
active_kb: ActiveKBConfig = ActiveKBConfig()

def load_active_kb_config():
    """
    Carga la configuración de la base de conocimiento activa desde la BD.
    Esta función debe ser llamada en el evento 'startup' de FastAPI.
    """
    from app.db.session import SessionLocal
    from app.models.knowledge_base import KnowledgeBase

    db = SessionLocal()
    try:
        print(f"INFO: Buscando la KB activa en la base de datos...")
        kb_record = db.query(KnowledgeBase).filter(KnowledgeBase.is_active == True).first()
        
        if not kb_record:
            # Fallback: si ninguna está activa, intentar cargar la 'default' o la primera que encuentre
            logger.warning("No se encontró una KB activa. Intentando cargar la KB 'default'.")
            kb_record = db.query(KnowledgeBase).filter(KnowledgeBase.id == 'default').first()
            if not kb_record:
                logger.warning("No se encontró la KB 'default'. Cargando la primera KB disponible.")
                kb_record = db.query(KnowledgeBase).first()
            if not kb_record:
                raise ValueError("No se encontró ninguna Base de Conocimiento en la base de datos.")

        active_kb.id = kb_record.id
        active_kb.name = kb_record.name
        active_kb.description = kb_record.description
        active_kb.pinecone_namespace = kb_record.pinecone_namespace
        
        print(f"INFO: Configuración de KB '{active_kb.name}' cargada exitosamente.")

    finally:
        db.close()

# --- Instancia única de las configuraciones ---

@lru_cache
def get_settings() -> GlobalSettings:
    return GlobalSettings()

settings = get_settings()