from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# --- Base de Datos Principal ---
engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# --- Base de Datos para PGVector (Caché de Q&A) ---
engine_pgvector = create_engine(settings.DATABASE_URL_PGVECTOR, pool_pre_ping=True)
SessionLocalPgVector = sessionmaker(autocommit=False, autoflush=False, bind=engine_pgvector)

def get_db():
    """
    Dependency function for the main database.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_pgvector_db():
    """
    Dependency function for the PGVector database.
    """
    db = SessionLocalPgVector()
    try:
        yield db
    finally:
        db.close()
