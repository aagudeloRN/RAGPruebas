
import sys
import os
from sqlalchemy.orm import Session

# Añadir el directorio raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar todos los modelos para que SQLAlchemy los reconozca
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.qa_cache import QACache

from app.db.session import SessionLocal

def clear_qa_cache_table():
    """
    Se conecta a la base de datos y elimina todos los registros de la tabla qa_cache.
    """
    db: Session = SessionLocal()
    try:
        print("--- Limpiando la tabla qa_cache ---")
        
        num_rows_deleted = db.query(QACache).delete()
        db.commit()
        
        print(f"Se eliminaron {num_rows_deleted} registros de la tabla qa_cache.")
        print("--- Limpieza completada ---")

    except Exception as e:
        print(f"Ocurrió un error al limpiar la tabla qa_cache: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    clear_qa_cache_table()
