
import sys
import os
from sqlalchemy.orm import Session

# Añadir el directorio raíz del proyecto al path para poder importar los módulos de la app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar todos los modelos para que SQLAlchemy los reconozca
from app.models.document import Document
from app.models.knowledge_base import KnowledgeBase
from app.models.qa_cache import QACache

from app.db.session import SessionLocal

def list_documents_from_db():
    """
    Se conecta a la base de datos y lista todos los documentos con su ID, título y nombre de archivo original.
    """
    db: Session = SessionLocal()
    try:
        print("--- Listando todos los documentos en la Base de Datos ---")
        
        # Asumiendo que tienes una KB 'default' o alguna KB con documentos.
        # Si necesitas filtrar por una KB específica, puedes añadir un .filter()
        documents = db.query(Document).order_by(Document.id).all()
        
        if not documents:
            print("No se encontraron documentos en la base de datos.")
            return

        for doc in documents:
            print(f"ID: {doc.id} | Título: {doc.title} | Archivo: {doc.filename} | KB_ID: {doc.kb_id} | URL Fuente: {doc.source_url}")
            
        print("\n--- Fin de la lista ---")

    finally:
        db.close()

if __name__ == "__main__":
    list_documents_from_db()
