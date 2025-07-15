from sqlalchemy.orm import Session
from app.models.document import Document
from typing import List, Optional

def create_document(db: Session, document_data: dict) -> Document:
    db_document = Document(**document_data)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def get_document_status(db: Session, document_id: int) -> Optional[str]:
    """Obtiene el estado de un documento por su ID."""
    document = db.query(Document.status).filter(Document.id == document_id).first()
    if not document:
        return None
    return document.status

def get_document(db: Session, document_id: int) -> Optional[Document]:
    """Obtiene un documento por su ID."""
    return db.query(Document).filter(Document.id == document_id).first()

def get_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
    """Recupera una lista de documentos, con paginación."""
    return db.query(Document).order_by(Document.id.desc()).offset(skip).limit(limit).all()

def update_document_processing_results(db: Session, document_id: int, results: dict) -> Document:
    db_document = db.query(Document).filter(Document.id == document_id).first()
    if db_document:
        for key, value in results.items():
            setattr(db_document, key, value)
        db.commit()
        db.refresh(db_document)
    return db_document