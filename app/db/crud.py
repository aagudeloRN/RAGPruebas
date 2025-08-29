# app/db/crud.py
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from datetime import datetime, timedelta
from app.models.document import Document
from typing import Dict, Any, List, Optional

def get_document_status(db: Session, document_id: int, kb_id: str) -> Optional[str]:
    """Obtiene el estado de un documento por su ID, dentro de una KB específica."""
    document = db.query(Document.status).filter(Document.id == document_id, Document.kb_id == kb_id).first()
    return document.status if document else None

def get_document(db: Session, document_id: int, kb_id: str) -> Optional[Document]:
    """Obtiene un documento por su ID, dentro de una KB específica."""
    return db.query(Document).filter(Document.id == document_id, Document.kb_id == kb_id).first()

def get_documents(
    db: Session, kb_id: str, skip: int = 0, limit: int = 100,
    sort_by: str = "id", sort_order: str = "desc",
    keyword: Optional[str] = None, publisher_filter: Optional[str] = None,
    year_start: Optional[int] = None, year_end: Optional[int] = None
) -> List[Document]:
    """Recupera una lista de documentos para una KB específica."""
    query = db.query(Document).filter(Document.kb_id == kb_id)

    # Aplicar filtros
    if keyword:
        search_keyword = f"%{keyword.lower()}%"
        query = query.filter(
            or_(
                Document.title.ilike(search_keyword),
                Document.summary.ilike(search_keyword),
                Document.publisher.ilike(search_keyword),
                Document.keywords.any(search_keyword)
            )
        )
    if publisher_filter:
        query = query.filter(Document.publisher.ilike(f"%{publisher_filter.lower()}%" ))
    if year_start:
        query = query.filter(Document.publication_year >= year_start)
    if year_end:
        query = query.filter(Document.publication_year <= year_end)

    # Aplicar ordenación
    order_column = getattr(Document, sort_by, Document.id)
    if sort_order == "asc":
        query = query.order_by(order_column.asc())
    else:
        query = query.order_by(order_column.desc())

    return query.offset(skip).limit(limit).all()

def create_document(db: Session, document_data: dict, kb_id: str) -> Document:
    """Crea un nuevo documento asociado a una KB específica."""
    db_document = Document(**document_data, kb_id=kb_id)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def update_document_processing_results(db: Session, kb_id: str, document_id: int, results: dict):
    """Actualiza un documento con los resultados del procesamiento RAG."""
    db_document = db.query(Document).filter(Document.id == document_id, Document.kb_id == kb_id).first()
    if db_document:
        for key, value in results.items():
            setattr(db_document, key, value)
        db.commit()
        db.refresh(db_document)
    return db_document

def update_document_status(db: Session, document_id: int, status: str, status_message: str):
    """Actualiza el estado y el mensaje de estado de un documento."""
    db_document = db.query(Document).filter(Document.id == document_id).first()
    if db_document:
        db_document.status = status
        db_document.status_message = status_message
        db.commit()
        db.refresh(db_document)
    return db_document


def get_unique_publishers(db: Session, kb_id: str, search_term: Optional[str] = None) -> List[str]:
    """Obtiene publicadores únicos para una KB específica."""
    query = db.query(Document.publisher).filter(Document.kb_id == kb_id).distinct()
    if search_term:
        query = query.filter(Document.publisher.ilike(f"%{search_term.lower()}%" ))
    return [p[0] for p in query.all() if p[0] and p[0].strip()]

def delete_document_by_id(db: Session, document_id: int, kb_id: str) -> bool:
    """Elimina un documento por su ID, dentro de una KB específica."""
    db_document = get_document(db, document_id, kb_id)
    if db_document:
        db.delete(db_document)
        db.commit()
        return True
    return False