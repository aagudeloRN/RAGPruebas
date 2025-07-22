from sqlalchemy.orm import Session
from sqlalchemy import or_
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

def get_documents(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "id",
    sort_order: str = "desc",
    keyword: Optional[str] = None,
    publisher_filter: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None
) -> List[Document]:
    """Recupera una lista de documentos, con paginación, ordenación y filtrado."""
    query = db.query(Document)

    # Aplicar filtros
    if keyword:
        search_keyword = f"%{keyword.lower()}%"
        query = query.filter(
            or_(
                Document.title.ilike(search_keyword),
                Document.summary.ilike(search_keyword),
                Document.publisher.ilike(search_keyword),
                Document.keywords.any(search_keyword) # Assuming keywords is an array of strings
            )
        )
    if publisher_filter:
        query = query.filter(Document.publisher.ilike(f"%{publisher_filter.lower()}%" ))
    if year_start:
        query = query.filter(Document.publication_year >= year_start)
    if year_end:
        query = query.filter(Document.publication_year <= year_end)

    # Aplicar ordenación
    if sort_by == "title":
        if sort_order == "asc":
            query = query.order_by(Document.title.asc())
        else:
            query = query.order_by(Document.title.desc())
    elif sort_by == "publication_year":
        if sort_order == "asc":
            query = query.order_by(Document.publication_year.asc())
        else:
            query = query.order_by(Document.publication_year.desc())
    else: # Default to id
        if sort_order == "asc":
            query = query.order_by(Document.id.asc())
        else:
            query = query.order_by(Document.id.desc())

    return query.offset(skip).limit(limit).all()

def update_document_processing_results(db: Session, document_id: int, results: dict) -> Document:
    db_document = db.query(Document).filter(Document.id == document_id).first()
    if db_document:
        for key, value in results.items():
            setattr(db_document, key, value)
        db.commit()
        db.refresh(db_document)
    return db_document

def delete_document(db: Session, document_id: int):
    """Elimina un documento de la base de datos por su ID."""
    db_document = db.query(Document).filter(Document.id == document_id).first()
    if db_document:
        db.delete(db_document)
        db.commit()

def get_unique_publishers(db: Session, search_term: Optional[str] = None) -> List[str]:
    """Obtiene una lista de publicadores únicos, opcionalmente filtrados por un término de búsqueda."""
    query = db.query(Document.publisher).distinct()
    if search_term:
        query = query.filter(Document.publisher.ilike(f"%{search_term.lower()}%" ))
    # Filter out None or empty strings and return as a list
    return [p for p in query.all() if p[0] is not None and p[0].strip() != ""]