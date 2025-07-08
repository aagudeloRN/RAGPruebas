from sqlalchemy.orm import Session
from app.models.document import Document

def create_document(db: Session, document_data: dict) -> Document:
    db_document = Document(**document_data)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def update_document_processing_results(db: Session, document_id: int, results: dict) -> Document:
    db_document = db.query(Document).filter(Document.id == document_id).first()
    if db_document:
        for key, value in results.items():
            setattr(db_document, key, value)
        db.commit()
        db.refresh(db_document)
    return db_document