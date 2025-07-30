from sqlalchemy.orm import Session
from app.models.document import Document
from typing import Dict, Any

def get_document(db: Session, document_id: int) -> Document:
    return db.query(Document).filter(Document.id == document_id).first()

def create_document(db: Session, document_data: dict) -> Document:
    db_document = Document(**document_data)
    db.add(db_document)
    db.commit()
    db.refresh(db_document)
    return db_document

def update_document_processing_results(db: Session, document_id: int, results: Dict[str, Any]) -> Document:
    db_document = get_document(db, document_id)
    if db_document:
        for key, value in results.items():
            setattr(db_document, key, value)
        db.commit()
        db.refresh(db_document)
    return db_document

def delete_document(db: Session, document_id: int):
    db_document = get_document(db, document_id)
    if db_document:
        db.delete(db_document)
        db.commit()
    return db_document
