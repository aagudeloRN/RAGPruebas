# app/db/crud_kb.py
from sqlalchemy.orm import Session
from app.models.knowledge_base import KnowledgeBase
from app.schemas.knowledge_base import KnowledgeBaseCreate

def get_knowledge_base(db: Session, kb_id: str) -> KnowledgeBase | None:
    """Recupera una Base de Conocimiento por su ID."""
    return db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()

def get_all_kbs(db: Session):
    """Recupera todas las Bases de Conocimiento."""
    return db.query(KnowledgeBase).all()

def create_kb(db: Session, kb: KnowledgeBaseCreate) -> KnowledgeBase:
    """
    Crea una nueva Base de Conocimiento.
    El namespace de Pinecone se deriva del ID para asegurar consistencia.
    """
    pinecone_namespace = f"rag-factory-{kb.id}"
    
    db_kb = KnowledgeBase(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        pinecone_namespace=pinecone_namespace,
        is_active=False # Las KBs nuevas nunca son activas por defecto
    )
    db.add(db_kb)
    db.commit()
    db.refresh(db_kb)
    return db_kb

def set_active_kb(db: Session, kb_id: str) -> KnowledgeBase | None:
    """Establece una KB como activa y desactiva todas las demás."""
    # Desactivar la que esté activa actualmente
    db.query(KnowledgeBase).filter(KnowledgeBase.is_active == True).update({"is_active": False})
    
    # Activar la nueva
    kb_to_activate = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if kb_to_activate:
        kb_to_activate.is_active = True
        db.commit()
        db.refresh(kb_to_activate)
    
    return kb_to_activate
