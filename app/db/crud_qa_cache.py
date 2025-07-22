from sqlalchemy.orm import Session
from app.models.qa_cache import QACache
from app.schemas.qa_cache import QACacheCreate, QACache as QACacheSchema

def get_qa_cache_by_question(db: Session, *, question: str) -> QACache | None:
    return db.query(QACache).filter(QACache.question == question).first()

def create_qa_cache(db: Session, *, qa_in: QACacheCreate) -> QACache:
    db_obj = QACache(
        question=qa_in.question,
        answer=qa_in.answer,
        context=qa_in.context
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    return db_obj
