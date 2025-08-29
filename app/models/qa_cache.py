from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.db.base_class import Base
from pgvector.sqlalchemy import Vector

class QACache(Base):
    __tablename__ = "qa_cache"

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(String, ForeignKey('knowledge_bases.id'), nullable=False, index=True)
    question = Column(String, index=True, nullable=False) # Se quita unique=True para permitir la misma pregunta en diferentes KBs
    answer = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    context_chunks = Column(JSONB, nullable=True)
    embedding = Column(Vector(1536), nullable=True)
    embedding_model = Column(String, nullable=True)
    hit_count = Column(Integer, default=0, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    knowledge_base = relationship("KnowledgeBase", back_populates="qa_cache")
