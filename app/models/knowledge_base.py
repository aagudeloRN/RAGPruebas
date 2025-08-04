# app/models/knowledge_base.py
from sqlalchemy import Column, String, Text, Boolean
from sqlalchemy.orm import relationship
from app.db.base import Base

class KnowledgeBase(Base):
    __tablename__ = 'knowledge_bases'

    id = Column(String, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    pinecone_namespace = Column(String, nullable=False, unique=True)
    is_active = Column(Boolean, default=False, nullable=False)

    documents = relationship("Document", back_populates="knowledge_base")
    qa_cache = relationship("QACache", back_populates="knowledge_base")
