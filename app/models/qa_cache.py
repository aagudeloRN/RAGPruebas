from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from app.db.base import Base

class QACache(Base):
    __tablename__ = "qa_cache"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(String, unique=True, index=True, nullable=False)
    answer = Column(Text, nullable=False)
    context = Column(Text, nullable=True)
    embedding = Column(Text, nullable=True) # Para almacenar el embedding de la pregunta
    embedding_model = Column(String, nullable=True) # Para registrar el modelo de embedding usado
    hit_count = Column(Integer, default=0, nullable=False) # Contador de aciertos para el top Q&A
    created_at = Column(DateTime(timezone=True), server_default=func.now())
