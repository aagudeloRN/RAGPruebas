from sqlalchemy import Column, Integer, String, Text, ARRAY, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base

class Document(Base):
    __tablename__ = "documents" # Cambiado de Repositori_Oficial a documents para estandarizar

    id = Column(Integer, primary_key=True, index=True)
    kb_id = Column(String, ForeignKey('knowledge_bases.id'), nullable=False, index=True)
    title = Column(String, index=True, nullable=True)
    filename = Column(Text, nullable=False)
    source_url = Column(Text, nullable=True)
    publisher = Column(Text, nullable=True)
    publication_year = Column(Integer, nullable=True)
    language = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    keywords = Column(ARRAY(String), nullable=True)
    cover_image_url = Column(Text, nullable=True)
    status = Column(String(50), default='processing', nullable=False)

    knowledge_base = relationship("KnowledgeBase", back_populates="documents")
