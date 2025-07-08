from sqlalchemy import Column, Integer, String, Text, ARRAY
from app.db.base_class import Base

class Document(Base):
    __tablename__ = "Repositori_Oficial"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(Text, nullable=False)
    source_url = Column(Text, nullable=True)
    publisher = Column(Text, nullable=True)
    publication_year = Column(Integer, nullable=True)
    language = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    keywords = Column(ARRAY(String), nullable=True) # Cambiado a ARRAY(String) para coincidir con el schema
    preview_image_url = Column(Text, nullable=True)
    status = Column(String(50), default='processing', nullable=False)