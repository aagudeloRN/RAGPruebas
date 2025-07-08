from sqlalchemy import Column, Integer, String, Text, ARRAY
from app.db.session import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    source_url = Column(String(2048), nullable=True)
    publisher = Column(String(255), nullable=True)
    publication_year = Column(Integer, nullable=True)
    language = Column(String(10), nullable=True)
    summary = Column(Text, nullable=True)
    keywords = Column(ARRAY(String), nullable=True)
    preview_image_url = Column(String(2048), nullable=True)
    status = Column(String(50), default='processing', nullable=False)