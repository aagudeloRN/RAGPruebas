from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class LanguageEnum(str, Enum):
    spanish = "es"
    english = "en"
    french = "fr"
    portuguese = "pt"
    german = "de"
    chinese = "zh"
    italian = "it"
    japanese = "ja"

class DocumentBase(BaseModel):
    title: Optional[str] = None
    filename: str
    source_url: Optional[str] = None
    publisher: Optional[str] = None
    publication_year: Optional[int] = None
    language: Optional[str] = None

class DocumentCreate(DocumentBase):
    pass


class DocumentResponse(DocumentBase):
    id: int
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    preview_image_url: Optional[str] = None
    status: str
    processing_error: Optional[str] = None

    class Config:
        from_attributes = True

# Schema para la respuesta del endpoint de status
class DocumentStatusResponse(BaseModel):
    status: str

# --- Schemas para la consulta RAG ---
class QueryRequest(BaseModel):
    query: str

class Source(BaseModel):
    id: int
    title: Optional[str]
    publisher: Optional[str]
    publication_year: Optional[int]
    source_url: Optional[str]

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]