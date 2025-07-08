from pydantic import BaseModel, Field
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
    korean = "ko"
    russian = "ru"
    arabic = "ar"
    hindi = "hi"
    turkish = "tr"
    dutch = "nl"
    swedish = "sv"
    danish = "da"
    norwegian = "no"
    finnish = "fi"
    polish = "pl"
    czech = "cs"
    hungarian = "hu"
    greek = "el"
    romanian = "ro"
    bulgarian = "bg"
    ukrainian = "uk"
    thai = "th"
    vietnamese = "vi"
    indonesian = "id"

class DocumentCreate(BaseModel):
    filename: str
    source_url: Optional[str] = None
    publisher: Optional[str] = None
    publication_year: Optional[int] = None
    language: Optional[LanguageEnum] = LanguageEnum.spanish


class DocumentResponse(DocumentCreate):
    id: int
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    preview_image_url: Optional[str] = None
    status: str

