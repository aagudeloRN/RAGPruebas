from pydantic import BaseModel, Field
from typing import List

class DocumentAnalysis(BaseModel):
    """
    Un esquema para estructurar la salida del LLM para el análisis de documentos.
    """
    title: str = Field(description="El título principal del documento. Obligatorio.")
    publisher: str = Field(description="La entidad o autor que publicó el documento. Obligatorio.")
    publication_year: int = Field(description="El año de publicación del documento. Obligatorio.")
    language: str = Field(description="El idioma principal del documento (código ISO 639-1, ej. 'es', 'en'). Obligatorio.")
    summary: str = Field(description="Un resumen conciso y completo del documento, obligatoriamente en español.")
    keywords: List[str] = Field(description="Una lista de 7 a 10 palabras clave relevantes del documento, obligatoriamente en español, homologadas con el tesauro de la UNESCO, y que no incluyan el nombre del documento o del autor.")

