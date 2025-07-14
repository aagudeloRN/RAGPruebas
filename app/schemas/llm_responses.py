from pydantic import BaseModel, Field
from typing import List

class DocumentAnalysis(BaseModel):
    """
    Un esquema para estructurar la salida del LLM para el análisis de documentos.
    """
    summary: str = Field(description="Un resumen conciso del documento.")
    keywords: List[str] = Field(description="Una lista de 5 a 10 palabras clave relevantes del documento.")

