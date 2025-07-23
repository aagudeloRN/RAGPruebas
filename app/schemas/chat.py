from pydantic import BaseModel
from typing import List, Optional

from .document import Source # Reutilizamos el schema de las fuentes

class ChatMessage(BaseModel):
    """Representa un único mensaje en el historial del chat."""
    role: str  # "user" o "assistant"
    content: str

class ChatRequest(BaseModel):
    """El cuerpo de la solicitud para el endpoint de chat conversacional."""
    query: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    """La respuesta del endpoint de chat, compatible con la respuesta de consulta existente."""
    answer: str
    sources: List[Source]
    cache_hit: bool = False
