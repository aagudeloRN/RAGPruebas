# app/schemas/knowledge_base.py
from pydantic import BaseModel, Field

class KnowledgeBaseBase(BaseModel):
    """Schema base con los campos comunes para una Base de Conocimiento."""
    id: str = Field(..., description="Identificador único para la KB (e.g., 'tendencias', 'procedimientos'). No se puede cambiar.")
    name: str = Field(..., description="Nombre legible para humanos (e.g., 'Informes de Tendencias').")
    description: str | None = Field(None, description="Descripción detallada del contenido de la KB.")

class KnowledgeBaseCreate(KnowledgeBaseBase):
    """Schema usado para crear una nueva Base de Conocimiento."""
    pass # No hay campos adicionales para la creación

class KnowledgeBaseUpdate(BaseModel):
    """Schema usado para actualizar una Base de Conocimiento existente."""
    name: str | None = None
    description: str | None = None

class KnowledgeBase(KnowledgeBaseBase):
    """Schema principal para representar una KB, incluyendo campos de la BD."""
    is_active: bool
    pinecone_namespace: str

    model_config = {
        "from_attributes": True
    }
