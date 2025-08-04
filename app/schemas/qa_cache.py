from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional

class QACacheBase(BaseModel):
    question: str
    answer: str
    context: str | None = None
    context_chunks: Optional[List[str]] = None

class QACacheCreate(QACacheBase):
    embedding: str | None = None
    embedding_model: str | None = None

class QACache(QACacheBase):
    id: int
    embedding: List[float] | None = None
    embedding_model: str | None = None
    hit_count: int
    created_at: datetime

    class Config:
        model_config = {
        "from_attributes": True
    }
