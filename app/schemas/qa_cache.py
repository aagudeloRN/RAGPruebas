from pydantic import BaseModel
from datetime import datetime

class QACacheBase(BaseModel):
    question: str
    answer: str
    context: str | None = None

class QACacheCreate(QACacheBase):
    embedding: str | None = None
    embedding_model: str | None = None

class QACache(QACacheBase):
    id: int
    embedding: str | None = None
    embedding_model: str | None = None
    hit_count: int
    created_at: datetime

    class Config:
        orm_mode = True
