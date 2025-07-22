from pydantic import BaseModel
from datetime import datetime

class QACacheBase(BaseModel):
    question: str
    answer: str
    context: str | None = None

class QACacheCreate(QACacheBase):
    pass

class QACache(QACacheBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True
