from sqlalchemy.ext.declarative import declarative_base

# This is the base class that all SQLAlchemy models will inherit.
Base = declarative_base()

# Import all models here to ensure they are registered with SQLAlchemy's metadata
from app.models.knowledge_base import KnowledgeBase
from app.models.document import Document
from app.models.qa_cache import QACache