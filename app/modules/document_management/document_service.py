import logging
import re
from sqlalchemy.orm import Session
from sqlalchemy import or_
from typing import List, Optional
import cloudinary
import cloudinary.uploader
from pinecone import Pinecone

from app.models.document import Document
from app.schemas.document import DocumentResponse
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)

def _get_cloudinary_public_id(url: str, resource_type: str = "image") -> Optional[str]:
    """Extracts the public_id from a Cloudinary URL."""
    if not url:
        return None
    # This regex is designed to capture the public_id from various Cloudinary URL formats
    match = re.search(r"/v\d+/(.+?)(?:\.pdf|\.png|\.jpg)?$", url)
    if match:
        # The public_id includes the folder structure
        return match.group(1)
    return None

def get_document_status(db: Session, document_id: int) -> Optional[str]:
    """Obtiene el estado de un documento por su ID."""
    document = db.query(Document.status).filter(Document.id == document_id).first()
    return document.status if document else None

def get_document(db: Session, document_id: int) -> Optional[Document]:
    """Obtiene un documento por su ID."""
    return db.query(Document).filter(Document.id == document_id).first()

def get_documents(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    sort_by: str = "id",
    sort_order: str = "desc",
    keyword: Optional[str] = None,
    publisher_filter: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None
) -> List[Document]:
    """Recupera una lista de documentos, con paginación, ordenación y filtrado."""
    query = db.query(Document)

    if keyword:
        search_keyword = f"%{keyword.lower()}%"
        query = query.filter(
            or_(
                Document.title.ilike(search_keyword),
                Document.summary.ilike(search_keyword),
                Document.publisher.ilike(search_keyword),
                Document.keywords.any(search_keyword)
            )
        )
    if publisher_filter:
        query = query.filter(Document.publisher.ilike(f"%{publisher_filter.lower()}%" ))
    if year_start:
        query = query.filter(Document.publication_year >= year_start)
    if year_end:
        query = query.filter(Document.publication_year <= year_end)

    order_column = getattr(Document, sort_by, Document.id)
    if sort_order == "asc":
        query = query.order_by(order_column.asc())
    else:
        query = query.order_by(order_column.desc())

    return query.offset(skip).limit(limit).all()

def update_document_processing_results(db: Session, document_id: int, results: dict) -> Document:
    db_document = get_document(db, document_id)
    if db_document:
        for key, value in results.items():
            setattr(db_document, key, value)
        db.commit()
        db.refresh(db_document)
    return db_document

def delete_document(db: Session, document_id: int, namespace: Optional[str] = None) -> bool:
    """
    Deletes a document and its associated assets from all services (Postgres, Cloudinary, Pinecone).
    """
    db_document = get_document(db, document_id)
    if not db_document:
        logger.warning(f"Document with ID {document_id} not found for deletion.")
        return False

    # 1. Delete from Cloudinary
    if db_document.preview_image_url:
        public_id = _get_cloudinary_public_id(db_document.preview_image_url, "image")
        if public_id:
            try:
                cloudinary.uploader.destroy(public_id, resource_type="image")
                logger.info(f"Deleted cover image from Cloudinary: {public_id}")
            except Exception as e:
                logger.error(f"Error deleting cover image {public_id} from Cloudinary: {e}")

    if db_document.source_url:
        public_id = _get_cloudinary_public_id(db_document.source_url, "raw")
        if public_id:
            try:
                cloudinary.uploader.destroy(public_id, resource_type="raw")
                logger.info(f"Deleted PDF from Cloudinary: {public_id}")
            except Exception as e:
                logger.error(f"Error deleting PDF {public_id} from Cloudinary: {e}")

    # 2. Delete from Pinecone
    try:
        # Fetch vector IDs by prefix, as we don't know how many chunks there were.
        # This is a workaround. A better approach is to query Pinecone for all vectors with the document_id.
        # As of now, Pinecone doesn't support deleting by metadata filter directly in all tiers.
        # We will assume a max number of chunks for now, which is a limitation.
        max_chunks_to_check = 1000 # A reasonable upper limit
        vector_ids_to_delete = [f"{document_id}-{i}" for i in range(max_chunks_to_check)]
        pinecone_index.delete(ids=vector_ids_to_delete, namespace=namespace)
        logger.info(f"Attempted to delete vectors for document {document_id} from Pinecone namespace {namespace}.")
    except Exception as e:
        logger.error(f"Error deleting vectors for document {document_id} from Pinecone: {e}")

    # 3. Delete from PostgreSQL
    try:
        db.delete(db_document)
        db.commit()
        logger.info(f"Deleted document {document_id} from PostgreSQL.")
        return True
    except Exception as e:
        logger.error(f"Error deleting document {document_id} from PostgreSQL: {e}")
        db.rollback()
        return False

def get_unique_publishers(db: Session, search_term: Optional[str] = None) -> List[str]:
    """Obtiene una lista de publicadores únicos, opcionalmente filtrados por un término de búsqueda."""
    query = db.query(Document.publisher).distinct()
    if search_term:
        query = query.filter(Document.publisher.ilike(f"%{search_term.lower()}%" ))
    return [p[0] for p in query.all() if p and p[0]]

def get_all_documents_for_export(db: Session) -> List[Document]:
    """Recupera todos los documentos de la base de datos para exportación."""
    return db.query(Document).all()

def get_abandoned_documents(db: Session, older_than_hours: int = 24) -> List[Document]:
    """Gets documents that are in 'awaiting_validation' state for more than a specified time."""
    from datetime import datetime, timedelta
    time_threshold = datetime.utcnow() - timedelta(hours=older_than_hours)
    return db.query(Document).filter(
        Document.status == 'awaiting_validation',
        Document.created_at < time_threshold
    ).all()

