# app/modules/document_management/document_service.py
import logging
import re
from sqlalchemy.orm import Session
import cloudinary
import cloudinary.uploader
from pinecone import Pinecone

from app.db.crud import get_document, delete_document_by_id
from app.core.config import settings

logger = logging.getLogger(__name__)

# --- Inicialización de Servicios Externos ---
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)

def _get_cloudinary_public_id(url: str) -> str | None:
    """Extrae el public_id de una URL de Cloudinary."""
    if not url:
        return None
    match = re.search(r"/v\d+/(.+?)(?:\.pdf|\.png|\.jpg)?$", url)
    return match.group(1) if match else None

def delete_document(db: Session, document_id: int, namespace: str) -> bool:
    """
    Orquesta la eliminación de un documento y sus activos asociados (Postgres, Cloudinary, Pinecone).
    """
    logger.info(f"Iniciando eliminación del documento ID: {document_id} en el namespace: {namespace}")
    
    # Necesitamos el kb_id del documento para la consulta, pero el namespace es suficiente para Pinecone
    # Asumimos que el documento existe y que el namespace es correcto.
    # Primero, obtenemos el documento de la BD para sus URLs.
    # Nota: Esta lógica asume que el kb_id se puede inferir si es necesario, pero aquí usamos el namespace.
    db_document = db.query(Document).filter(Document.id == document_id).first()
    if not db_document:
        logger.warning(f"Documento con ID {document_id} no encontrado.")
        return False

    # 1. Eliminar de Cloudinary
    if db_document.preview_image_url:
        public_id = _get_cloudinary_public_id(db_document.preview_image_url)
        if public_id:
            try:
                cloudinary.uploader.destroy(public_id, resource_type="image")
                logger.info(f"Imagen eliminada de Cloudinary: {public_id}")
            except Exception as e:
                logger.error(f"Error al eliminar imagen de Cloudinary: {e}")

    # 2. Eliminar de Pinecone por metadata
    try:
        pinecone_index.delete(filter={"document_id": document_id}, namespace=namespace)
        logger.info(f"Vectores para el documento {document_id} eliminados de Pinecone (namespace: {namespace}).")
    except Exception as e:
        logger.error(f"Error al eliminar vectores de Pinecone: {e}")

    # 3. Eliminar de PostgreSQL (usando la función del CRUD)
    if delete_document_by_id(db, document_id, db_document.kb_id):
        logger.info(f"Documento {document_id} eliminado de PostgreSQL.")
        return True
    else:
        logger.error(f"Error al eliminar el documento {document_id} de PostgreSQL.")
        return False

def check_pinecone_vectors_exist(document_id: int, namespace: str) -> bool:
    """Verifica si un documento tiene vectores asociados en Pinecone."""
    try:
        response = pinecone_index.query(
            vector=[0.0] * 1536, # Vector dummy
            top_k=1,
            namespace=namespace,
            filter={"document_id": document_id}
        )
        return len(response.matches) > 0
    except Exception as e:
        logger.error(f"Error al verificar vectores en Pinecone: {e}", exc_info=True)
        return False