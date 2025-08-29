
import logging
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from app.core.config import active_kb

from app.db.crud import update_document_status, get_document

from app.modules.data_ingestion_v2.pipeline_v2 import IngestionPipelineV2
from app.schemas.document import DocumentCreateRequest

logger = logging.getLogger(__name__)

def run_ingestion_v2_task(document_id: int, doc_path: str, metadata: dict, namespace: str, db: Session):
    """
    Tarea que se ejecuta en segundo plano para el pipeline de ingesta V2.
    """
    try:
        logger.info(f"[V2] Iniciando tarea de ingesta para el documento ID: {document_id}")
        update_document_status(db, document_id=document_id, status=DocumentStatus.processing, status_message="Iniciando pipeline V2...")

        pipeline = IngestionPipelineV2(doc_path=doc_path, metadata=metadata, namespace=namespace)
        # El método run del pipeline V2 ahora es asíncrono, pero lo llamamos desde un entorno síncrono de background task.
        # En un escenario ideal, se usaría un worker asíncrono como Celery con un broker.
        # Por ahora, lo ejecutamos de forma simple.
        import asyncio
        asyncio.run(pipeline.run())

        update_document_status(db, document_id=document_id, status="completed", status_message="Ingesta V2 completada exitosamente.")
        logger.info(f"[V2] Tarea de ingesta completada para el documento ID: {document_id}")

    except Exception as e:
        error_message = f"Error en el pipeline de ingesta V2: {e}"
        logger.error(error_message, exc_info=True)
        update_document_status(db, document_id=document_id, status="failed", status_message=error_message)

async def handle_ingestion_v2(
    document_id: int, 
    background_tasks: BackgroundTasks,
    document_data: DocumentCreateRequest, 
    db: Session,
    pinecone_namespace: str
):
    """
    Maneja la solicitud de ingesta para el pipeline V2.
    """
    doc = get_document(db, document_id=document_id, kb_id=active_kb.id)
    if not doc or not doc.file_path:
        logger.error(f"No se encontró el documento o su ruta para el ID: {document_id}")
        return

    metadata = {
        "document_id": str(doc.id),
        "title": document_data.title,
        "publisher": document_data.publisher,
        "publication_year": str(document_data.publication_year)
    }

    background_tasks.add_task(
        run_ingestion_v2_task, 
        document_id=doc.id, 
        doc_path=doc.file_path, 
        metadata=metadata, 
        namespace=pinecone_namespace,
        db=db
    )
    
    logger.info(f"[V2] Tarea de ingesta para el documento {doc.id} añadida a segundo plano.")
    return doc
