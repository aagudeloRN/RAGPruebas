# app/modules/data_ingestion/ingestion_service.py
import os
import uuid
import logging
from typing import Dict, Any
from fastapi import UploadFile, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import shutil

from .pipeline import upload_and_extract_metadata, process_document_for_rag
from app.db.crud import create_document, update_document_processing_results, get_document
from app.schemas.document import DocumentCreateRequest, DocumentResponse

logger = logging.getLogger(__name__)

TEMP_FILE_DIR = "/tmp/rag_uploads"
os.makedirs(TEMP_FILE_DIR, exist_ok=True)

async def handle_upload_document(file: UploadFile, db: Session, kb_id: str) -> Dict[str, Any]:
    """
    Guarda un PDF, extrae metadatos y crea un registro preliminar en la BD para una KB específica.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")
    
    file_bytes = await file.read()
    temp_filename = f"{uuid.uuid4()}-{file.filename}"
    temp_file_path = os.path.join(TEMP_FILE_DIR, temp_filename)
    with open(temp_file_path, "wb") as f:
        f.write(file_bytes)

    try:
        all_metadata = await upload_and_extract_metadata(file_bytes=file_bytes, filename=file.filename)
    except Exception as e:
        logger.error(f"Error en la extracción de metadatos: {e}", exc_info=True)
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo: {e}")

    try:
        db_safe_metadata = {
            "filename": file.filename,
            "title": all_metadata.get("title"),
            "publisher": all_metadata.get("publisher"),
            "publication_year": all_metadata.get("publication_year"),
            "language": all_metadata.get("language"),
            "summary": all_metadata.get("summary"),
            "keywords": all_metadata.get("keywords", []),
            "status": "awaiting_validation",
        }
        # Pasamos el kb_id al crear el documento
        db_document = create_document(db=db, document_data=db_safe_metadata, kb_id=kb_id)
        
        final_temp_path = os.path.join(TEMP_FILE_DIR, f"{db_document.id}.pdf")
        shutil.move(temp_file_path, final_temp_path)

        return {
            "document_id": db_document.id,
            "metadata": {
                **DocumentResponse.from_orm(db_document).model_dump(),
                "cover_image_url": all_metadata.get("cover_image_url"),
                "is_ocr": all_metadata.get("is_ocr", False),
                "processing_notes": all_metadata.get("processing_notes", "")
            }
        }
    except Exception as e:
        db.rollback()
        os.remove(temp_file_path)
        logger.error(f"Error al crear el registro del documento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error al guardar el documento en la BD.")

async def handle_process_document(
    document_id: int, background_tasks: BackgroundTasks,
    document_data: DocumentCreateRequest, db: Session, pinecone_namespace: str
) -> DocumentResponse:
    """
    Actualiza metadatos y lanza el procesamiento RAG en segundo plano para el namespace correcto.
    """
    # El kb_id se infiere del documento, no se pasa como argumento
    db_document = db.query(Document).filter(Document.id == document_id).first()
    if not db_document:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    temp_file_path = os.path.join(TEMP_FILE_DIR, f"{document_id}.pdf")
    if not os.path.exists(temp_file_path):
        raise HTTPException(status_code=404, detail="Archivo temporal no encontrado. Suba de nuevo.")

    try:
        update_data_db = document_data.model_dump(exclude_unset=True)
        update_data_db["status"] = "processing"
        
        updated_document = update_document_processing_results(
            db, document_id, db_document.kb_id, update_data_db
        )

        background_tasks.add_task(
            process_document_for_rag,
            document_id=document_id,
            namespace=pinecone_namespace, # Usamos el namespace de la KB activa
            temp_file_path=temp_file_path,
            validated_data=document_data.model_dump()
        )

        return updated_document
    except Exception as e:
        db.rollback()
        logger.error(f"Error al procesar documento {document_id}: {e}", exc_info=True)
        # Asegurarse de que el kb_id se pasa a la función de actualización
        update_document_processing_results(db, document_id, db_document.kb_id, {"status": "failed"})
        raise HTTPException(status_code=500, detail=f"Error al procesar el documento: {e}")