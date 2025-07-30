
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

async def handle_upload_document(file: UploadFile, db: Session) -> Dict[str, Any]:
    """
    Guarda un PDF temporalmente, extrae metadatos y crea un registro preliminar en la BD.
    Devuelve todos los metadatos (incluidos los que no están en la BD) para validación.
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
        # Ahora que el modelo de BD es correcto, podemos guardar todos los metadatos directamente.
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
        db_document = create_document(db=db, document_data=db_safe_metadata)
        db.commit()
        db.refresh(db_document)

        # Renombrar el archivo temporal con el ID de la BD para encontrarlo después
        final_temp_path = os.path.join(TEMP_FILE_DIR, f"{db_document.id}.pdf")
        shutil.move(temp_file_path, final_temp_path)

        # Devolver el ID y los metadatos para la validación
        # Usamos DocumentResponse para asegurar que la estructura sea la correcta
        return {
            "document_id": db_document.id,
            "metadata": DocumentResponse.from_orm(db_document).model_dump()
        }
    except Exception as e:
        db.rollback()
        os.remove(temp_file_path)
        logger.error(f"Error al crear el registro inicial del documento: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error al guardar el documento en la base de datos.")

async def handle_process_document(
    document_id: int,
    background_tasks: BackgroundTasks,
    document_data: DocumentCreateRequest,
    db: Session,
    selected_kb_namespace: str
) -> DocumentResponse:
    """
    Actualiza el documento con metadatos validados y lanza el procesamiento RAG en segundo plano.
    """
    db_document = get_document(db=db, document_id=document_id)
    if not db_document:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")

    temp_file_path = os.path.join(TEMP_FILE_DIR, f"{document_id}.pdf")
    if not os.path.exists(temp_file_path):
        raise HTTPException(status_code=404, detail="Archivo temporal no encontrado. Por favor, suba el documento de nuevo.")

    try:
        # Actualiza la BD con los datos validados que SÍ pertenecen al modelo
        update_data_db = {
            "title": document_data.title,
            "publisher": document_data.publisher,
            "publication_year": document_data.publication_year,
            "source_url": document_data.source_url,
            "language": document_data.language,
            "status": "processing"
        }
        updated_document = update_document_processing_results(db, document_id, update_data_db)
        db.commit()
        db.refresh(updated_document)

        # Lanza la tarea de fondo con TODOS los datos, incluidos los que no están en la BD
        background_tasks.add_task(
            process_document_for_rag,
            document_id=document_id,
            namespace=selected_kb_namespace,
            temp_file_path=temp_file_path,
            validated_data=document_data.model_dump() # Pasa todos los datos validados
        )

        return updated_document
    except Exception as e:
        db.rollback()
        logger.error(f"Error al iniciar el procesamiento final del documento {document_id}: {e}", exc_info=True)
        update_document_processing_results(db, document_id, {"status": "failed", "processing_notes": "Error al iniciar el procesamiento."})
        db.commit()
        raise HTTPException(status_code=500, detail=f"Error al procesar el documento: {e}")
