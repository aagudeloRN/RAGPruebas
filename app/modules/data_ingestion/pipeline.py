import os
import PyPDF2
import logging
import uuid
import json
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from pinecone import Pinecone
from openai import AsyncOpenAI
import fitz  # PyMuPDF
import io
import cloudinary
import cloudinary.uploader
import requests
from fastapi import HTTPException

from app.core.config import settings
from app.db.crud import update_document_processing_results
from app.models.document import Document
from app.db.session import get_db

logger = logging.getLogger(__name__)

# Initialize clients
client_openai = AsyncOpenAI()
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

# Configure Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET
)


async def upload_and_extract_metadata(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Extracts metadata from a PDF file without uploading the PDF itself.
    The PDF file bytes are processed in memory.
    """
    # El PDF ya no se sube aquí. Se manejará temporalmente en el servidor.
    # El `source_url` se establecerá a partir de la entrada del usuario en la validación.
    pdf_url = "local_temp_file" # Placeholder

    text_content = ""
    is_ocr = False
    pdf_stream = io.BytesIO(file_bytes)

    try:
        reader = PyPDF2.PdfReader(pdf_stream)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_content += page_text + "\n"
        if not text_content.strip():
            is_ocr = True
            logger.info(f"PDF {filename} appears to be scanned.")
    except Exception as e:
        logger.warning(f"Error during PDF text extraction for {filename}: {e}. Assuming OCR needed.")
        is_ocr = True

    metadata = {
        "title": "Untitled Document",
        "summary": "No summary available.",
        "publisher": "Unknown",
        "publication_year": None,
        "source_url": pdf_url,
        "keywords": [],
        "language": "unknown",
        "is_ocr": is_ocr,
        "processing_notes": "",
        "cover_image_url": None
    }

    try:
        pdf_stream.seek(0)
        reader = PyPDF2.PdfReader(pdf_stream)
        info = reader.metadata
        if info:
            metadata["title"] = info.get("/Title") or "Untitled Document"
            metadata["publisher"] = info.get("/Author") or "Unknown"
            date_str = info.get("/CreationDate") or info.get("/ModDate")
            if date_str and len(date_str) > 5:
                try:
                    metadata["publication_year"] = int(date_str[2:6])
                except (ValueError, IndexError):
                    pass
    except Exception as e:
        logger.warning(f"Could not extract basic PDF metadata from {filename}: {e}")

    if not is_ocr and text_content.strip():
        try:
            llm_prompt_parts = [
                "Dado el siguiente texto de un documento, extrae la siguiente información en formato JSON:",
                "- 'summary': Un resumen conciso en español que describa principalmente qué se va a encontrar en el documento (máximo 3-4 oraciones).",
                "- 'keywords': Una lista de 5-10 palabras clave relevantes en español.",
                "- 'language': El idioma principal del documento (ej., 'es' para español, 'en' para inglés)."
            ]
            if metadata["title"] == "Untitled Document":
                llm_prompt_parts.append("- 'title': El título más relevante del documento.")
            if metadata["publisher"] == "Unknown":
                llm_prompt_parts.append("- 'publisher': El autor o la organización que publicó el documento.")
            
            llm_prompt_parts.append(f"\nDocument Text:\n{text_content[:4000]}")
            llm_prompt = "\n".join(llm_prompt_parts)

            llm_response = await client_openai.chat.completions.create(
                model=settings.OPENAI_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "Eres un analista de documentos experto. Extrae información en formato JSON."},
                    {"role": "user", "content": llm_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            llm_data = json.loads(llm_response.choices[0].message.content)
            
            metadata.update({k: v for k, v in llm_data.items() if v}) # Update only with non-empty values
            metadata["processing_notes"] += "AI metadata extracted. "
        except Exception as e:
            logger.error(f"Error extracting AI metadata for {filename}: {e}", exc_info=True)
            metadata["processing_notes"] += f"AI metadata extraction failed: {e}. "
    elif is_ocr:
        metadata["processing_notes"] += "Document requires OCR. "
    else:
        metadata["processing_notes"] += "No extractable text for AI metadata. "

    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(150/72, 150/72))
        img_byte_arr = io.BytesIO(pix.tobytes("png"))
        
        cover_upload_result = cloudinary.uploader.upload(img_byte_arr, folder="document_covers")
        metadata["cover_image_url"] = cover_upload_result['secure_url']
        logger.info(f"Cover image for {filename} uploaded to Cloudinary.")
    except Exception as e:
        logger.error(f"Error generating or uploading cover image for {filename}: {e}", exc_info=True)

    return metadata


async def process_document_for_rag(document_id: int, namespace: str, temp_file_path: str, validated_data: dict, kb_id: str):
    """
    Procesa un documento para RAG usando datos validados por el usuario.
    """
    db: Session = next(get_db())
    try:
        # Extraer texto del archivo temporal
        text_content = ""
        try:
            with open(temp_file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text_content += page.extract_text() + "\n"
            if not text_content.strip():
                raise ValueError("No se extrajo texto del PDF para fragmentar.")
        except Exception as e:
            logger.error(f"Error al extraer texto del PDF temporal {document_id}: {e}")
            update_document_processing_results(db, kb_id, document_id, results={"status": "failed", "processing_error": f"Error inesperado: {e}"})
            return

        # Fragmentar y crear embeddings
        chunks = [text_content[i:i + 1000] for i in range(0, len(text_content), 1000)]
        if not chunks:
            update_document_processing_results(db, kb_id, document_id, results={"status": "failed", "processing_error": "No se generaron fragmentos de texto."})
            return

        vectors_to_upsert = []
        for i, chunk in enumerate(chunks):
            try:
                embedding_response = await client_openai.embeddings.create(input=[chunk], model=settings.OPENAI_EMBEDDING_MODEL)
                embedding = embedding_response.data[0].embedding
                vectors_to_upsert.append({
                    "id": f"{document_id}-{i}",
                    "values": embedding,
                    "metadata": {
                        "document_id": document_id,
                        "title": validated_data.get('title'),
                        "publisher": validated_data.get('publisher'),
                        "publication_year": validated_data.get('publication_year'),
                        "summary": validated_data.get('summary'), # Usar resumen validado
                        "keywords": ", ".join(validated_data.get('keywords', [])), # Convertir lista a string
                        "text": chunk,
                    }
                })
            except Exception as e:
                logger.error(f"Error al crear embedding para el fragmento {i} del documento {document_id}: {e}")

        # Subir a Pinecone en lotes
        if vectors_to_upsert:
            batch_size = 100  # Número de vectores por lote
            total_upserted = 0
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                try:
                    pinecone_index.upsert(vectors=batch, namespace=namespace)
                    total_upserted += len(batch)
                    logger.info(f"Se subieron {len(batch)} vectores (lote {i//batch_size + 1}) para el documento {document_id} al namespace {namespace}.")
                except Exception as e:
                    logger.error(f"Error al subir lote a Pinecone para el documento {document_id}: {e}")
                    update_document_processing_results(db, kb_id, document_id, results={"status": "failed", "processing_error": f"Error en subida a Pinecone (lote): {e}"})
                    return # Salir si un lote falla

            logger.info(f"Se subieron un total de {total_upserted} vectores para el documento {document_id} al namespace {namespace}.")
            # Actualizar estado final y URL de la imagen de vista previa
            update_document_processing_results(db, kb_id, document_id, results={
                "status": "completed", 
                "vector_count": total_upserted,
                "cover_image_url": validated_data.get("cover_image_url")
            })
        else:
            update_document_processing_results(db, kb_id, document_id, results={"status": "failed", "processing_error": "No se generaron vectores para indexar."})

    except Exception as e:
        logger.error(f"Error no manejado durante el procesamiento RAG para el documento {document_id}: {e}", exc_info=True)
        update_document_processing_results(db, kb_id, document_id, results={"status": "failed", "processing_error": f"Error inesperado: {e}"})
