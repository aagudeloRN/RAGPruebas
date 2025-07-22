import logging
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from pinecone import Pinecone
from pydantic import ValidationError
import cloudinary
import cloudinary.uploader
import tempfile
import os

from ..core.config import settings
from ..db.session import SessionLocal
from ..db.crud import update_document_processing_results, get_document, delete_document
from ..schemas.llm_responses import DocumentAnalysis
from ..core.prompts import METADATA_EXTRACTION_PROMPT # <-- NUEVA IMPORTACIÓN

# --- Inicialización de Clientes ---
# Se inicializan una vez cuando el módulo se carga para mayor eficiencia.

# Configurar logging
logger = logging.getLogger(__name__)

# OpenAI
# La librería de OpenAI lee la variable de entorno OPENAI_API_KEY automáticamente.
client_openai = openai.OpenAI()

# Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

# Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
)

def extract_metadata_pipeline(file_bytes: bytes) -> dict:
    """
    Función en segundo plano para extraer metadatos de un PDF usando LLM y actualizar la DB.
    """
    logger.info(f"BACKGROUND TASK: Starting metadata extraction.")
    try:
        # Guardar el PDF en un archivo temporal
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file_bytes)
            pdf_path = tmp_file.name

        # 1. Abrir PDF y extraer texto
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        pdf_metadata = doc.metadata

        if len(full_text.strip()) < 100:
            raise ValueError("El PDF no contiene texto extraíble o es demasiado corto para ser procesado.")

        # 2. Generar Resumen y Keywords con OpenAI usando Tool Calling para una salida robusta
        # Usamos solo una parte del texto para no exceder el límite de tokens del LLM
        text_for_summary = " ".join(full_text.split()[:4000]) # Limitar a los primeros 4000 tokens

        response = client_openai.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": METADATA_EXTRACTION_PROMPT}, # <-- USANDO LA VARIABLE DEL PROMPT
                {"role": "user", "content": f"Analiza el siguiente texto:\n\n{text_for_summary}"}
            ],
            tools=[{"type": "function", "function": {"name": "document_analysis", "description": "Extracts a summary, keywords, title, publisher, publication year, and language from a text.", "parameters": DocumentAnalysis.model_json_schema()}}],
            tool_choice={"type": "function", "function": {"name": "document_analysis"}},
            temperature=0.2,
        )

        tool_call = response.choices[0].message.tool_calls[0]
        if tool_call.function.name == "document_analysis":
            try:
                analysis_args = json.loads(tool_call.function.arguments)
                validated_analysis = DocumentAnalysis(**analysis_args)
                
                # Consolidar metadatos extraídos y del PDF
                extracted_metadata = {
                    "title": validated_analysis.title if validated_analysis.title is not None else pdf_metadata.get("title"),
                    "publisher": validated_analysis.publisher if validated_analysis.publisher is not None else pdf_metadata.get("author"),
                    "publication_year": validated_analysis.publication_year if validated_analysis.publication_year is not None else (int(pdf_metadata.get("creationDate")[2:6]) if pdf_metadata.get("creationDate", "").startswith("D:") else None),
                    "language": validated_analysis.language if validated_analysis.language is not None else None,
                    "summary": validated_analysis.summary if validated_analysis.summary is not None else None,
                    "keywords": validated_analysis.keywords if validated_analysis.keywords is not None else [],
                    "filename": doc.name # Asegurarse de que el nombre del archivo esté incluido
                }
                
                return {"metadata": extracted_metadata, "pdf_path": pdf_path}

            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse or validate LLM response: {e}")
                raise ValueError("Error al procesar la respuesta del modelo de lenguaje.") from e
        else:
            raise ValueError("El modelo de lenguaje no devolvió la función esperada.")

    except Exception as e:
        logger.error(f"BACKGROUND TASK ERROR: Failed to extract metadata.", exc_info=True)
        raise e # Re-lanzar la excepción para que el llamador la maneje
    finally:
        # El archivo temporal se eliminará después de que el procesamiento completo termine
        # o si hay un error irrecuperable en la extracción de metadatos.
        pass


def process_pdf_pipeline(document_id: int, pdf_path: str, namespace: str):
    """
    Función que orquesta el pipeline de procesamiento.
    Diseñada para ejecutarse en segundo plano. Si falla, el registro del documento se elimina
    para mantener la integridad de la base de datos (transacción de compensación).
    """
    logger.info(f"BACKGROUND TASK: Starting processing for document ID {document_id}")
    db = SessionLocal()
    try:
        document = get_document(db, document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found in DB.")

        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)

        if len(full_text.strip()) < 100:
            raise ValueError("El PDF no contiene texto extraíble o es demasiado corto.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(full_text)

        # Generar Embeddings por lotes
        all_embeddings = []
        batch_size = 250  # Tamaño de lote seguro
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1} for doc {document_id}")
            response = client_openai.embeddings.create(
                input=batch_chunks, model=settings.OPENAI_EMBEDDING_MODEL
            )
            all_embeddings.extend([item.embedding for item in response.data])

        # Preparar y subir vectores a Pinecone
        vectors_to_upsert = []
        pinecone_metadata = {
            "document_id": document_id,
            "title": document.title,
            "publisher": document.publisher,
            "publication_year": document.publication_year,
            "source_url": document.source_url
        }
        for i in range(0, len(chunks), 1):
            vectors_to_upsert.append({
                "id": f"doc_{document_id}_chunk_{i}",
                "values": all_embeddings[i],
                "metadata": {**pinecone_metadata, "text": chunks[i]}
            })
        
        pinecone_batch_size = settings.PINECONE_BATCH_SIZE
        for i in range(0, len(vectors_to_upsert), pinecone_batch_size):
            batch = vectors_to_upsert[i:i + pinecone_batch_size]
            logger.info(f"Upserting batch {i//pinecone_batch_size + 1} to Pinecone for doc {document_id}")
            pinecone_index.upsert(vectors=batch, namespace=namespace)

        # Generar y subir imagen de vista previa
        first_page = doc.load_page(0)
        pix = first_page.get_pixmap(dpi=150)
        image_bytes = pix.tobytes("png")
        cloudinary_response = cloudinary.uploader.upload(
            image_bytes, public_id=f"doc_preview_{document_id}", overwrite=True, folder="rag_previews"
        )
        preview_image_url = cloudinary_response['secure_url']

        # Actualizar el estado final a "completed"
        final_results = {"status": "completed", "preview_image_url": preview_image_url}
        update_document_processing_results(db=db, document_id=document_id, results=final_results)
        logger.info(f"BACKGROUND TASK: Successfully completed processing for document ID {document_id}")

    except Exception as e:
        logger.error(f"BACKGROUND TASK ERROR: Failed to process document ID {document_id}. Deleting record.", exc_info=True)
        # Transacción de compensación: si algo falla, eliminamos el registro de la DB.
        delete_document(db=db, document_id=document_id)
        # También podríamos querer eliminar los datos de Pinecone si se subieron parcialmente, 
        # pero por ahora, eliminar de la DB es lo más crítico.
    finally:
        db.close()
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        logger.info(f"BACKGROUND TASK: Finished processing task for document ID {document_id}.")