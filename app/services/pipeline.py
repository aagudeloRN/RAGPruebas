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
from ..db.crud import update_document_processing_results, get_document, get_document
from ..schemas.llm_responses import DocumentAnalysis

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
                {"role": "system", "content": "You are an expert document analysis assistant. Analyze the user's text and provide a summary, keywords, title, publisher, publication year, and language based on the requested format. Infer language from the text content. All fields are mandatory. The summary must be in Spanish. Keywords must be 7 to 10, in Spanish, homologated with the UNESCO thesaurus, and must not include the document title or author name."},                {"role": "user", "content": f"Analiza el siguiente texto:\n\n{text_for_summary}"}
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
    Diseñada para ejecutarse en segundo plano, por lo que gestiona su propia sesión de DB.
    """
    logger.info(f"BACKGROUND TASK: Starting processing for document ID {document_id}")
    db = SessionLocal()
    try:
        # Obtener el documento de la base de datos para acceder a sus metadatos
        document = get_document(db, document_id)
        if not document:
            raise ValueError(f"Document with ID {document_id} not found in DB.")

        # Leer el PDF desde la ruta temporal
        with open(pdf_path, "rb") as f:
            file_bytes = f.read()

        # 1. Abrir PDF y extraer texto
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        pdf_metadata = doc.metadata

        if len(full_text.strip()) < 100:
            raise ValueError("El PDF no contiene texto extraíble o es demasiado corto para ser procesado.")

        # Lógica de Cascada para Metadatos (usando los metadatos del documento de la DB)
        final_title = document.title or "Título no disponible"
        final_publisher = document.publisher
        final_publication_year = document.publication_year
        final_source_url = document.source_url

        # 2. Generar y subir imagen de la primera página a Cloudinary
        first_page = doc.load_page(0)
        pix = first_page.get_pixmap(dpi=150)  # Aumentamos un poco la resolución
        image_bytes = pix.tobytes("png")
        
        cloudinary_response = cloudinary.uploader.upload(
            image_bytes,
            public_id=f"doc_preview_{document_id}",
            overwrite=True,
            folder="rag_previews"  # Opcional: para organizar en Cloudinary
        )
        preview_image_url = cloudinary_response['secure_url']

        # 3. Chunking del texto
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_text(full_text)

        # 4. Generar Embeddings con OpenAI y subirlos a Pinecone
        response = client_openai.embeddings.create(
            input=chunks,
            model=settings.OPENAI_EMBEDDING_MODEL
        )
        embeddings = [item.embedding for item in response.data]

        vectors_to_upsert = []

        # Preparamos los metadatos para RAG/Pinecone
        pinecone_metadata = {
            "document_id": document_id,
            "title": final_title,
            "publisher": final_publisher,
            "publication_year": final_publication_year,
            "source_url": final_source_url
        }

        for i, chunk in enumerate(chunks):
            vector_id = f"doc_{document_id}_chunk_{i}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embeddings[i],
                "metadata": {**pinecone_metadata, "text": chunk} # Combinamos metadatos de cita con el texto del chunk
            })
        
        # Subir a Pinecone en lotes para evitar errores de tamaño de petición
        batch_size = settings.PINECONE_BATCH_SIZE
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            logger.info(f"BACKGROUND TASK: Upserting batch {i//batch_size + 1} to Pinecone for doc {document_id}")
            pinecone_index.upsert(vectors=batch, namespace=namespace)


        # 5. Generar Resumen y Keywords con OpenAI usando Tool Calling para una salida robusta
        # Usamos solo una parte del texto para no exceder el límite de tokens del LLM
        text_for_summary = " ".join(full_text.split()[:4000])
        
        response = client_openai.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert document analysis assistant. Analyze the user's text and provide a summary and keywords based on the requested format."},
                {"role": "user", "content": f"Analiza el siguiente texto:\n\n{text_for_summary}"}
            ],
            tools=[{"type": "function", "function": {"name": "document_analysis", "description": "Extracts a summary and keywords from a text.", "parameters": DocumentAnalysis.model_json_schema()}}],
            tool_choice={"type": "function", "function": {"name": "document_analysis"}},
            temperature=0.2,
        )

        tool_call = response.choices[0].message.tool_calls[0]
        if tool_call.function.name == "document_analysis":
            try:
                analysis_args = json.loads(tool_call.function.arguments)
                validated_analysis = DocumentAnalysis(**analysis_args)
                summary = validated_analysis.summary
                keywords = validated_analysis.keywords
            except (json.JSONDecodeError, ValidationError) as e:
                logger.error(f"Failed to parse or validate LLM response for doc {document_id}: {e}")
                raise ValueError("Error al procesar la respuesta del modelo de lenguaje.") from e
        else:
            raise ValueError("El modelo de lenguaje no devolvió la función esperada.")

        # Consolidar todos los resultados para la actualización de la base de datos
        final_results = {
            "title": final_title,
            "publisher": final_publisher,
            "publication_year": final_publication_year,
            "source_url": final_source_url,
            "status": "completed",
            "summary": summary,
            "keywords": keywords,
            "preview_image_url": preview_image_url
        }

    except Exception as e:
        logger.error(f"BACKGROUND TASK ERROR: Failed to process document ID {document_id}", exc_info=True)
        final_results = {"status": "failed", "processing_error": str(e)}
    finally:
        # Actualizar la base de datos con el estado final y todos los resultados
        if final_results:
            update_document_processing_results(db=db, document_id=document_id, results=final_results)
        db.close()
        # Eliminar el archivo temporal después de usarlo
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        logger.info(f"BACKGROUND TASK: Finished processing for document ID {document_id}. Final status: {final_results.get('status')}")
