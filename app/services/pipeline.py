import logging
import json
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openai
from pinecone import Pinecone
from pydantic import ValidationError
import cloudinary
import cloudinary.uploader

from ..core.config import settings
from ..db.session import SessionLocal
from ..db.crud import update_document_processing_results
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

def process_pdf_pipeline(file_bytes: bytes, document_id: int, user_metadata: dict):
    """
    Función que orquesta el pipeline de procesamiento.
    Diseñada para ejecutarse en segundo plano, por lo que gestiona su propia sesión de DB.
    """
    logger.info(f"BACKGROUND TASK: Starting processing for document ID {document_id}")
    db = SessionLocal()
    final_results = {}
    try:
        # 1. Abrir PDF y extraer texto
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        pdf_metadata = doc.metadata

        if len(full_text.strip()) < 100:
            raise ValueError("El PDF no contiene texto extraíble o es demasiado corto para ser procesado.")

        # Lógica de Cascada para Metadatos
        # Prioridad 1: Datos del usuario. Prioridad 2: Datos del PDF.
        
        # Parsear la fecha del PDF de forma segura
        pdf_date_str = pdf_metadata.get("creationDate", "") # D:YYYYMMDD...
        pdf_year = None
        if pdf_date_str.startswith("D:"):
            try:
                pdf_year = int(pdf_date_str[2:6])
            except (ValueError, IndexError):
                pdf_year = None

        final_title = user_metadata.get("title") or pdf_metadata.get("title") or user_metadata.get("filename") or "Título no disponible"
        final_publisher = user_metadata.get("publisher") or pdf_metadata.get("author")
        final_publication_year = user_metadata.get("publication_year") or pdf_year
        final_source_url = user_metadata.get("source_url")

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
            pinecone_index.upsert(vectors=batch, namespace="default")


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
        logger.info(f"BACKGROUND TASK: Finished processing for document ID {document_id}. Final status: {final_results.get('status')}")
