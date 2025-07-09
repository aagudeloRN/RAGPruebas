import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
import openai
from pinecone import Pinecone
import cloudinary
import cloudinary.uploader
from app.db.session import SessionLocal
from app.db.crud import update_document_processing_results

# --- Inicialización de Clientes ---
# Se inicializan una vez cuando el módulo se carga para mayor eficiencia.

# OpenAI
# La librería de OpenAI lee la variable de entorno OPENAI_API_KEY automáticamente.
client_openai = openai.OpenAI()

# Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# NOTA: Asegúrate de tener un índice en Pinecone llamado 'vigilancia-dev-index'.
# Si tu índice tiene otro nombre, cámbialo aquí.
pinecone_index = pc.Index("vigilancia-dev-index")

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
    print(f"BACKGROUND TASK: Starting processing for document ID {document_id}")
    db = SessionLocal()
    final_results = {}
    try:
        # 1. Abrir PDF y extraer texto
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)
        pdf_metadata = doc.metadata

        if len(full_text.strip()) < 100:
            raise ValueError("El PDF parece estar basado en imágenes o tiene muy poco texto.")

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
            model="text-embedding-3-small"  # Modelo económico y eficiente
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
        batch_size = 100  # Un tamaño de lote seguro para Pinecone
        for i in range(0, len(vectors_to_upsert), batch_size):
            batch = vectors_to_upsert[i:i + batch_size]
            print(f"BACKGROUND TASK: Subiendo lote {i//batch_size + 1} a Pinecone...")
            pinecone_index.upsert(vectors=batch, namespace="default")


        # 5. Generar Resumen y Keywords con OpenAI
        # Usamos solo una parte del texto para no exceder el límite de tokens del LLM
        text_for_summary = " ".join(full_text.split()[:4000])
        
        summary_response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en analizar documentos. Genera un resumen conciso (máximo 150 palabras) y después, en una nueva línea, escribe 'Palabras clave:' seguido de 10 palabras clave relevantes separadas por comas, para estas palabras toma como base el listado de tesauros homologados de la UNESCO, no incluyas nombres própios ni de organizaciones, solo información referente al contenido del texto base."},
                {"role": "user", "content": f"Analiza el siguiente texto:\n\n{text_for_summary}"}
            ],
            temperature=0.2,
        )
        content = summary_response.choices[0].message.content
        parts = content.split("Palabras clave:")
        summary = parts[0].replace("Resumen:", "").strip()
        keywords_str = parts[1].strip() if len(parts) > 1 else ""
        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

        # Consolidar todos los resultados para la actualización de la base de datos
        final_results = {
            "title": final_title, # Guardamos el título verificado
            "filename": user_metadata.get("filename"), # Guardamos el nombre original del archivo
            "publisher": final_publisher,
            "publication_year": final_publication_year,
            "source_url": final_source_url,
            "status": "completed",
            "summary": summary,
            "keywords": keywords,
            "preview_image_url": preview_image_url,
            "language": user_metadata.get("language") # Mantenemos el idioma que el usuario seleccionó
        }

    except Exception as e:
        print(f"BACKGROUND TASK ERROR: Error processing document ID {document_id}: {e}")
        final_results = {"status": "failed"}
    finally:
        # Actualizar la base de datos con el estado final y todos los resultados
        if final_results:
            update_document_processing_results(db=db, document_id=document_id, results=final_results)
        db.close()
        print(f"BACKGROUND TASK: Finished processing for document ID {document_id}. Final status: {final_results.get('status')}")


