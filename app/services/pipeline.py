import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.config import settings
import openai
from pinecone import Pinecone
import cloudinary
import cloudinary.uploader

# --- Inicialización de Clientes ---
# Se inicializan una vez cuando el módulo se carga para mayor eficiencia.

# OpenAI
# La librería de OpenAI lee la variable de entorno OPENAI_API_KEY automáticamente.
client_openai = openai.OpenAI()

# Pinecone
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
# NOTA: Asegúrate de tener un índice en Pinecone llamado 'rag-documents'.
# Si tu índice tiene otro nombre, cámbialo aquí.
pinecone_index = pc.Index("vigilancia-dev-index")

# Cloudinary
cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
)

def process_pdf_pipeline(file_bytes: bytes, document_id: int):
    """
    Función principal que orquesta todo el pipeline de procesamiento de un PDF.
    """
    try:
        # 1. Abrir PDF y extraer texto
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        full_text = "".join(page.get_text() for page in doc)

        if len(full_text.strip()) < 100:
            raise ValueError("El PDF parece estar basado en imágenes o tiene muy poco texto.")

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
        for i, chunk in enumerate(chunks):
            vector_id = f"doc_{document_id}_chunk_{i}"
            vectors_to_upsert.append({
                "id": vector_id,
                "values": embeddings[i],
                "metadata": {"document_id": document_id, "text": chunk}
            })
        
        pinecone_index.upsert(vectors=vectors_to_upsert, namespace="default")

        # 5. Generar Resumen y Keywords con OpenAI
        # Usamos solo una parte del texto para no exceder el límite de tokens del LLM
        text_for_summary = " ".join(full_text.split()[:4000])
        
        summary_response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Eres un asistente experto en analizar documentos. Genera un resumen conciso (máximo 150 palabras) y después, en una nueva línea, escribe 'Palabras clave:' seguido de 5 a 7 palabras clave relevantes separadas por comas, para estas palabras toma como base el listado de tesauros homologados de la UNESCO."},
                {"role": "user", "content": f"Analiza el siguiente texto:\n\n{text_for_summary}"}
            ],
            temperature=0.2,
        )
        content = summary_response.choices[0].message.content
        parts = content.split("Palabras clave:")
        summary = parts[0].replace("Resumen:", "").strip()
        keywords_str = parts[1].strip() if len(parts) > 1 else ""
        keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]

        return {"status": "completed", "summary": summary, "keywords": keywords, "preview_image_url": preview_image_url}

    except Exception as e:
        print(f"Error procesando el documento ID {document_id}: {e}")
        # Devolvemos un diccionario con el status, que es lo que espera el crud.update
        return {"status": "failed"}

