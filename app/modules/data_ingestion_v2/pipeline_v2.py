
# app/modules/data_ingestion_v2/pipeline_v2.py
import logging
from typing import List, Dict, Any
from openai import AsyncOpenAI
from pinecone import Pinecone
from app.core.config import settings
from app.modules.document_processing.text_extractor import process_text_document
from app.modules.document_processing.table_extractor import extract_and_format_tables

logger = logging.getLogger(__name__)

# ... (código de inicialización de clientes y funciones de embedding/upsert) ...
client_openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

async def generate_embeddings(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []
    logger.info(f"Generando embeddings para {len(chunks)} chunks...")
    response = await client_openai.embeddings.create(
        input=chunks,
        model=settings.OPENAI_EMBEDDING_MODEL
    )
    return [item.embedding for item in response.data]

async def upsert_chunks_to_pinecone(namespace: str, chunks: List[str], embeddings: List[List[float]], metadata: Dict[str, Any]):
    if not embeddings:
        logger.warning("No hay embeddings para guardar en Pinecone.")
        return

    vectors_to_upsert = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{metadata['document_id']}_chunk_{i}"
        vector_metadata = {
            "document_id": metadata["document_id"],
            "title": metadata["title"],
            "publisher": metadata["publisher"],
            "publication_year": metadata["publication_year"],
            "text": chunk
        }
        vectors_to_upsert.append((vector_id, embedding, vector_metadata))
    
    logger.info(f"Guardando {len(vectors_to_upsert)} vectores en Pinecone en el namespace '{namespace}'...")
    pinecone_index.upsert(vectors=vectors_to_upsert, namespace=namespace)

class IngestionPipelineV2:
    def __init__(self, doc_path: str, metadata: Dict[str, Any], namespace: str):
        self.doc_path = doc_path
        self.metadata = metadata
        self.namespace = namespace
        self.final_chunks = []

    async def run(self):
        logger.info(f"Iniciando pipeline de ingesta V2 para: {self.doc_path}")
        
        # 1. Procesar texto y tablas usando los módulos independientes
        text_chunks = process_text_document(self.doc_path)
        table_chunks = extract_and_format_tables(self.doc_path)
        self.final_chunks = text_chunks + table_chunks
        logger.info(f"Pipeline V2: Total de chunks generados: {len(self.final_chunks)}")

        # 2. Generar embeddings y guardar
        embeddings = await generate_embeddings(self.final_chunks)
        await upsert_chunks_to_pinecone(self.namespace, self.final_chunks, embeddings, self.metadata)
        
        logger.info("Pipeline V2 completado exitosamente.")
        return {"status": "success", "message": f"Se procesaron y guardaron {len(self.final_chunks)} chunks."}
