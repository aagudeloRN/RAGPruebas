# app/modules/rag_chat/rag_service.py
import logging
import json
from typing import List, Optional

import cohere
from openai import AsyncOpenAI
from pinecone import Pinecone
from sqlalchemy.orm import Session
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings, active_kb
from app.core.prompts import RAG_ANALYST_PROMPT_TEMPLATE, QUERY_REFINEMENT_PROMPT
from app.db.crud_qa_cache import get_qa_cache_by_question, create_qa_cache
from app.schemas.document import QueryResponse, Source
from app.schemas.qa_cache import QACacheCreate

logger = logging.getLogger(__name__)

# --- Inicialización de Clientes ---
client_openai = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
co_client = cohere.Client(settings.COHERE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

@retry(wait=wait_exponential(multiplier=1, min=2, max=6), stop=stop_after_attempt(3))
async def perform_rag_query(query: str, db: Session, pgvector_db: Session) -> QueryResponse:
    """
    Orquesta el proceso RAG para la base de conocimiento activa.
    """
    if not active_kb.id or not active_kb.pinecone_namespace:
        raise ValueError("La Base de Conocimiento activa no está configurada.")

    # 1. Búsqueda en Caché
    cached_item = await get_qa_cache_by_question(pgvector_db, question=query, kb_id=active_kb.id)
    if cached_item:
        logger.info(f"Cache HIT para KB '{active_kb.id}' (ID: {cached_item.id}).")
        sources_from_cache = [Source.model_validate(s) for s in json.loads(cached_item.context)]
        return QueryResponse(answer=cached_item.answer, sources=sources_from_cache, cache_hit=True)

    logger.info(f"Cache MISS para KB '{active_kb.id}'. Iniciando proceso RAG.")

    # 2. Generar Embedding para la consulta
    try:
        query_embedding_response = await client_openai.embeddings.create(
            input=[query],
            model=settings.OPENAI_EMBEDDING_MODEL
        )
        query_embedding = query_embedding_response.data[0].embedding
    except Exception as e:
        logger.error(f"Error al generar embedding: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al procesar la consulta.")

    # 3. Búsqueda en Pinecone
    try:
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=15,
            include_metadata=True,
            namespace=active_kb.pinecone_namespace
        )
        if not search_response.matches:
            return QueryResponse(answer="No se encontró información relevante en la base de conocimiento.", sources=[])
    except Exception as e:
        logger.error(f"Error en la búsqueda de Pinecone: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al buscar en la base de datos vectorial.")

    # 4. Re-ranking con Cohere
    docs_to_rerank = [match.metadata.get('text', '') for match in search_response.matches]
    try:
        rerank_results = co_client.rerank(
            model='rerank-multilingual-v3.0',
            query=query,
            documents=docs_to_rerank,
            top_n=10
        )
    except Exception as e:
        logger.error(f"Error en el re-ranking de Cohere: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al procesar los resultados.")

    # 5. Construir Contexto y Fuentes
    context = ""
    sources = {}
    context_chunks_for_cache = []
    for rank in rerank_results.results:
        if rank.relevance_score < 0.60: # Umbral de relevancia
            continue
        
        original_match = search_response.matches[rank.index]
        chunk_text = original_match.metadata.get('text', '')
        context_chunks_for_cache.append(chunk_text)

        publisher = original_match.metadata.get('publisher', 'N/A')
        year = original_match.metadata.get('publication_year', 's.f.')
        context += f"Fuente: ({publisher}, {year})\n{chunk_text}\n\n"
        
        doc_id = original_match.metadata.get('document_id')
        if doc_id and doc_id not in sources:
            sources[doc_id] = {
                "id": doc_id,
                "title": original_match.metadata.get('title', 'N/A'),
                "publisher": publisher,
                "publication_year": str(year),
                "source_url": original_match.metadata.get('source_url')
            }

    if not context:
        return QueryResponse(answer="No se encontró información suficientemente relevante.", sources=[])

    # 6. Generar Respuesta Final con LLM
    prompt = RAG_ANALYST_PROMPT_TEMPLATE.format(context=context, query=query)
    try:
        final_response = await client_openai.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.1
        )
        answer = final_response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error en la generación de respuesta final: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error interno al generar la respuesta.")

    # 7. Guardar en Caché
    source_objects = [Source.model_validate(s) for s in sources.values()]
    try:
        qa_to_cache = QACacheCreate(
            question=query,
            answer=answer,
            context=json.dumps([s.model_dump() for s in source_objects]),
            context_chunks=context_chunks_for_cache
        )
        await create_qa_cache(pgvector_db, qa_in=qa_to_cache, kb_id=active_kb.id)
    except Exception as e:
        logger.error(f"Error al guardar en caché: {e}", exc_info=True)

    return QueryResponse(answer=answer, sources=source_objects, cache_hit=False)
