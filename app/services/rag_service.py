import openai
from pinecone import Pinecone
import cohere
import json
from typing import List, Optional
from sqlalchemy.orm import Session
import logging

# Configurar logging
logger = logging.getLogger(__name__)

from ..core.config import settings
from ..db.crud import get_document
from ..db.crud_qa_cache import get_qa_cache_by_question, create_qa_cache
from ..schemas.qa_cache import QACacheCreate
from ..schemas.document import DocumentResponse, RefinedQuerySuggestion, QueryResponse, Source
from ..core.config import settings
from ..db.session import get_db
from ..core.prompts import RAG_ANALYST_PROMPT_TEMPLATE, QUERY_REFINEMENT_PROMPT, QUERY_DECOMPOSITION_PROMPT # <-- AÑADIR QUERY_DECOMPOSITION_PROMPT

# --- Inicialización de Clientes ---
client_openai = openai.OpenAI()
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
co_client = cohere.Client(settings.COHERE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

def get_refined_query_suggestions(query: str) -> List[RefinedQuerySuggestion]:
    """
    Usa un LLM para refinar la pregunta de un usuario y devolver una lista de sugerencias con descripciones.
    """
    logger.info(f"Refinando la consulta: '{query}'")
    try:
        refinement_response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": QUERY_REFINEMENT_PROMPT
                },
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        suggestions_data = json.loads(refinement_response.choices[0].message.content)
        
        parsed_suggestions = []
        for s in suggestions_data.get("suggestions", []):
            if isinstance(s, dict) and "query" in s and "description" in s:
                parsed_suggestions.append(RefinedQuerySuggestion(query=s["query"], description=s["description"]))
            elif isinstance(s, str):
                parsed_suggestions.append(RefinedQuerySuggestion(query=s, description="Sugerencia general."))

        if not any(s.query == query for s in parsed_suggestions):
            parsed_suggestions.insert(0, RefinedQuerySuggestion(query=query, description="Tu pregunta original."))

        logger.info(f"Sugerencias generadas: {parsed_suggestions}")
        return parsed_suggestions
    except Exception as e:
        logger.error(f"Error al refinar la consulta: {e}", exc_info=True)
        return [RefinedQuerySuggestion(query=query, description="No se pudieron generar sugerencias. Intenta con tu pregunta original.")]

def semantic_search_documents(
    query: str,
    namespace: str,
    db: Session,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> List[DocumentResponse]:
    """
    Realiza una búsqueda semántica de documentos en Pinecone y recupera los detalles completos de la DB.
    """
    try:
        logger.info(f"Generando embedding para la consulta: '{query}'")
        query_embedding_response = client_openai.embeddings.create(
            input=[query],
            model=settings.OPENAI_EMBEDDING_MODEL
        )
        query_embedding = query_embedding_response.data[0].embedding

        pinecone_filter = {}
        if start_year is not None:
            pinecone_filter["publication_year"] = {"$gte": start_year}
        if end_year is not None:
            if "publication_year" in pinecone_filter:
                pinecone_filter["publication_year"]["$lte"] = end_year
            else:
                pinecone_filter["publication_year"] = {"$lte": end_year}

        logger.info(f"Buscando en Pinecone con filtro: {pinecone_filter}")
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=20,
            include_metadata=True,
            namespace=namespace,
            filter=pinecone_filter if pinecone_filter else None
        )

        unique_document_ids_ordered = []
        seen_document_ids = set()
        for match in search_response.matches:
            doc_id = match.metadata.get("document_id")
            if doc_id and doc_id not in seen_document_ids:
                unique_document_ids_ordered.append(doc_id)
                seen_document_ids.add(doc_id)

        logger.info(f"Recuperando {len(unique_document_ids_ordered)} documentos únicos de PostgreSQL.")
        documents = []
        for doc_id in unique_document_ids_ordered:
            doc = get_document(db, doc_id)
            if doc:
                documents.append(DocumentResponse.model_validate(doc))

        return documents

    except Exception as e:
        logger.error(f"Error en la búsqueda semántica: {e}", exc_info=True)
        return []

def perform_rag_query(query: str, namespace: str) -> QueryResponse:
    """
    Orquesta el proceso de Retrieval-Augmented Generation (RAG).
    Ahora devuelve siempre un objeto QueryResponse.
    """
    db: Session = next(get_db())
    try:
        cached_item = get_qa_cache_by_question(db, question=query)
        if cached_item:
            logger.info(f"Cache hit for query: '{query}'")
            sources_from_cache = [Source.model_validate(s) for s in json.loads(cached_item.context)]
            return QueryResponse(answer=cached_item.answer, sources=sources_from_cache, cache_hit=True)

        logger.info("--- RAG DEBUG START ---")
        logger.info(f"Query: '{query}'")

        # --- Query Decomposition (Nuevo Paso) ---
        decomposed_queries = [query] # Por defecto, la pregunta original
        try:
            decomposition_response = client_openai.chat.completions.create(
                model="gpt-4o", # Usar un modelo capaz de entender la complejidad
                messages=[
                    {"role": "system", "content": QUERY_DECOMPOSITION_PROMPT},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            decomposition_data = json.loads(decomposition_response.choices[0].message.content)
            if decomposition_data and "sub_queries" in decomposition_data and isinstance(decomposition_data["sub_queries"], list):
                decomposed_queries = decomposition_data["sub_queries"]
            logger.info(f"Pregunta descompuesta en: {decomposed_queries}")
        except Exception as e:
            logger.error(f"Error en la descomposición de consulta, usando solo la original. Error: {e}", exc_info=True)
        
        # --- Query Expansion (Existente, ahora sobre las preguntas descompuestas) ---
        all_queries_for_embedding = []
        for dq in decomposed_queries:
            try:
                expansion_response = client_openai.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "\n                        You are an expert at query expansion for vector search. Given a user's query, generate 3 additional, different versions of the query. The new queries should use synonyms, rephrase the question, or explore sub-topics. \n                        **Crucially, one of the queries must be specifically crafted to find quantitative data, statistics, or numerical figures related to the original question.**\n                        Return a JSON object with a key 'queries' containing a list of 4 strings: the original query and the 3 new ones.\n                        "},
                        {"role": "user", "content": dq} # Usar la pregunta descompuesta
                    ],
                    response_format={"type": "json_object"}
                )
                queries_data = json.loads(expansion_response.choices[0].message.content)
                all_queries_for_embedding.extend(queries_data.get("queries", [dq]))
            except Exception as e:
                logger.error(f"Error en la expansión de consulta para '{dq}', usando solo la original. Error: {e}", exc_info=True)
                all_queries_for_embedding.append(dq)

        # Asegurarse de que la pregunta original esté siempre incluida
        if query not in all_queries_for_embedding:
            all_queries_for_embedding.insert(0, query)

        logger.info(f"Consultas finales para embedding: {all_queries_for_embedding}")

        try:
            query_embedding_response = client_openai.embeddings.create(
                input=all_queries_for_embedding,
                model="text-embedding-3-small"
            )
            query_embeddings = [item.embedding for item in query_embedding_response.data]
        except Exception as e:
            logger.error(f"Error al crear embeddings: {e}", exc_info=True)
            return QueryResponse(answer="Error interno al generar embeddings para la consulta.", sources=[])

        all_matches = []
        seen_chunk_ids = set()
        try:
            for emb in query_embeddings:
                search_response = pinecone_index.query(
                    vector=emb,
                    top_k=15,
                    include_metadata=True,
                    namespace=namespace
                )
                for match in search_response.matches:
                    if match.id not in seen_chunk_ids:
                        seen_chunk_ids.add(match.id)
                        all_matches.append(match)
            
            if not all_matches:
                return QueryResponse(answer="La información no se encuentra en la base de conocimiento.", sources=[])
        except Exception as e:
            logger.error(f"Error al buscar en Pinecone: {e}", exc_info=True)
            return QueryResponse(answer="Error interno al buscar en la base de vectores.", sources=[])

        docs_to_rerank = [match.metadata.get('text', '') for match in all_matches]
        try:
            rerank_results = co_client.rerank(
                model='rerank-multilingual-v3.0',
                query=query,
                documents=docs_to_rerank,
                top_n=12
            )
        except Exception as e:
            logger.error(f"Error al re-rankear con Cohere: {e}", exc_info=True)
            return QueryResponse(answer="Error interno al re-rankear los resultados.", sources=[])

        context = ""
        sources = {}
        for rank in rerank_results.results:
            original_match = all_matches[rank.index]
            if rank.relevance_score < 0.6:
                continue
            publisher = original_match.metadata.get('publisher', 'N/A')
            year = original_match.metadata.get('publication_year', 's.f.')
            context += f"Fuente: ({publisher}, {year})\n"
            context += original_match.metadata.get('text', '') + "\n\n"
            doc_id = original_match.metadata.get('document_id')
            if doc_id not in sources:
                sources[doc_id] = {
                    "id": doc_id,
                    "title": original_match.metadata.get('title', 'Título no disponible'),
                    "publisher": publisher,
                    "publication_year": str(year),
                    "source_url": original_match.metadata.get('source_url')
                }

        if not context:
            return QueryResponse(answer="La información no se encuentra en la base de conocimiento.", sources=[])

        prompt_template = RAG_ANALYST_PROMPT_TEMPLATE.format(context=context, query=query)

        try:
            final_response = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt_template}],
                temperature=0.1
            )
            answer = final_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error al generar respuesta con LLM: {e}", exc_info=True)
            return QueryResponse(answer="Error interno al generar la respuesta final.", sources=[])

        source_objects = [Source.model_validate(s) for s in sources.values()]
        try:
            qa_to_cache = QACacheCreate(
                question=query,
                answer=answer,
                context=json.dumps([s.model_dump() for s in source_objects])
            )
            create_qa_cache(db, qa_in=qa_to_cache)
        except Exception as e:
            logger.error(f"Error al guardar en caché: {e}", exc_info=True)

        return QueryResponse(answer=answer, sources=source_objects)
    finally:
        db.close()