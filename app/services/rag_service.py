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
from ..schemas.document import DocumentResponse, RefinedQuerySuggestion, QueryResponse # <-- Añadir QueryResponse
from ..core.config import settings
from ..db.session import get_db
from ..core.prompts import RAG_ANALYST_PROMPT_TEMPLATE

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
                    "content": """
                    Eres un experto en refinar preguntas de usuario para un sistema de búsqueda RAG. Tu objetivo es ayudar al usuario a formular la mejor pregunta posible para obtener resultados precisos y completos.
                    Dada la pregunta del usuario, genera 3 versiones mejoradas y una versión que sea una reformulación directa de la pregunta original pero optimizada para búsqueda.
                    Para cada sugerencia, proporciona una breve descripción de por qué es útil o qué tipo de información busca.
                    INSTRUCCIONES:
                    1.  **Reformulación Optimizada (Primera Sugerencia):** La primera sugerencia debe ser una reformulación directa de la pregunta original, optimizada para una mejor recuperación en un sistema RAG (ej. más concisa, con palabras clave claras, sin ambigüedades).
                    2.  **Desglosa la pregunta:** Si es una pregunta compleja, divide en sub-preguntas más específicas.
                    3.  **Añade palabras clave:** Sugiere una versión que incluya términos o entidades clave que probablemente se encuentren en los documentos relevantes.
                    4.  **Enfócate en datos:** Crea una versión específicamente diseñada para encontrar datos cuantitativos, como estadísticas, cifras, porcentajes o tendencias.
                    5.  **Devuelve solo un objeto JSON** con una única clave `suggestions` que contenga una lista de 4 objetos. Cada objeto debe tener dos claves: `query` (la pregunta sugerida) y `description` (una breve explicación de la sugerencia).
                    Ejemplo de formato de salida:
                    {
                        "suggestions": [
                            {"query": "Pregunta original reformulada y optimizada", "description": "Reformulación concisa y optimizada de tu pregunta original."},
                            {"query": "Pregunta desglosada", "description": "Desglosa la consulta en un subtema específico."},
                            {"query": "Pregunta con enfoque en palabras clave", "description": "Incorpora términos clave para una búsqueda más precisa."},
                            {"query": "Pregunta con enfoque en datos", "description": "Busca estadísticas y cifras clave."
                        ]
                    }
                    """
                },
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        suggestions_data = json.loads(refinement_response.choices[0].message.content)
        
        # Asegurarse de que la respuesta del LLM sea una lista de diccionarios con 'query' y 'description'
        parsed_suggestions = []
        for s in suggestions_data.get("suggestions", []):
            if isinstance(s, dict) and "query" in s and "description" in s:
                parsed_suggestions.append(RefinedQuerySuggestion(query=s["query"], description=s["description"]))
            elif isinstance(s, str): # Manejar el caso de que el LLM devuelva solo strings por error
                parsed_suggestions.append(RefinedQuerySuggestion(query=s, description="Sugerencia general."))

        # Si el LLM no devuelve la pregunta original, la añadimos como primera opción
        if not any(s.query == query for s in parsed_suggestions):
            parsed_suggestions.insert(0, RefinedQuerySuggestion(query=query, description="Tu pregunta original."))

        logger.info(f"Sugerencias generadas: {parsed_suggestions}")
        return parsed_suggestions
    except Exception as e:
        logger.error(f"Error al refinar la consulta: {e}", exc_info=True)
        # En caso de error, devolver solo la pregunta original con una descripción genérica
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
        # 1. Generar embedding para la consulta
        logger.info(f"Generando embedding para la consulta: '{query}'")
        query_embedding_response = client_openai.embeddings.create(
            input=[query],
            model=settings.OPENAI_EMBEDDING_MODEL
        )
        query_embedding = query_embedding_response.data[0].embedding

        # Construir filtro de Pinecone
        pinecone_filter = {}
        if start_year is not None:
            pinecone_filter["publication_year"] = {"$gte": start_year}
        if end_year is not None:
            if "publication_year" in pinecone_filter:
                pinecone_filter["publication_year"]["$lte"] = end_year
            else:
                pinecone_filter["publication_year"] = {"$lte": end_year}

        # 2. Buscar en Pinecone
        logger.info(f"Buscando en Pinecone con filtro: {pinecone_filter}")
        search_response = pinecone_index.query(
            vector=query_embedding,
            top_k=20, # Aumentado de 10 a 15 para un barrido inicial más amplio
            include_metadata=True,
            namespace=namespace,
            filter=pinecone_filter if pinecone_filter else None
        )

        # Recopilar IDs de documentos únicos en orden de relevancia
        unique_document_ids_ordered = []
        seen_document_ids = set()
        for match in search_response.matches:
            doc_id = match.metadata.get("document_id")
            if doc_id and doc_id not in seen_document_ids:
                unique_document_ids_ordered.append(doc_id)
                seen_document_ids.add(doc_id)

        # 3. Recuperar detalles completos de los documentos desde PostgreSQL
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

def perform_rag_query(query: str, namespace: str):
    """
    Orquesta el proceso de Retrieval-Augmented Generation (RAG).
    """
    db: Session = next(get_db())
    try:
        # 1. Consultar la caché primero
        cached_item = get_qa_cache_by_question(db, question=query)
        if cached_item:
            logger.info(f"Cache hit for query: '{query}'")
            return {"answer": cached_item.answer, "sources": json.loads(cached_item.context), "cache_hit": True}

        logger.info("--- RAG DEBUG START ---")
        logger.info(f"Query: '{query}'")

        # 2. Expansión de Consulta con LLM
        logger.info("Paso 1: Expansión de consulta con LLM...")
        try:
            expansion_response = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """
                    You are an expert at query expansion for vector search. Given a user's query, generate 3 additional, different versions of the query. The new queries should use synonyms, rephrase the question, or explore sub-topics. 
                    **Crucially, one of the queries must be specifically crafted to find quantitative data, statistics, or numerical figures related to the original question.**
                    Return a JSON object with a key 'queries' containing a list of 4 strings: the original query and the 3 new ones.
                    """},
                    {"role": "user", "content": query}
                ],
                response_format={"type": "json_object"}
            )
            queries_data = json.loads(expansion_response.choices[0].message.content)
            all_queries = queries_data.get("queries", [query])
            logger.info(f"Consultas expandidas: {all_queries}")
        except Exception as e:
            logger.error(f"Error en la expansión de consulta, usando solo la original. Error: {e}", exc_info=True)
            all_queries = [query]

        # 3. Crear embeddings para todas las preguntas
        logger.info("Paso 2: Creando embeddings para todas las preguntas...")
        try:
            query_embedding_response = client_openai.embeddings.create(
                input=all_queries,
                model="text-embedding-3-small"
            )
            query_embeddings = [item.embedding for item in query_embedding_response.data]
        except Exception as e:
            logger.error(f"Error al crear embeddings: {e}", exc_info=True)
            return {"answer": "Error interno al generar embeddings para la consulta.", "sources": []}

        # 4. Buscar en Pinecone con todos los embeddings y combinar resultados
        logger.info("Paso 3: Recuperación amplia de Pinecone con consultas expandidas...")
        seen_chunk_ids = set()
        all_matches = []
        try:
            for emb in query_embeddings:
                search_response = pinecone_index.query(
                    vector=emb,
                    top_k=15,  # Aumentado de 10 a 15 para un barrido inicial más amplio
                    include_metadata=True,
                    namespace=namespace
                )
                for match in search_response.matches:
                    if match.id not in seen_chunk_ids:
                        seen_chunk_ids.add(match.id)
                        all_matches.append(match)
            
            logger.info(f"Recuperados {len(all_matches)} chunks únicos de Pinecone para re-ranking.")
            if not all_matches:
                logger.info("Context is empty after Pinecone search. Returning 'not found' message.")
                logger.info("--- RAG DEBUG END ---")
                return {"answer": "La información no se encuentra en la base de conocimiento.", "sources": []}
        except Exception as e:
            logger.error(f"Error al buscar en Pinecone: {e}", exc_info=True)
            return {"answer": "Error interno al buscar en la base de vectores.", "sources": []}

        # 5. Usar Cohere para re-rankear los resultados combinados
        logger.info("Paso 4: Re-ranking de precisión con Cohere...")
        docs_to_rerank = [match.metadata.get('text', '') for match in all_matches]
       
        try:
            rerank_results = co_client.rerank(
                model='rerank-multilingual-v3.0',
                query=query,
                documents=docs_to_rerank,
                top_n=12 # Aumentado de 8 a 12 para un contexto final más rico
            )
            logger.info("Resultados del re-ranking de Cohere recibidos.")
        except Exception as e:
            logger.error(f"Error al re-rankear con Cohere: {e}", exc_info=True)
            return {"answer": "Error interno al re-rankear los resultados.", "sources": []}

        # 6. Construir el contexto y recopilar las fuentes a partir de los resultados re-rankeados
        logger.info("Paso 5: Construyendo contexto con los mejores chunks re-rankeados.")

        context = ""
        sources = {}

        for rank in rerank_results.results:
            original_match = all_matches[rank.index]
            
            if rank.relevance_score < 0.6:
                logger.info(f"Descartando chunk con bajo score de re-ranking: {rank.relevance_score:.4f}")
                continue

            # Extraer metadatos para la cita
            publisher = original_match.metadata.get('publisher', 'N/A')
            year = original_match.metadata.get('publication_year', 's.f.')
            
            # Anteponer la fuente al texto del chunk
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

        
        logger.info(f"Constructed context length: {len(context)} characters")
        logger.info(f"Found {len(sources)} unique sources.")

        if not context:
            logger.info("Context is empty. Returning 'not found' message.")
            logger.info("--- RAG DEBUG END ---")
            return {"answer": "La información no se encuentra en la base de conocimiento.", "sources": []}

        # 7. Construir el prompt para el LLM
        prompt_template = RAG_ANALYST_PROMPT_TEMPLATE.format(
            context=context,
            query=query
        )

        # 8. Generar la respuesta final con el LLM
        logger.info("Paso 6: Generando respuesta final con LLM...")
        try:
            final_response = client_openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt_template}],
                temperature=0.1
            )
            answer = final_response.choices[0].message.content
            logger.info(f"LLM Answer: {answer}")
            logger.info("--- RAG DEBUG END ---")
        except Exception as e:
            logger.error(f"Error al generar respuesta con LLM: {e}", exc_info=True)
            return {"answer": "Error interno al generar la respuesta final.", "sources": []}

        # 9. Guardar en caché
        logger.info("Paso 7: Guardando en caché...")
        try:
            qa_to_cache = QACacheCreate(
                question=query,
                answer=answer,
                context=json.dumps(list(sources.values()))
            )
            create_qa_cache(db, qa_in=qa_to_cache)
            logger.info("Respuesta guardada en caché.")
        except Exception as e:
            logger.error(f"Error al guardar en caché: {e}", exc_info=True)

        return QueryResponse(answer=answer, sources=list(sources.values()))
    finally:
        db.close()