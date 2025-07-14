import openai
from pinecone import Pinecone
import cohere
import json
from ..core.config import settings

# --- Inicialización de Clientes ---
client_openai = openai.OpenAI()
pc = Pinecone(api_key=settings.PINECONE_API_KEY)
co_client = cohere.Client(settings.COHERE_API_KEY)
pinecone_index = pc.Index(settings.PINECONE_INDEX_NAME)

def perform_rag_query(query: str):
    """
    Orquesta el proceso de Retrieval-Augmented Generation (RAG).
    """
    print("\n--- RAG DEBUG START ---")
    print(f"Query: '{query}'")

    # 1. Expansión de Consulta con LLM
    print("Paso 1: Expansión de consulta con LLM...")
    try:
        expansion_response = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at query expansion for vector search. Given a user's query, generate 3 additional, different versions of the query. The new queries should use synonyms, rephrase the question, or explore sub-topics. Return a JSON object with a key 'queries' containing a list of 4 strings: the original query and the 3 new ones."},
                {"role": "user", "content": query}
            ],
            response_format={"type": "json_object"}
        )
        queries_data = json.loads(expansion_response.choices[0].message.content)
        all_queries = queries_data.get("queries", [query])
        print(f"Consultas expandidas: {all_queries}")
    except Exception as e:
        print(f"Error en la expansión de consulta, usando solo la original. Error: {e}")
        all_queries = [query]

    # 2. Crear embeddings para todas las preguntas
    query_embedding_response = client_openai.embeddings.create(
        input=all_queries,
        model="text-embedding-3-small"
    )
    query_embeddings = [item.embedding for item in query_embedding_response.data]

    # 3. Buscar en Pinecone con todos los embeddings y combinar resultados
    print("Paso 2: Recuperación amplia de Pinecone con consultas expandidas...")
    seen_chunk_ids = set()
    all_matches = []
    for emb in query_embeddings:
        search_response = pinecone_index.query(
            vector=emb,
            top_k=10,  # Pedimos 10 por cada consulta expandida #######################
            include_metadata=True,
            namespace="default"
        )
        for match in search_response.matches:
            if match.id not in seen_chunk_ids:
                seen_chunk_ids.add(match.id)
                all_matches.append(match)
    
    print(f"Recuperados {len(all_matches)} chunks únicos de Pinecone para re-ranking.")
    if not all_matches:
        print("Context is empty after Pinecone search. Returning 'not found' message.")
        print("--- RAG DEBUG END ---\n")
        return {"answer": "La información no se encuentra en la base de conocimiento.", "sources": []}

    # 4. Usar Cohere para re-rankear los resultados combinados
    print("Paso 3: Re-ranking de precisión con Cohere...")
    docs_to_rerank = [match.metadata.get('text', '') for match in all_matches]
   
    rerank_results = co_client.rerank(
        #model='rerank-v3.5', # o 'rerank-multilingual-v3.0' si tienes documentos en varios idiomas
        model='rerank-multilingual-v3.0', # o 'rerank-v3.5' si tenemos documentos solo en inglés
        query=query,
        documents=docs_to_rerank,
        top_n=8 # Nos quedamos con los 8 mejores después del re-ranking
    )
    print("Resultados del re-ranking de Cohere recibidos.")

    # 5. Construir el contexto y recopilar las fuentes a partir de los resultados re-rankeados
    print("Paso 4: Construyendo contexto con los mejores chunks re-rankeados.")

    context = ""
    sources = {}
    source_details_for_prompt = ""

    # Iteramos sobre los resultados de Cohere, que están ordenados por relevancia
    for rank in rerank_results.results:
        # Usamos el índice del resultado re-rankeado para encontrar el documento original de la lista combinada
        original_match = all_matches[rank.index]
        
        # Opcional: Podemos añadir un umbral al score del re-ranker si queremos ser aún más estrictos
        if rank.relevance_score < 0.6:
            print(f"Descartando chunk con bajo score de re-ranking: {rank.relevance_score:.4f}")
            continue

        context += original_match.metadata.get('text', '') + "\n\n"
        doc_id = original_match.metadata.get('document_id')

        if doc_id not in sources:
            publisher = original_match.metadata.get('publisher', 'N/A')
            year = original_match.metadata.get('publication_year')
            title = original_match.metadata.get('title', 'Título no disponible')

            sources[doc_id] = {
                "id": doc_id,
                "title": title,
                "publisher": publisher,
                "publication_year": str(year) if year else 's.f.',
                "source_url": original_match.metadata.get('source_url')
            }
            # Crear una referencia para el prompt
            year_str = str(year) if year else 's.f.'
            source_details_for_prompt += f"- {publisher} ({year_str}). *{title}*.\n"

    
    print(f"Constructed context length: {len(context)} characters")
    print(f"Found {len(sources)} unique sources.")

    # Si el contexto sigue vacío después del bucle, devolvemos la respuesta directamente.
    if not context:
        print("Context is empty. Returning 'not found' message.")
        print("--- RAG DEBUG END ---\n")
        return {"answer": "La información no se encuentra en la base de conocimiento.", "sources": []}

    # 6. Construir el prompt para el LLM
    prompt_template = f"""
    Eres un asistente de investigación experto. Tu tarea es responder a la pregunta del usuario basándote únicamente en el contexto y las fuentes proporcionadas.
    INSTRUCCIONES:
    1.  Lee el contexto y la pregunta cuidadosamente.
    2.  Formula una respuesta completa y detallada en español, que integre los mejores detalles del contexto para responder a la pregunta del usuario.
    3.  CADA VEZ que utilices información de una fuente, debes citarla al final de la oración usando el formato (Autor, Año). Por ejemplo: (Foro Económico Mundial, 2023).
    4.  Si la respuesta requiere información de múltiples fuentes en una misma oración, cítalas todas, por ejemplo: (Autor 1, Año 1; Autor 2, Año 2).
    5.  Si la información no se encuentra en el contexto, responde exactamente: "La información no se encuentra en la base de conocimiento.".
    6.  No inventes información, no especules ni hagas suposiciones, tampoco repitas o menciones la pregunta del usuario en tu respuesta.

    FUENTES DISPONIBLES PARA CITAR:
    {source_details_for_prompt}
    CONTEXTO:
    {context}

    PREGUNTA: {query}
    """

    # 7. Generar la respuesta final con el LLM
    print("Sending prompt to LLM...")
    final_response = client_openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt_template}],
        temperature=0.1
    )
    answer = final_response.choices[0].message.content
    print(f"LLM Answer: {answer}")
    print("--- RAG DEBUG END ---\n")

    return {"answer": answer, "sources": list(sources.values())}
