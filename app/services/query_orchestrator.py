# app/services/query_orchestrator.py
import logging
import json
from typing import List, Dict, Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.core.prompts import CONDENSE_QUESTION_PROMPT, ROUTER_PROMPT
from app.services.rag_service import perform_rag_query
from ..schemas.chat import ChatRequest, ChatResponse, ChatMessage
from ..schemas.document import QueryResponse

logger = logging.getLogger(__name__)

# Inicializar el cliente de OpenAI de forma asíncrona
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def _condense_query_with_history(history: List[ChatMessage], query: str) -> str:
    """Condensa el historial y la nueva pregunta en una única pregunta auto-contenida."""
    if not history:
        return query

    chat_history_str = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in history])
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": CONDENSE_QUESTION_PROMPT.format(chat_history=chat_history_str, question=query)},
                {"role": "user", "content": query}
            ],
            temperature=0.0,
        )
        condensed_query = response.choices[0].message.content.strip()
        logger.info(f"Pregunta original: '{query}' -> Pregunta condensada: '{condensed_query}'")
        return condensed_query
    except Exception as e:
        logger.error(f"Error al condensar la consulta: {e}", exc_info=True)
        return query

async def _route_query(history: List[ChatMessage], query: str) -> Dict[str, Any]:
    """Decide qué herramienta usar analizando la intención de la consulta."""
    chat_history_str = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in history])
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": ROUTER_PROMPT.format(chat_history=chat_history_str, question=query)}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        decision = json.loads(response.choices[0].message.content)
        logger.info(f"Decisión del enrutador: {decision}")
        return decision
    except Exception as e:
        logger.error(f"Error en el enrutador de consultas: {e}", exc_info=True)
        # Fallback seguro: si el enrutador falla, asumimos que es una consulta a la base de conocimiento
        return {"tool": "query_knowledge_base", "query": query}

async def _answer_from_history(history: List[ChatMessage], query: str) -> ChatResponse:
    """Genera una respuesta utilizando únicamente el historial del chat."""
    logger.info("Generando respuesta basada en el historial del chat.")
    chat_history_str = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in history])
    prompt = f"Basándote únicamente en el siguiente historial de chat, responde a la última pregunta del usuario.\n\n**Historial:**\n{chat_history_str}\n\n**Pregunta:**\n{query}\n\n**Respuesta:**"
    
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt}
            ],
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        # Como la respuesta se basa en el historial, no hay nuevas fuentes que citar.
        return ChatResponse(answer=answer, sources=[])
    except Exception as e:
        logger.error(f"Error al generar respuesta desde el historial: {e}", exc_info=True)
        return ChatResponse(answer="Lo siento, no pude procesar tu solicitud utilizando el historial.", sources=[])

class QueryOrchestrator:
    """
    Orquesta las consultas de forma inteligente, enrutándolas a la herramienta adecuada.
    """
    def decide_and_execute(self, query: str, kb_id: str) -> QueryResponse:
        """Maneja una consulta simple y directa (sin historial)."""
        logger.info(f"[Orchestrator] Consulta simple recibida, redirigiendo a RAG para el namespace: {kb_id}")
        return perform_rag_query(query=query, namespace=kb_id)

    async def handle_conversational_query(self, chat_request: ChatRequest, kb_id: str) -> ChatResponse:
        """
        Maneja una consulta conversacional, enrutándola a la herramienta correcta.
        """
        logger.info(f"[Orchestrator] Iniciando enrutamiento para la consulta: '{chat_request.query}'")
        
        # 1. Decidir la ruta
        decision = await _route_query(chat_request.history, chat_request.query)
        tool_to_use = decision.get("tool", "query_knowledge_base")

        # 2. Ejecutar la herramienta seleccionada
        if tool_to_use == "answer_from_history":
            # El historial completo es necesario aquí
            full_history = chat_request.history + [ChatMessage(role="user", content=chat_request.query)]
            return await _answer_from_history(full_history, chat_request.query)

        # Fallback y ruta principal: query_knowledge_base (incluye verificación de caché)
        # 3. Condensar la pregunta si es necesario
        condensed_query = await _condense_query_with_history(chat_request.history, chat_request.query)
        
        # 4. Ejecutar la consulta RAG con la pregunta condensada
        rag_response = await perform_rag_query(query=condensed_query, namespace=kb_id)
        
        # 5. Devolver la respuesta en el formato de ChatResponse
        return ChatResponse(
            answer=rag_response.answer,
            sources=rag_response.sources,
            cache_hit=rag_response.cache_hit
        )