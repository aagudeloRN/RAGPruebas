# app/modules/rag_chat/query_orchestrator.py
import logging
import json
from typing import List, Dict, Any

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.core.config import settings, active_kb
from app.core.prompts import CONDENSE_QUESTION_PROMPT, ROUTER_PROMPT
from app.modules.rag_chat.rag_service import perform_rag_query
from app.schemas.chat import ChatRequest, ChatResponse, ChatMessage
from app.schemas.document import QueryResponse

logger = logging.getLogger(__name__)

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

async def _condense_query_with_history(history: List[ChatMessage], query: str) -> str:
    if not history:
        return query
    chat_history_str = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in history])
    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": CONDENSE_QUESTION_PROMPT.format(chat_history=chat_history_str, question=query)}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error al condensar la consulta: {e}", exc_info=True)
        return query

async def _route_query(history: List[ChatMessage], query: str) -> Dict[str, Any]:
    chat_history_str = "\n".join([f"{msg.role.capitalize()}: {msg.content}" for msg in history])
    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT.format(chat_history=chat_history_str, question=query)}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Error en el enrutador de consultas: {e}", exc_info=True)
        return {"tool": "query_knowledge_base", "query": query}

class QueryOrchestrator:
    """Orquesta las consultas, enrutándolas a la herramienta adecuada para la KB activa."""

    async def decide_and_execute(self, query: str, db: Session, pgvector_db: Session) -> QueryResponse:
        """Maneja una consulta simple y directa (sin historial) para la KB activa."""
        logger.info(f"[Orchestrator] Consulta simple para KB '{active_kb.id}', redirigiendo a RAG.")
        return await perform_rag_query(query=query, db=db, pgvector_db=pgvector_db)

    async def handle_conversational_query(self, chat_request: ChatRequest, db: Session, pgvector_db: Session) -> ChatResponse:
        """Maneja una consulta conversacional para la KB activa."""
        logger.info(f"[Orchestrator] Enrutando consulta para KB '{active_kb.id}': '{chat_request.query}'")
        
        decision = await _route_query(chat_request.history, chat_request.query)
        tool_to_use = decision.get("tool", "query_knowledge_base")

        if tool_to_use == "answer_from_history":
            # Esta lógica es independiente de la KB
            return await self._answer_from_history(chat_request.history, chat_request.query)

        condensed_query = await _condense_query_with_history(chat_request.history, chat_request.query)
        
        rag_response = await perform_rag_query(query=condensed_query, db=db, pgvector_db=pgvector_db)
        
        return ChatResponse(
            answer=rag_response.answer,
            sources=rag_response.sources,
            cache_hit=rag_response.cache_hit
        )

    async def _answer_from_history(self, history: List[ChatMessage], query: str) -> ChatResponse:
        logger.info("Generando respuesta basada únicamente en el historial del chat.")
        # Lógica para responder desde el historial (independiente de la KB)
        # ... (código omitido por brevedad, es el mismo que antes)
        return ChatResponse(answer="Lo siento, no pude encontrar una respuesta en el historial.", sources=[])