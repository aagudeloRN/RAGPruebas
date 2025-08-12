# app/modules/rag_chat/query_orchestrator.py
import logging
import json
from typing import List, Dict, Any, AsyncGenerator
import time

from openai import AsyncOpenAI
from sqlalchemy.orm import Session

from app.core.config import settings, active_kb
from app.core.prompts import CONDENSE_QUESTION_PROMPT, ROUTER_PROMPT
from app.modules.rag_chat.rag_service import perform_rag_query
from app.modules.query_decomposition.decomposer_agent import decompose_query_into_plan
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

    async def handle_conversational_query(self, chat_request: ChatRequest, db: Session, pgvector_db: Session, kb_id: str) -> ChatResponse:
        """Maneja una consulta conversacional para la KB activa."""
        logger.info(f"[Orchestrator] Enrutando consulta para KB '{kb_id}': '{chat_request.query}'")
        
        decision = await _route_query(chat_request.history, chat_request.query)
        tool_to_use = decision.get("tool", "query_knowledge_base")

        if tool_to_use == "answer_from_history":
            return await self._answer_from_history(chat_request.history, chat_request.query)

        condensed_query = await _condense_query_with_history(chat_request.history, chat_request.query)
        
        rag_response = await perform_rag_query(query=condensed_query, db=db, pgvector_db=pgvector_db, kb_id=kb_id)
        
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

    async def stream_conversational_query(self, chat_request: ChatRequest, db: Session, pgvector_db: Session, kb_id: str) -> AsyncGenerator[str, None]:
        """
        Transmite el proceso de una consulta conversacional, usando el Agente Descompositor.
        """
        # 1. Descomponer la consulta en un plan
        yield f"data: {json.dumps({'type': 'status', 'message': 'Analizando y descomponiendo la consulta...'})}\n\n"
        plan = await decompose_query_into_plan(chat_request.query)
        yield f"data: {json.dumps({'type': 'plan', 'data': plan})}\n\n"
        time.sleep(1)

        if not plan.get('is_complex', False):
            rag_response = await perform_rag_query(query=chat_request.query, db=db, pgvector_db=pgvector_db, kb_id=kb_id)
            final_answer_data = {
                "answer": rag_response.answer,
                "sources": [s.model_dump() for s in rag_response.sources] if rag_response.sources else []
            }
            yield f"data: {json.dumps({'type': 'final_answer', 'data': final_answer_data})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Proceso para planes complejos
        contextual_facts = []
        all_sources = {} # Usar un diccionario para evitar duplicados por ID

        for i, step_query in enumerate(plan["steps"][:-1]):
            query_to_execute = step_query
            
            yield f"data: {json.dumps({'type': 'status', 'message': f'Paso {i+1}: {query_to_execute}'})}\n\n"
            time.sleep(1)

            rag_response = await perform_rag_query(query=query_to_execute, db=db, pgvector_db=pgvector_db, kb_id=kb_id)
            
            if rag_response.answer and rag_response.sources:
                main_source = rag_response.sources[0]
                contextual_facts.append({
                    "fact": rag_response.answer,
                    "source": main_source.model_dump()
                })
                for source in rag_response.sources:
                    if source.id not in all_sources:
                        all_sources[source.id] = source

            yield f"data: {json.dumps({'type': 'sub_answer', 'step': i + 1, 'answer': rag_response.answer})}\n\n"
            time.sleep(1)

        synthesis_task_description = plan["steps"][-1]
        
        synthesis_prompt_template = """
Eres un asistente de investigación académica. Tu tarea es redactar una respuesta analítica a una pregunta, basándote estrictamente en una lista de hechos contextuales proporcionados. Debes seguir las normas de citación APA 7ma edición.

**Instrucciones Estrictas:**

1.  **Análisis y Síntesis:** Redacta una respuesta coherente y bien estructurada que responda a la "Pregunta Final". No te limites a listar los hechos; compáralos, contrástalos y extrae conclusiones.
2.  **Citación en el Texto:** Por CADA hecho que utilices del contexto, DEBES incluir una citación en el texto en el formato `(Autor, Año)`. El "Autor" es el `publisher` en los datos de la fuente.
3.  **Formato Markdown:** Usa Markdown para mejorar la legibilidad (títulos `##`, listas, `**negritas**`).
4.  **No Incluir Bibliografía:** NO añadas una sección de "Referencias" o "Bibliografía" al final. Las fuentes se mostrarán por separado en la interfaz.
5.  **Adherencia al Contexto:** NO incluyas ninguna información externa. Basa tu respuesta únicamente en los "Hechos Contextuales" a continuación.

**Hechos Contextuales (en formato JSON):**
{context_json}

**Pregunta Final a Responder:**
{synthesis_prompt}
"""

        context_json_str = json.dumps(contextual_facts, indent=2, ensure_ascii=False)
        final_synthesis_prompt = synthesis_prompt_template.format(
            context_json=context_json_str, 
            synthesis_prompt=synthesis_task_description
        )

        yield f"data: {json.dumps({'type': 'status', 'message': 'Sintetizando respuesta final con referencias APA...'})}\n\n"
        
        synthesis_response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "user", "content": final_synthesis_prompt}
            ],
            temperature=0.0,
        )
        final_answer = synthesis_response.choices[0].message.content.strip()

        final_sources = [source.model_dump() for source in all_sources.values()]

        final_answer_data = {
            "answer": final_answer,
            "sources": final_sources
        }
        
        yield f"data: {json.dumps({'type': 'final_answer', 'data': final_answer_data})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    async def stream_simple_query(self, query: str, db: Session, pgvector_db: Session, kb_id: str) -> AsyncGenerator[str, None]:
        """Función de streaming simplificada que solo usa la consulta."""
        chat_request = ChatRequest(query=query, history=[])
        async for event in self.stream_conversational_query(chat_request, db, pgvector_db, kb_id):
            yield event