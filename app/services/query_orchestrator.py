# app/services/query_orchestrator.py
from app.services.rag_service import perform_rag_query
from ..schemas.document import QueryResponse
import logging

logger = logging.getLogger(__name__)

class QueryOrchestrator:
    """
    Una clase simplificada para dirigir todas las consultas directamente al sistema RAG.
    """
    def decide_and_execute(self, query: str, kb_id: str) -> QueryResponse:
        """
        Redirige la consulta directamente al servicio RAG sin lógica de decisión.
        """
        logger.info(f"[Orchestrator] Consulta recibida, redirigiendo directamente a RAG para el namespace: {kb_id}")
        try:
            # Llama directamente a la función de RAG
            return perform_rag_query(query=query, namespace=kb_id)
        except Exception as e:
            logger.error(f"Error inesperado al ejecutar la consulta RAG: {e}", exc_info=True)
            return QueryResponse(answer=f"Lo siento, ocurrió un error inesperado al procesar su consulta: {e}", sources=[])