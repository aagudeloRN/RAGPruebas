
# app/modules/query_decomposition/decomposer_agent.py
import logging
import json
from typing import Dict, Any

from openai import AsyncOpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)
client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

DECOMPOSER_PROMPT = """
Tu rol es ser un agente experto en descomposición de consultas.
Dada una pregunta del usuario, tu objetivo es determinar si puede ser respondida en un solo paso (simple) o si requiere múltiples pasos de búsqueda y razonamiento (compleja).

**Instrucciones:**

1.  **Analiza la Pregunta:** Lee la pregunta del usuario cuidadosamente.
2.  **Determina la Complejidad:**
    *   **Simple:** Si la pregunta busca un hecho directo. Ej: "¿Qué es la tecnología blockchain?", "¿Quién es el CEO de Apple?".
    *   **Compleja:** Si la pregunta requiere encontrar una entidad primero y luego buscar información sobre esa entidad. Ej: "¿Cuál es el último libro del autor de 'Cien Años de Soledad'?", "¿Qué empresa fundó el director de la película 'Oppenheimer'?".
3.  **Genera un Plan JSON:** Responde únicamente con un objeto JSON que siga esta estructura:

    ```json
    {{
      "is_complex": boolean,
      "steps": [
        "pregunta_paso_1",
        "pregunta_paso_2",
        ...
        "pregunta_final_de_síntesis"
      ]
    }}
    ```

**Reglas del JSON:**

*   `is_complex`: `false` si la pregunta es simple, `true` si es compleja.
*   `steps`:
    *   Si es simple, el array debe contener una sola cadena: la pregunta original del usuario.
    *   Si es compleja, el array debe contener una serie de preguntas simples que, al ser respondidas en orden, resuelven la pregunta original. La última pregunta en el array siempre debe ser una instrucción para sintetizar la respuesta final.

**Ejemplos:**

*   **Usuario:** "¿Qué son las stablecoins?"
    **Tu Salida JSON:**
    ```json
    {{
      "is_complex": false,
      "steps": [
        "¿Qué son las stablecoins?"
      ]
    }}
    ```

*   **Usuario:** "¿Cuál fue el primer cargo público del autor del informe sobre el futuro del trabajo de 2025?"
    **Tu Salida JSON:**
    ```json
    {{
      "is_complex": true,
      "steps": [
        "¿Quién es el autor del 'Future of Jobs Report 2025'?",
        "¿Cuál fue el primer cargo público de la persona identificada en el paso anterior?",
        "Sintetiza la respuesta final combinando la información de los pasos anteriores."
      ]
    }}
    ```

Ahora, analiza la siguiente pregunta del usuario.

**Pregunta del Usuario:**
{query}
"""

async def decompose_query_into_plan(query: str) -> Dict[str, Any]:
    """
    Usa un LLM para descomponer una consulta de usuario en un plan de ejecución.
    """
    logger.info(f"Descomponiendo la consulta: '{query}'")
    try:
        response = await client.chat.completions.create(
            model=settings.OPENAI_LLM_MODEL,
            messages=[
                {"role": "system", "content": DECOMPOSER_PROMPT.format(query=query)}
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        plan_str = response.choices[0].message.content
        plan = json.loads(plan_str)
        logger.info(f"Plan generado: {plan}")
        return plan
    except Exception as e:
        logger.error(f"Error al descomponer la consulta: {e}", exc_info=True)
        # Si falla, devolvemos un plan simple por defecto
        return {
            "is_complex": False,
            "steps": [query]
        }
