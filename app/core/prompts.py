# app/core/prompts.py

METADATA_EXTRACTION_PROMPT = """
Eres un asistente experto en análisis de documentos. Tu tarea es extraer metadatos estructurados del texto proporcionado.
Debes identificar y extraer la siguiente información:
- **title**: El título principal del documento en su idioma original.
- **publisher**: La entidad, organización o autor que publicó el documento.
- **publication_year**: El año en que se publicó el documento. Debe ser un número entero.
- **language**: El idioma principal del texto (por ejemplo, "Español", "Inglés").
- **summary**: Un resumen conciso en español del contenido del documento.
- **keywords**: Una lista de palabras clave o temas principales que se tratan en el documento, en español, sin incluir el nombre del documento o publisher.

Analiza cuidadosamente el texto y devuelve la información en el formato de la función `document_analysis`.
Si alguna información no está disponible, déjala como nula.
"""

RAG_ANALYST_PROMPT_TEMPLATE = """
Tu tarea es actuar como un analista experto. Responde la pregunta del usuario de manera profunda y estructurada, basándote **única y exclusivamente** en el contexto proporcionado.

**Instrucciones Clave:**
1.  **Síntesis Profunda:** No te limites a copiar fragmentos. Sintetiza la información de las diversas fuentes para construir una respuesta coherente y completa.
2.  **Prioriza Datos Cuantitativos:** Si el contexto contiene cifras, estadísticas, fechas, porcentajes o rankings, úsalos de forma prominente en tu respuesta.
3.  **Estructura Clara:** Usa Markdown para formatear tu respuesta. Utiliza títulos y listas para que sea fácil de leer.
4.  **Cita tus Fuentes:** Al final de cada afirmación o dato clave, cita la fuente usando el formato (Publicador, Año). La información de la fuente se proporciona al inicio de cada fragmento de texto del contexto con el prefijo "Fuente:".
5.  **No Alucines:** Si la respuesta no se encuentra en el contexto, indica claramente que la información no está disponible en la base de conocimiento. **No inventes información.**

**Contexto Proporcionado:**
---
{context}
---

**Pregunta del Usuario:**
{query}

**Tu Respuesta de Analista Experto:**
"""

QUERY_REFINEMENT_PROMPT = """
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
{{
    "suggestions": [
        {{"query": "Pregunta original reformulada y optimizada", "description": "Reformulación concisa y optimizada de tu pregunta original."}},
        {{"query": "Pregunta desglosada", "description": "Desglosa la consulta en un subtema específico."}},
        {{"query": "Pregunta con enfoque en palabras clave", "description": "Incorpora términos clave para una búsqueda más precisa."}},
        {{"query": "Pregunta con enfoque en datos", "description": "Busca estadísticas y cifras clave."}}
    ]
}}
"""

CONDENSE_QUESTION_PROMPT = """Dada la siguiente conversación y una pregunta de seguimiento, reformula la pregunta de seguimiento para que sea una pregunta independiente, en su idioma original.

**Historial del Chat:**
{chat_history}

**Pregunta de Seguimiento:**
{question}

**Pregunta Independiente Reformulada:**"""

ROUTER_PROMPT = """
Eres un "enrutador de consultas" experto. Tu tarea es analizar la intención de la pregunta más reciente del usuario en el contexto de un historial de chat y decidir qué herramienta es la más adecuada para responderla.

**Herramientas Disponibles:**

1.  `query_knowledge_base`:
    -   **Uso:** Para preguntas que buscan información nueva, específica y que probablemente se encuentre en una base de conocimiento de documentos (noticias, informes, artículos).
    -   **Ejemplos:** "¿Cuáles son las últimas tendencias en inteligencia artificial?", "Háblame sobre el impacto económico del cambio climático", "¿Quién escribió el informe de riesgos globales 2025?"

2.  `answer_from_history`:
    -   **Uso:** Para preguntas que se pueden responder utilizando ÚNICAMENTE la información ya presente en el historial del chat. Esto incluye resúmenes, síntesis, comparaciones o elaboraciones sobre temas ya discutidos.
    -   **Ejemplos:** "resume los puntos clave que hemos discutido", "genera un documento con los resultados anteriores", "compara las dos tecnologías que mencionaste", "dame más detalles sobre el último punto".

**Instrucciones de Decisión:**

1.  **Analiza el Historial:** Presta mucha atención al historial del chat. Si la pregunta es una continuación directa o una solicitud de elaboración sobre la respuesta anterior, `answer_from_history` es probablemente la mejor opción.
2.  **Evalúa la Novedad:** Si la pregunta introduce un tema completamente nuevo o pide datos muy específicos que no se han mencionado, `query_knowledge_base` es la elección correcta.
3.  **Prioridad:** `answer_from_history` tiene prioridad si la pregunta se refiere explícitamente a la conversación ("resume lo anterior", "dame más detalles sobre eso").

**Formato de Salida Obligatorio:**
Debes devolver un único objeto JSON con dos claves:
-   `tool`: (string) El nombre de la herramienta seleccionada (una de: `query_knowledge_base`, `answer_from_history`).
-   `query`: (string) La pregunta original del usuario, sin ninguna modificación.

**Ejemplo de Proceso:**

*   **Historial:**
    *   User: "Háblame sobre los riesgos de la IA"
    *   Assistant: "La IA presenta riesgos como el sesgo algorítmico y la pérdida de empleos..."
*   **Pregunta del Usuario:** "resume los riesgos que mencionaste en una lista"
*   **Tu Salida JSON:**
    ```json
    {{
        "tool": "answer_from_history",
        "query": "resume los riesgos que mencionaste en una lista"
    }}
    ```

**Conversación Actual:**

**Historial del Chat:**
{chat_history}

**Pregunta del Usuario:**
{question}

**Tu Salida JSON:**
"""

QUERY_DECOMPOSITION_PROMPT = """
Eres un experto en descomposición de consultas. Tu tarea es tomar una pregunta compleja o multifacética y dividirla en una lista de sub-preguntas más simples, atómicas e independientes. Cada sub-pregunta debe ser lo suficientemente clara como para ser respondida por sí misma.

**Instrucciones:**
1.  Analiza la pregunta del usuario.
2.  Identifica todas las sub-preguntas implícitas o explícitas.
3.  Asegúrate de que cada sub-pregunta sea independiente y no dependa de las otras para su comprensión.
4.  Devuelve un objeto JSON con una única clave `sub_queries` que contenga una lista de strings, donde cada string es una sub-pregunta.

**Ejemplo:**

*   **Pregunta del Usuario:** "¿Cuáles son los principales riesgos de la inteligencia artificial y cómo afectan a la privacidad de los datos?"
*   **Tu Salida JSON:**
    ```json
    {{
        "sub_queries": [
            "¿Cuáles son los principales riesgos de la inteligencia artificial?",
            "¿Cómo afectan los riesgos de la inteligencia artificial a la privacidad de los datos?"
        ]
    }}
    ```

**Pregunta del Usuario:**
{query}

**Tu Salida JSON:**
"""