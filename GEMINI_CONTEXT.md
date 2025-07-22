# Gemini CLI Project Context Summary

**Last Updated:** 2025-07-16

## Project Overview
This is a Retrieval-Augmented Generation (RAG) system designed to process PDF documents, extract information, and provide intelligent answers based on the ingested knowledge. It uses FastAPI for the web interface, SQLAlchemy for database management, Pinecone for vector storage, OpenAI for embeddings and LLM interactions, and Cohere for re-ranking.

## Current State
The project is in a stable state after significant refactoring and feature implementation.

### Key Features Implemented:

### Recent Enhancements:
-   **Technique 2: Re-ranking de Resultados (con Cohere)**
    -   **Objetivo:** Obtener un contexto más amplio y relevante para el LLM.
    -   **Acciones Implementadas:**
        -   Incrementado `top_k` de la recuperación inicial de Pinecone de 10 a 15 (número de chunks iniciales).
        -   Incrementado `top_n` del re-ranking de Cohere de 8 a 12 (número de chunks más relevantes seleccionados).

-   **Technique 3: Ingeniería de Prompts Avanzada**
    -   **Objetivo:** Instruir al LLM para que genere respuestas más completas, profundas, estructuradas y basadas en datos.
    -   **Acciones Implementadas:**
        -   Comentado el prompt anterior en `app/services/rag_service.py` como respaldo.
        -   Implementado un nuevo prompt de "Analista Experto" que solicita al LLM:
            -   Síntesis profunda de la información.
            -   Priorización de datos cuantitativos (estadísticas, cifras, rankings).
            -   Identificación de tendencias y patrones.
            -   Estructura clara de la respuesta usando Markdown.
            -   Citas precisas de las fuentes.
            -   Manejo claro de la incertidumbre si la información no está en el contexto.


-   **Core RAG System:** Document ingestion, embedding generation, Pinecone upsert, and RAG query processing.
-   **Multi-Knowledge-Base Architecture:**
    -   A new landing page (`/`) allows users to select a specific knowledge base (namespace).
    -   The selected knowledge base is stored in a cookie for session persistence.
-   **Intelligent Document Ingestion System (Recently Overhauled):**
    -   **Automated & Streamlined UI:** The PDF upload process is now fully automated. The user can drag & drop a file or select it, and the metadata extraction begins immediately.
    -   **User-Friendly Validation:** The user is redirected to a validation page where they can review and complete the auto-extracted metadata. All required fields are clearly marked with an asterisk (*).
    -   **Non-Blocking Workflow:** The final processing is handled as a background task, allowing the user to immediately navigate to other parts of the application (like the library) or upload a new document without waiting.
    -   **Clear Navigation:** The validation page now includes the main navigation bar, and upon successful submission, provides clear action buttons to either visit the library or upload another document.
    -   **Crucially: No data is persisted in PostgreSQL or Pinecone until the user explicitly confirms the metadata.**
-   **Intelligent Filtering System in the Library:**
    -   Semantic search functionality in the document library (`/library`).
    -   Results are displayed uniquely and ordered by relevance.
    -   Added filtering by publication year range.

### Project Cleanup & Refactoring:
-   **Bug Fix:** Restored the `/documents/` API endpoint, fixing a 404 error that prevented the document library from loading.
-   Removed old/redundant files (`documentOLD.py`, `sessionOLD.py`).
-   Consolidated virtual environments and fixed module import errors.
-   Added missing configuration variables to `app/core/config.py`.
-   Updated `.gitignore` and installed `ruff` linter.

## Next Planned Phases (Prioritized)

The following improvements are planned, categorized by estimated difficulty:

### Phase 1: Quick Wins & Core Enhancements (Easy/Medium) - ✅ COMPLETED

1.  **Implement Q&A Memory/Cache & FAQ Generation:**
    *   **Status:** ✅ **Done (V1).**
    *   **Objective:** Expedite automated responses, reduce costs and latency for recurring queries.
    *   **Details:** A database cache for Questions & Answers has been implemented. The system currently checks for exact matches before performing a full RAG search.
    *   **Next Step (Future Phase):** Enhance the cache to perform semantic searches (e.g., keyword/synonym analysis) to handle similar, non-identical questions.

2.  **Enhanced Document Organization & Filtering in Library:**
    *   **Status:** ⏳ **Pending.**
    *   **Objective:** Improve user experience and document discoverability within the library.
    *   **Details:** Add UI controls for sorting documents (A/Z, Z/A, by topic/category) and implement a keyword filtering mechanism with autocomplete suggestions.

### Phase 2: Expanding Knowledge & Query Intelligence (Medium/Difficult) - ⏳ PENDING

3.  **Integrate External Databases (Airtable, DANE, etc.):**
    *   **Status:** ⏳ **Pending.**
    *   **Objective:** Enrich the knowledge base with structured information from various external sources.
    *   **Note:** Initial attempts to integrate Airtable were reverted to simplify the system for its first production deployment. The focus is now on a stable RAG-only system. Future integrations will build upon this stable core.

4.  **Intelligent Query Refinement Tool:
    *   **Objective:** Improve the quality of user queries before information retrieval, leading to more precise and relevant answers.
    *   **Details:** Utilize a powerful LLM to rephrase, expand, or decompose user questions. This tool will incorporate corporate context (e.g., internal glossaries, past interactions) to better understand and refine queries.

5.  **Integrate Traditional Databases (DANE, Geographic Data, Google Maps):**
    *   **Objective:** Broaden the system's knowledge base beyond vector stores to include specific, structured data from external sources.
    *   **Details:** Develop connectors and query mechanisms for databases like DANE (statistical data), geographic information systems, and Google Maps, allowing the system to consult these sources for highly specific data points.

### Phase 3: Advanced Orchestration & Continuous Improvement (Difficult)

6.  **Intelligent Query Routing:**
    *   **Objective:** Optimize response generation by directing queries to the most appropriate data source (e.g., Q&A cache, Pinecone, Airtable, DANE) based on query intent.
    *   **Details:** Implement an orchestration layer (potentially using an LLM) to analyze incoming questions and route them efficiently, reducing processing time and costs.

7.  **Feedback Loop for Continuous RAG Improvement:**
    *   **Objective:** Enable the system to learn and improve over time based on user feedback.
    *   **Details:** Implement a mechanism for users to rate response quality. This feedback will be used to identify areas for improvement, adjust RAG parameters, and prioritize knowledge base updates.

8.  **Sentiment and Tone Analysis in Queries:**
    *   **Objective:** Enhance user interaction by adapting responses based on the emotional context of the user's query.
    *   **Details:** Integrate an NLP model to detect sentiment or tone in user questions, allowing the system to generate more empathetic or direct responses as appropriate.

9.  **Integration with Collaboration Tools (e.g., Slack, Microsoft Teams):**
    *   **Objective:** Increase accessibility and adoption by allowing users to interact with the RAG system directly within their daily communication platforms.
    *   **Details:** Develop specific bots or connectors to enable seamless querying and response delivery within popular corporate collaboration environments.

### UX/UI Improvements (Ongoing)

*   **Chat/Query Interface Enhancement:**
    *   **Objective:** Make the RAG query interface (`/query/`) to be more conversational and user-friendly.
    *   **Details:** Implement a pre-processing step to help users refine their search queries based on conversation history before performing the RAG sweep.

*   **Visual Design Overhaul:**
    *   **Objective:** Enhance the application's visual design to be more corporate and user-friendly.
    *   **Details:** Requires design guidelines (logo, color palette, typography, layout preferences) to implement a polished and professional UI/UX.

## Instructions for Future Sessions
**IMPORTANT:** At the beginning of any future session, please read this `GEMINI_CONTEXT.md` file to quickly recall the project's state and ongoing tasks. This will ensure continuity and efficiency.

## Next Interaction Focus
The project has been simplified to focus solely on the RAG (Pinecone) functionality for its initial deployment. The next steps will involve preparing the application for deployment on Railway.