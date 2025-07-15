# Gemini CLI Project Context Summary

**Last Updated:** 2025-07-14

## Project Overview
This is a Retrieval-Augmented Generation (RAG) system designed to process PDF documents, extract information, and provide intelligent answers based on the ingested knowledge. It uses FastAPI for the web interface, SQLAlchemy for database management, Pinecone for vector storage, OpenAI for embeddings and LLM interactions, and Cohere for re-ranking.

## Current State
The project is in a stable state after significant refactoring and the implementation of a multi-knowledge-base architecture, intelligent document ingestion, and intelligent library filtering.

### Key Features Implemented:
-   **Core RAG System:** Document ingestion, embedding generation, Pinecone upsert, and RAG query processing.
-   **Multi-Knowledge-Base Architecture:**
    -   A new landing page (`/`) allows users to select a specific knowledge base (namespace).
    -   The selected knowledge base is stored in a cookie for session persistence.
    -   Document ingestion and RAG queries are performed within the selected namespace.
-   **Intelligent Document Ingestion System:**
    -   Simplified PDF upload (file only).
    -   Automated metadata extraction using LLM.
    -   Single-page user validation and completion of extracted metadata.
    -   **Crucially: No data is persisted in PostgreSQL or Pinecone until the user explicitly confirms the metadata.** Temporary PDF storage is handled in memory during the validation phase.
-   **Intelligent Filtering System in the Library:**
    -   Semantic search functionality in the document library (`/library`).
    -   Results are displayed uniquely and ordered by relevance.
    -   Added filtering by publication year range.

### Project Cleanup & Refactoring:
-   Removed old/redundant files (`documentOLD.py`, `sessionOLD.py`).
-   Consolidated virtual environments (removed `venv`, recreated `.venv` for Linux compatibility).
-   Fixed `ModuleNotFoundError` by renaming `app/db/base_class.py` to `app/db/base.py`.
-   Added missing configuration variables (`PINECONE_INDEX_NAME`, `PINECONE_BATCH_SIZE`, `OPENAI_EMBEDDING_MODEL`, `OPENAI_LLM_MODEL`) to `app/core/config.py` with default values.
-   Updated `.gitignore` to properly ignore `.venv/` and `.vscode/`.
-   Installed `ruff` linter for static code analysis.
-   Corrected `AttributeError` in LLM response parsing.
-   Ensured `filename` is correctly handled during document creation.

## Next Planned Phases (Prioritized)
The following improvements are planned:

1.  **Chat of Query more Conversational:**
    -   Enhance the RAG query interface (`/query/`) to be more conversational.
    -   Implement a pre-processing step where the system helps the user refine their search query based on conversation history before performing the RAG sweep.
2.  **Visual Improvements (UI/UX):**
    -   Enhance the application's visual design to be more corporate and user-friendly. (Requires design guidelines: logo, color palette, typography, layout preferences).

## Instructions for Future Sessions
**IMPORTANT:** At the beginning of any future session, please read this `GEMINI_CONTEXT.md` file to quickly recall the project's state and ongoing tasks. This will ensure continuity and efficiency.