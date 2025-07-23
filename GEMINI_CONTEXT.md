# Project Summary: RAG Intelligence System

**Last Updated:** 2025-07-23

## 1. Overview
This is a Retrieval-Augmented Generation (RAG) system designed to process PDF documents, extract information, and provide intelligent answers based on the ingested knowledge.
**Tech Stack:** FastAPI (web), SQLAlchemy (database), Pinecone (vector storage), and various AI models including Gemini, GPT, and Cohere (language processing).

## 2. Current Status & Key Features

The project is stable and deployed for demonstration purposes.
-   **Deployment:** Successfully deployed to Railway on the `multi-agent-architecture` branch, including the PostgreSQL database.
-   **Multi-Knowledge-Base Architecture:** Allows users to select a specific Pinecone namespace at the start of a session.
-   **Intelligent Document Ingestion:**
    -   **Automated Workflow:** PDF upload with automatic metadata extraction.
    -   **User Validation:** A UI for users to review and confirm metadata before any data is persisted.
    -   **Background Processing:** Final ingestion runs as a background task to avoid blocking the UI.
-   **Advanced Document Library:**
    -   **Semantic Search:** For meaning-based queries, not just keywords.
    -   **Advanced Filtering:** By publication year range, keywords, and publisher/author.
    -   **Pagination & Sorting:** Controls for navigating and ordering results.
-   **Optimized RAG Engine:**
    -   **Context Re-ranking (Cohere):** Improves the relevance of the context fed to the LLM.
    -   **Advanced Prompt Engineering:** Uses an "Expert Analyst" prompt to generate deep, structured, data-driven answers with citations.
    -   **Intelligent Query Refinement:** An LLM rewrites and expands user queries for more accurate RAG results.
-   **Q&A Cache (Memory):**
    -   Stores exact question-answer pairs in the database to accelerate responses for recurring queries, reducing cost and latency.

## 3. Next Steps & Future Implementations

-   **Traditional Database Integration:**
    -   **Objective:** Enrich RAG answers by connecting to structured data sources (e.g., DANE for industry/technology stats).
    -   **Details:** Develop connectors and query mechanisms to consult external databases for highly specific data points.
-   **Intelligent Query Routing:**
    -   **Objective:** Create an orchestrator to route queries to the most appropriate data source (Cache, Pinecone, or a traditional DB) to optimize efficiency and enrich results.
-   **Q&A Cache Enhancement:**
    -   **Objective:** Evolve the cache to perform semantic searches for similar (not just identical) questions.
    -   **Task:** Generate a publicly viewable list of the "Top 5" most frequently asked questions.
 -   **User Feedback Loop:**
    -   **Objective:** Implement a system for users to rate answer quality, enabling continuous improvement of the RAG system.
-   **UI/UX Enhancements:**
    -   Make the chat interface more conversational.
    -   Implement a polished, corporate visual redesign.
    -   Improve the pagination system in the document library.

## 4. Deprioritized / Archived Ideas
-   **Airtable Integration:** Initial attempts were reverted to simplify the system for its first stable deployment. The current focus is on consolidating the core RAG functionality. Future integrations will build upon this stable foundation.

## 5. Instructions for Future Sessions
**IMPORTANT:** At the beginning of any future session, please read this `GEMINI_CONTEXT.md` file to quickly recall the project's state and ongoing tasks. This will ensure continuity and efficiency.