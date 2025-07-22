from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Request, Response, Query
import os # Import os for file operations
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import logging
from typing import Optional, List
from sqlalchemy.orm import Session
import uuid # Import uuid for temporary IDs

from app.core.config import settings # <-- AÑADIR ESTA LÍNEA

# Almacenamiento temporal en memoria para PDFs y metadatos antes de la confirmación
temp_pdf_storage = {}

from .services.pipeline import process_pdf_pipeline, extract_metadata_pipeline
from .services.rag_service import perform_rag_query, semantic_search_documents, get_refined_query_suggestions
from .services.query_orchestrator import QueryOrchestrator
from .schemas.document import (
    DocumentResponse, DocumentStatusResponse, QueryRequest, 
    QueryResponse, DocumentCreateRequest, QueryRefinementRequest, 
    QueryRefinementResponse, SortByEnum, SortOrderEnum
)
from .db.crud import create_document, update_document_processing_results, get_document_status, get_documents, get_unique_publishers
from .db.session import get_db, engine
from .models.document import Document # Asegúrate que este archivo ahora existe en app/models/document.py
from .db.base import Base # Apuntamos al nuevo archivo base.py
from .models.qa_cache import QACache

# Configurar logging básico
logging.basicConfig(level=logging.INFO)


Base.metadata.create_all(bind=engine)

# Crear una instancia única del orquestador para ser usada en toda la aplicación
query_orchestrator = QueryOrchestrator()

app = FastAPI(
    title="Sistema RAG de Vigilancia e Inteligencia",
    description="API para procesar documentos y alimentar la base de conocimiento.",
    version="0.1.0"
)

# Montar directorio estático y configurar plantillas
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Lista de bases de conocimiento (namespaces)
KNOWLEDGE_BASES = {
    "default": "Base de Conocimiento Principal", # Mapea al namespace 'default' de Pinecone
}

@app.get("/", response_class=HTMLResponse)
async def select_knowledge_base_page(request: Request):
    """Muestra la página para seleccionar una base de conocimiento."""
    return templates.TemplateResponse("select_kb.html", {"request": request, "kbs": KNOWLEDGE_BASES})

@app.post("/select-kb")
async def select_knowledge_base_action(request: Request, kb_namespace: str = Form(...)):
    """Guarda la base de conocimiento seleccionada en una cookie y redirige."""
    if kb_namespace not in KNOWLEDGE_BASES:
        raise HTTPException(status_code=400, detail="Base de conocimiento no válida.")
    
    response = RedirectResponse(url="/home", status_code=303)
    response.set_cookie(key="selected_kb_namespace", value=kb_namespace, httponly=True)
    return response

@app.get("/home", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página principal de la interfaz de usuario."""
    selected_kb = request.cookies.get("selected_kb_namespace")
    if not selected_kb or selected_kb not in KNOWLEDGE_BASES:
        return RedirectResponse(url="/")

    kb_name = KNOWLEDGE_BASES[selected_kb]
    return templates.TemplateResponse("index.html", {"request": request, "kb_name": kb_name})

@app.post("/upload-document/", response_model=dict, status_code=200)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="El archivo PDF a analizar."),
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")
    
    file_bytes = await file.read()
    
    # Extraer metadatos y obtener la ruta temporal del PDF
    try:
        extracted_data = extract_metadata_pipeline(file_bytes=file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al extraer metadatos: {e}")

    # Generar un ID temporal para este archivo y sus metadatos
    temp_id = str(uuid.uuid4())
    temp_pdf_storage[temp_id] = {
        "pdf_path": extracted_data["pdf_path"],
        "metadata": extracted_data["metadata"],
        "filename": file.filename # Guardar el filename original aquí
    }

    # Devolver los metadatos extraídos y el ID temporal al frontend
    # para que el usuario los valide.
    return {"temp_id": temp_id, "metadata": extracted_data["metadata"]}

@app.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def read_document_status(document_id: int, db: Session = Depends(get_db)):
    """Endpoint para hacer polling y obtener el estado de un documento."""
    status = get_document_status(db=db, document_id=document_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": status}

@app.get("/documents/", response_model=List[DocumentResponse])
def read_documents(
    request: Request,
    skip: int = 0,
    limit: int = 100,
    query: Optional[str] = None,
    sort_by: Optional[SortByEnum] = Query(SortByEnum.id),
    sort_order: Optional[SortOrderEnum] = Query(SortOrderEnum.desc),
    keyword: Optional[str] = Query(None),
    publisher_filter: Optional[str] = Query(None),
    year_start: Optional[int] = Query(None),
    year_end: Optional[int] = Query(None),
    db: Session = Depends(get_db)
) -> List[DocumentResponse]:
    """Recupera una lista de todos los documentos procesados, opcionalmente filtrados y ordenados."""
    selected_kb = request.cookies.get("selected_kb_namespace", "default")
    if query:
        documents = semantic_search_documents(query=query, namespace=selected_kb, db=db)
        return documents
    else:
        documents = get_documents(
            db,
            skip=skip,
            limit=limit,
            sort_by=sort_by.value if sort_by else None,
            sort_order=sort_order.value if sort_order else None,
            keyword=keyword,
            publisher_filter=publisher_filter,
            year_start=year_start,
            year_end=year_end
        )
        return documents

@app.get("/validate-metadata/{temp_id}", response_class=HTMLResponse)
async def validate_metadata_page(temp_id: str, request: Request):
    """Sirve la página para validar y completar los metadatos de un documento usando un ID temporal."""
    # Verificar si el ID temporal es válido
    if temp_id not in temp_pdf_storage:
        raise HTTPException(status_code=404, detail="Documento temporal no encontrado o la sesión ha expirado.")
    
    # Pasamos el temp_id a la plantilla. Los metadatos se cargarán desde sessionStorage en el lado del cliente.
    return templates.TemplateResponse("validate_metadata.html", {"request": request, "temp_id": temp_id})

@app.post("/process-document/{temp_id}", response_model=DocumentResponse, status_code=202)
async def process_document(
    temp_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    document_data: DocumentCreateRequest, # Usamos DocumentCreateRequest para validar los datos recibidos
):
    """Recibe los metadatos validados por el usuario y dispara el procesamiento completo del documento."""
    # Recuperar la ruta del PDF temporal y los metadatos originales
    temp_data = temp_pdf_storage.pop(temp_id, None) # Usar pop para eliminarlo del almacenamiento temporal
    if not temp_data:
        raise HTTPException(status_code=404, detail="Documento temporal no encontrado o ya procesado.")

    pdf_path = temp_data["pdf_path"]
    original_filename = temp_data["filename"]

    # Validar que el archivo temporal aún existe
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="PDF original no encontrado en el almacenamiento temporal.")

    # Crear la entrada en la base de datos con los metadatos validados por el usuario
    db = next(get_db()) # Obtener una sesión de DB
    try:
        update_data = document_data.model_dump(exclude_unset=True)
        update_data["status"] = "processing" # Cambiar estado a processing
        update_data["filename"] = original_filename # Asegurarse de usar el filename original

        db_document = create_document(db=db, document_data=update_data)
        db.commit()
        db.refresh(db_document)

        # Disparar el procesamiento pesado
        selected_kb = request.cookies.get("selected_kb_namespace", "default")
        background_tasks.add_task(process_pdf_pipeline, document_id=db_document.id, pdf_path=pdf_path, namespace=selected_kb)

        return db_document
    finally:
        db.close()

@app.get("/library", response_class=HTMLResponse)
async def read_library(request: Request, query: Optional[str] = None):
    """Sirve la página de la biblioteca de documentos con funcionalidad de búsqueda."""
    return templates.TemplateResponse("library.html", {"request": request, "query": query})

@app.get("/query", response_class=HTMLResponse)
async def read_query_page(request: Request):
    """Sirve la página de consulta RAG."""
    return templates.TemplateResponse("query.html", {"request": request})

@app.get("/publishers/autocomplete", response_model=List[str])
async def get_publishers_autocomplete(
    search_term: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """Obtiene sugerencias de autocompletado para publicadores."""
    publishers = get_unique_publishers(db, search_term=search_term)
    return publishers

@app.post("/query/refine", response_model=QueryRefinementResponse)
async def refine_query(refine_request: QueryRefinementRequest):
    """Recibe una consulta y devuelve sugerencias de refinamiento generadas por un LLM."""
    try:
        suggestions = get_refined_query_suggestions(query=refine_request.query)
        return {"suggestions": suggestions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al refinar la consulta: {e}")

@app.post("/query/", response_model=QueryResponse)
def handle_query(request: Request, query_request: QueryRequest):
    """Maneja una consulta del usuario utilizando el orquestador."""
    try:
        selected_kb = request.cookies.get("selected_kb_namespace", "default")
        result = query_orchestrator.decide_and_execute(query=query_request.query, kb_id=selected_kb)
        
        if isinstance(result, QueryResponse):
            return result
        else:
            # Esto no debería ocurrir si el orquestador siempre devuelve QueryResponse
            logging.error(f"Tipo de respuesta inesperado del orquestador: {type(result)} - {result}")
            raise HTTPException(status_code=500, detail="Error interno: El orquestador devolvió un tipo de respuesta inesperado.")

    except HTTPException as e:
        # Re-lanzar HTTPExceptions directamente
        raise e
    except Exception as e:
        # Capturar cualquier otra excepción inesperada
        logging.error(f"Error inesperado en handle_query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {e}")