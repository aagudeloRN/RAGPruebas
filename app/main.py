# app/main.py
import logging
from typing import List, Optional

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form,
    HTTPException, Query, Request, Response, UploadFile
)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

# --- Core Imports ---
from app.core.config import settings, active_kb, load_active_kb_config
from app.db.session import get_db, get_pgvector_db, SessionLocal
from app.db.base import Base
from app.db.session import engine

# --- Service and CRUD Imports ---
from app.modules.rag_chat.query_orchestrator import QueryOrchestrator
from app.modules.document_management.document_service import (
    delete_document as service_delete_document,
    check_pinecone_vectors_exist
)
from app.modules.data_ingestion.ingestion_service import (
    handle_process_document, handle_upload_document
)
from app.db.crud import (
    create_document, get_document, get_document_status,
    get_documents, get_unique_publishers
)
from app.db.crud_qa_cache import get_top_qa_cache

# --- Schema Imports ---
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.document import (
    DocumentCreateRequest, DocumentResponse, DocumentStatusResponse, QueryRequest,
    QueryResponse, SortByEnum, SortOrderEnum
)
from app.schemas.qa_cache import QACache as QACacheSchema

import asyncio

# --- App Initialization ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

query_orchestrator = QueryOrchestrator()

app = FastAPI(
    title="RAG Factory - Sistema de Vigilancia e Inteligencia",
    description="API para gestionar y consultar múltiples bases de conocimiento.",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# --- Startup and Cleanup Events ---

async def periodic_cleanup():
    """Tarea periódica para limpiar documentos antiguos en estado 'awaiting_validation'."""
    while True:
        await asyncio.sleep(3600) # Ejecutar cada hora
        logger.info("Ejecutando limpieza periódica de documentos...")
        db = SessionLocal()
        try:
            # Esta función ahora necesitará el kb_id para operar correctamente
            # Se necesitará una refactorización de la lógica de limpieza
            # delete_old_awaiting_validation_documents(db, hours_old=1)
            logger.warning("La lógica de limpieza periódica necesita ser adaptada para multi-tenant.")
        finally:
            db.close()

@app.on_event("startup")
def on_startup():
    """Evento de inicio: crea tablas de BD y carga la configuración de la KB activa."""
    logger.info("Creando tablas de la base de datos si no existen...")
    Base.metadata.create_all(bind=engine)
    
    logger.info("Cargando configuración de la Base de Conocimiento activa...")
    try:
        load_active_kb_config()
    except Exception as e:
        logger.error(f"FATAL: No se pudo cargar la configuración de la KB. La aplicación no puede iniciar. Error: {e}")
        # En un escenario real, podrías querer que la app no inicie si la config falla.
        # Por ahora, solo lo logueamos.

    # Iniciar tareas en segundo plano
    # asyncio.create_task(periodic_cleanup())

# --- Endpoints de la Interfaz de Usuario Principal ---

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página principal de la interfaz de usuario."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "kb_name": active_kb.name or "N/A",
        "kb_description": active_kb.description or "No se ha cargado ninguna Base de Conocimiento."
    })

@app.get("/library", response_class=HTMLResponse)
async def read_library(request: Request, query: Optional[str] = None):
    return templates.TemplateResponse("library.html", {"request": request, "query": query})

@app.get("/upload", response_class=HTMLResponse)
async def upload_form_page(request: Request):
    return templates.TemplateResponse("upload_form.html", {"request": request})

@app.get("/query", response_class=HTMLResponse)
async def read_query_page(request: Request):
    return templates.TemplateResponse("query.html", {"request": request})

@app.get("/faq")
async def read_faq_page(request: Request):
    return templates.TemplateResponse("faq.html", {"request": request})

# --- Endpoints de la API de Chat y Documentos ---

@app.post("/chat/", response_model=ChatResponse)
async def handle_chat(chat_request: ChatRequest, db: Session = Depends(get_db), pgvector_db: Session = Depends(get_pgvector_db)):
    if not active_kb.id:
        raise HTTPException(status_code=503, detail="Sistema no configurado. No hay una Base de Conocimiento activa.")
    try:
        return await query_orchestrator.handle_conversational_query(
            chat_request=chat_request,
            kb_id=active_kb.id,
            db=db,
            pgvector_db=pgvector_db
        )
    except Exception as e:
        logger.error(f"Error en handle_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/", response_model=QueryResponse)
async def handle_query(query_request: QueryRequest, db: Session = Depends(get_db)):
    if not active_kb.id:
        raise HTTPException(status_code=503, detail="Sistema no configurado. No hay una Base de Conocimiento activa.")
    try:
        return await query_orchestrator.decide_and_execute(
            query=query_request.query, 
            kb_id=active_kb.id, 
            db=db
        )
    except Exception as e:
        logger.error(f"Error en handle_query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/", response_model=List[DocumentResponse])
def read_documents(
    skip: int = 0, limit: int = 100,
    sort_by: Optional[SortByEnum] = Query(SortByEnum.id),
    sort_order: Optional[SortOrderEnum] = Query(SortOrderEnum.desc),
    keyword: Optional[str] = Query(None),
    publisher_filter: Optional[str] = Query(None),
    year_start: Optional[int] = Query(None),
    year_end: Optional[int] = Query(None),
    db: Session = Depends(get_db)
) -> List[DocumentResponse]:
    if not active_kb.id:
        raise HTTPException(status_code=503, detail="Sistema no configurado. No hay una Base de Conocimiento activa.")
    
    documents = get_documents(
        db, kb_id=active_kb.id, skip=skip, limit=limit,
        sort_by=sort_by.value, sort_order=sort_order.value,
        keyword=keyword, publisher_filter=publisher_filter,
        year_start=year_start, year_end=year_end
    )
    
    # La comprobación de vectores de Pinecone se hará en el frontend o se simplificará
    return [DocumentResponse.from_orm(doc) for doc in documents]

# --- Endpoints de Administración (Nuevos) ---

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Página principal del panel de administración."""
    return templates.TemplateResponse("admin/dashboard.html", {"request": request, "active_kb": active_kb})

@app.get("/admin/kbs", response_class=HTMLResponse)
async def manage_kbs_page(request: Request, db: Session = Depends(get_db)):
    """Página para gestionar todas las Bases de Conocimiento."""
    from app.db.crud_kb import get_all_kbs
    all_kbs = get_all_kbs(db)
    return templates.TemplateResponse("admin/kbs.html", {"request": request, "kbs": all_kbs, "active_kb_id": active_kb.id})

@app.post("/admin/kbs/create")
async def create_kb_endpoint(id: str = Form(...), name: str = Form(...), description: str = Form(None), db: Session = Depends(get_db)):
    """Endpoint para crear una nueva Base de Conocimiento."""
    from app.db.crud_kb import get_kb, create_kb
    from app.schemas.knowledge_base import KnowledgeBaseCreate
    
    if get_kb(db, id):
        raise HTTPException(status_code=400, detail=f"La Base de Conocimiento con ID '{id}' ya existe.")
    
    kb_create = KnowledgeBaseCreate(id=id, name=name, description=description)
    create_kb(db, kb_create)
    
    return RedirectResponse(url="/admin/kbs", status_code=303)

@app.post("/admin/kbs/activate")
async def activate_kb_endpoint(kb_id: str = Form(...), db: Session = Depends(get_db)):
    """Endpoint para activar una Base de Conocimiento."""
    from app.db.crud_kb import set_active_kb
    
    set_active_kb(db, kb_id)
    logger.info(f"Se ha activado la KB '{kb_id}'. Es necesario reiniciar la aplicación para que los cambios surtan efecto.")
    
    # Idealmente, aquí mostraríamos un mensaje al usuario.
    # Por ahora, simplemente redirigimos.
    return RedirectResponse(url="/admin/kbs", status_code=303)

# --- Endpoints de Ingesta (Refactorizados) ---

@app.post("/upload-document/", response_model=dict)
async def upload_document_endpoint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not active_kb.id:
        raise HTTPException(status_code=503, detail="Sistema no configurado. No hay una Base de Conocimiento activa.")
    return await handle_upload_document(file=file, db=db, kb_id=active_kb.id)

@app.post("/process-document/{document_id}", response_model=DocumentResponse, status_code=202)
async def process_document(
    document_id: int, background_tasks: BackgroundTasks,
    document_data: DocumentCreateRequest, db: Session = Depends(get_db)
):
    if not active_kb.pinecone_namespace:
        raise HTTPException(status_code=503, detail="Sistema no configurado. No hay un namespace de Pinecone activo.")
    return await handle_process_document(
        document_id=document_id, background_tasks=background_tasks,
        document_data=document_data, db=db,
        pinecone_namespace=active_kb.pinecone_namespace
    )

@app.delete("/documents/{document_id}", status_code=204)
def delete_document_endpoint(document_id: int, db: Session = Depends(get_db)):
    if not active_kb.pinecone_namespace:
        raise HTTPException(status_code=503, detail="Sistema no configurado. No hay un namespace de Pinecone activo.")
    
    success = service_delete_document(
        db=db, document_id=document_id, 
        namespace=active_kb.pinecone_namespace
    )
    if not success:
        raise HTTPException(status_code=404, detail="Documento no encontrado o no se pudo eliminar.")
    return Response(status_code=204)

# --- Otros Endpoints ---

@app.get("/faq/top5", response_model=List[QACacheSchema])
async def get_top_qa(db: Session = Depends(get_pgvector_db)):
    if not active_kb.id:
        return [] # Devolver lista vacía si no hay KB activa
    return get_top_qa_cache(db, kb_id=active_kb.id)