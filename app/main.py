# app/main.py
import logging
from typing import List, Optional

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form,
    HTTPException, Query, Request, Response, UploadFile
)
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
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
from app.modules.data_ingestion_v2.ingestion_service_v2 import handle_ingestion_v2
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

from starlette.middleware.sessions import SessionMiddleware

# --- App Initialization ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

query_orchestrator = QueryOrchestrator()

app = FastAPI(
    title="RAG Factory - Sistema de Vigilancia e Inteligencia",
    description="API para gestionar y consultar múltiples bases de conocimiento.",
    version="1.0.0"
)

# Añadir middleware de sesión para mensajes flash
# ¡IMPORTANTE! La SECRET_KEY debe ser una cadena aleatoria y segura en producción.
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

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

# --- Endpoints de Validación de Metadatos (Nuevos) ---

@app.get("/validate-metadata/{document_id}", response_class=HTMLResponse)
async def validate_metadata_page(request: Request, document_id: int, db: Session = Depends(get_db)):
    """Muestra la página para que el usuario valide y complete los metadatos."""
    doc = get_document(db, document_id=document_id, kb_id=active_kb.id)
    if not doc:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return templates.TemplateResponse("validate_metadata.html", {"request": request, "document": doc, "recommended_pipeline": doc.recommended_pipeline})

@app.post("/save-metadata/{document_id}")
async def save_metadata_endpoint(document_id: int, background_tasks: BackgroundTasks, request: Request, db: Session = Depends(get_db)):
    """Guarda los metadatos validados y lanza el procesamiento en segundo plano."""
    form_data = await request.form()
    update_data = DocumentCreateRequest(**form_data)
    
    if not active_kb.pinecone_namespace:
        raise HTTPException(status_code=503, detail="Sistema no configurado. No hay un namespace de Pinecone activo.")

    await handle_process_document(
        document_id=document_id, 
        background_tasks=background_tasks,
        document_data=update_data, 
        db=db,
        pinecone_namespace=active_kb.pinecone_namespace
    )
    
    return RedirectResponse(url=f"/library?query={update_data.title}", status_code=303)


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

@app.get("/chat/stream")
async def handle_chat_stream(query: str, db: Session = Depends(get_db), pgvector_db: Session = Depends(get_pgvector_db)):
    if not active_kb.id:
        async def error_generator():
            yield f"data: {json.dumps({'type': 'error', 'message': 'Sistema no configurado. No hay KB activa.'})}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream")

    # La lógica de descomposición se centrará en la consulta actual.
    return StreamingResponse(
        query_orchestrator.stream_simple_query(
            query=query,
            kb_id=active_kb.id,
            db=db,
            pgvector_db=pgvector_db
        ),
        media_type="text/event-stream"
    )

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
    
    response_docs = []
    for doc in documents:
        doc_response = DocumentResponse.from_orm(doc)
        doc_response.has_pinecone_vectors = check_pinecone_vectors_exist(doc.id, active_kb.pinecone_namespace)
        response_docs.append(doc_response)
    
    return response_docs

@app.get("/publishers/", response_model=List[str])
def read_publishers(db: Session = Depends(get_db)):
    if not active_kb.id:
        return []
    return get_unique_publishers(db, kb_id=active_kb.id)


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
async def activate_kb_endpoint(request: Request, kb_id: str = Form(...), db: Session = Depends(get_db)):
    """Endpoint para activar una Base de Conocimiento."""
    from app.db.crud_kb import set_active_kb
    
    set_active_kb(db, kb_id)
    logger.info(f"Se ha activado la KB '{kb_id}'. Es necesario reiniciar la aplicación para que los cambios surtan efecto.")
    
    request.session["message"] = f"KB '{kb_id}' activada. Reinicia la aplicación para aplicar los cambios."
    return RedirectResponse(url="/admin/kbs", status_code=303)

# --- Endpoints de Ingesta (Refactorizados) ---

@app.post("/upload-document/", response_model=dict)
async def upload_document_endpoint(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not active_kb.id:
        raise HTTPException(status_code=503, detail="Sistema no configurado. No hay una Base de Conocimiento activa.")
    response_data = await handle_upload_document(file=file, db=db, kb_id=active_kb.id)
    document_id = response_data["document_id"]
    pipeline_rec = response_data["recommended_pipeline"]
    return RedirectResponse(url=f"/validate-metadata/{document_id}?pipeline_rec={pipeline_rec}", status_code=303)

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

@app.post("/process-document-v2/{document_id}", response_model=DocumentResponse, status_code=202)
async def process_document_v2(
    document_id: int, background_tasks: BackgroundTasks,
    document_data: DocumentCreateRequest, db: Session = Depends(get_db)
):
    if not active_kb.pinecone_namespace:
        raise HTTPException(status_code=503, detail="Sistema no configurado. No hay un namespace de Pinecone activo.")
    return await handle_ingestion_v2(
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

# --- Endpoints de Servicios de Procesamiento de Documentos ---

@app.get("/documents/{document_id}/extract-tables", response_model=List[str])
def get_document_tables(document_id: int, db: Session = Depends(get_db)):
    """Extrae todas las tablas de un documento y las devuelve como una lista de strings en formato Markdown."""
    doc = get_document(db, document_id=document_id, kb_id=active_kb.id)
    if not doc or not doc.file_path:
        raise HTTPException(status_code=404, detail="Documento no encontrado o sin ruta de archivo.")
    
    try:
        tables_as_markdown = extract_and_format_tables(doc.file_path)
        return tables_as_markdown
    except Exception as e:
        logger.error(f"Error al extraer tablas del documento {document_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"No se pudieron extraer las tablas: {e}")
