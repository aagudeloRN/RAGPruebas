import logging
from typing import List, Optional

from fastapi import (BackgroundTasks, Depends, FastAPI, File, Form,
                     HTTPException, Query, Request, Response, UploadFile)
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

# Importaciones para la inicialización de la BD
from app.db.base import Base
from app.db.session import engine

# Importar modelos para que SQLAlchemy los reconozca
from app.models import document, qa_cache

from app.core.config import settings
from app.db.crud import create_document
from app.db.crud_qa_cache import get_top_qa_cache
from app.db.session import engine, get_db
from app.modules.data_ingestion.ingestion_service import (
    handle_process_document, handle_upload_document)
from app.modules.document_management.document_service import (
    delete_document, get_document, get_document_status, get_documents,
    get_unique_publishers)
from app.modules.rag_chat.query_orchestrator import QueryOrchestrator
from app.modules.rag_chat.rag_service import (
    get_refined_query_suggestions, perform_rag_query,
    semantic_search_documents)
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.document import (DocumentCreateRequest, DocumentResponse,
                                  DocumentStatusResponse, QueryRequest,
                                  QueryResponse, QueryRefinementRequest,
                                  QueryRefinementResponse, SortByEnum,
                                  SortOrderEnum)
from app.schemas.qa_cache import QACache as QACacheSchema

# Configurar logging básico
logging.basicConfig(level=logging.INFO)

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
    "default": "Base de Conocimiento Principal",
}

@app.on_event("startup")
def on_startup():
    # Este evento asegura que las tablas se creen al iniciar la aplicación.
    Base.metadata.create_all(bind=engine)

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
async def upload_document_endpoint(
    file: UploadFile = File(..., description="El archivo PDF a analizar."),
    db: Session = Depends(get_db)
):
    return await handle_upload_document(file=file, db=db)

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

@app.get("/validate-metadata/{document_id}", response_class=HTMLResponse)
async def validate_metadata_page(document_id: int, request: Request, db: Session = Depends(get_db)):
    """Sirve la página para validar y completar los metadatos de un documento."""
    document = get_document(db, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="Documento no encontrado")
    return templates.TemplateResponse("validate_metadata.html", {"request": request, "document": document})

@app.post("/process-document/{document_id}", response_model=DocumentResponse, status_code=202)
async def process_document(document_id: int, request: Request, background_tasks: BackgroundTasks, document_data: DocumentCreateRequest, db: Session = Depends(get_db)):
    selected_kb = request.cookies.get("selected_kb_namespace", "default")
    return await handle_process_document(
        document_id=document_id,
        background_tasks=background_tasks,
        document_data=document_data,
        db=db,
        selected_kb_namespace=selected_kb
    )

@app.delete("/documents/{document_id}", status_code=204)
def delete_document(document_id: int, request: Request, db: Session = Depends(get_db)):
    """Endpoint para eliminar un documento y sus recursos asociados."""
    selected_kb = request.cookies.get("selected_kb_namespace", "default")
    success = delete_document(db=db, document_id=document_id, namespace=selected_kb)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or could not be deleted.")
    return Response(status_code=204)

@app.get("/library", response_class=HTMLResponse)
async def read_library(request: Request, query: Optional[str] = None):
    """Sirve la página de la biblioteca de documentos con funcionalidad de búsqueda."""
    return templates.TemplateResponse("library.html", {"request": request, "query": query})

@app.get("/upload", response_class=HTMLResponse)
async def upload_form_page(request: Request):
    """Sirve la página con el formulario para subir documentos."""
    return templates.TemplateResponse("upload_form.html", {"request": request})

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
def handle_query(request: Request, query_request: QueryRequest, db: Session = Depends(get_db)):
    """Maneja una consulta del usuario utilizando el orquestador."""
    try:
        selected_kb = request.cookies.get("selected_kb_namespace", "default")
        result = query_orchestrator.decide_and_execute(query=query_request.query, kb_id=selected_kb, db=db)
        
        if isinstance(result, QueryResponse):
            return result
        else:
            # Esto no debería ocurrir si el orquestador siempre devuelve QueryResponse
            logging.error(f"Tipo de respuesta inesperado del orquestador: {type(result)} - {result}")
            raise HTTPException(status_code=500, detail="Error interno: El orquestador devolvió un tipo de respuesta inesperado.")

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error inesperado en handle_query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {e}")

@app.post("/chat/", response_model=ChatResponse)
async def handle_chat(request: Request, chat_request: ChatRequest, db: Session = Depends(get_db)):
    """Maneja una consulta conversacional utilizando el orquestador."""
    try:
        selected_kb = request.cookies.get("selected_kb_namespace", "default")
        
        result = await query_orchestrator.handle_conversational_query(
            chat_request=chat_request, 
            kb_id=selected_kb,
            db=db
        )
        
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        logging.error(f"Error inesperado en handle_chat: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta de chat: {e}")

@app.get("/faq")
async def read_faq_page(request: Request):
    """Sirve la página de preguntas frecuentes."""
    return templates.TemplateResponse("faq.html", {"request": request})

@app.get("/faq/top5", response_model=List[QACacheSchema])
async def get_top_qa(db: Session = Depends(get_db)):
    """Recupera las 5 preguntas y respuestas más frecuentes de la caché."""
    top_qas = get_top_qa_cache(db)
    return top_qas
