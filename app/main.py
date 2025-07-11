from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
from sqlalchemy.orm import Session

from app.services.pipeline import process_pdf_pipeline
from app.services.rag_service import perform_rag_query
from app.schemas.document import DocumentResponse, LanguageEnum, DocumentStatusResponse, QueryRequest, QueryResponse
from app.db.crud import create_document, update_document_processing_results, get_document_status, get_documents
from app.db.session import get_db, engine
from app.models.document import Document
from app.db.base_class import Base


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sistema RAG de Vigilancia e Inteligencia",
    description="API para procesar documentos y alimentar la base de conocimiento.",
    version="0.1.0"
)

# Montar directorio estático y configurar plantillas
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sirve la página principal de la interfaz de usuario."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload-document/", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="El archivo PDF a analizar."),
    title: Optional[str] = Form(None, description="Título del documento."),
    source_url: Optional[str] = Form(None, description="URL de origen del documento."),
    publisher: Optional[str] = Form(None, description="Entidad que publica el informe."),
    publication_year: Optional[int] = Form(None, description="Año de publicación."),
    language: LanguageEnum = Form(LanguageEnum.spanish, description="Idioma del documento."),
    db: Session = Depends(get_db)
):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="El archivo debe ser un PDF.")
    
    file_bytes = await file.read()
    
    metadata = {
        "filename": file.filename, 
        "title": title,
        "source_url": source_url, 
        "publisher": publisher, 
        "publication_year": publication_year, 
        "language": language.value # Usamos .value para obtener el string "es", "en", etc.
    }
    
    # 1. Crear la entrada inicial en la base de datos
    db_document = create_document(db=db, document_data=metadata)
    
    # 2. Añadir la tarea de procesamiento pesado al fondo
    background_tasks.add_task(process_pdf_pipeline, file_bytes=file_bytes, document_id=db_document.id, user_metadata=metadata)

    # 3. Devolver la respuesta inmediatamente
    return db_document

@app.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def read_document_status(document_id: int, db: Session = Depends(get_db)):
    """Endpoint para hacer polling y obtener el estado de un documento."""
    status = get_document_status(db=db, document_id=document_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": status}

@app.get("/documents/", response_model=List[DocumentResponse])
def read_documents(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Recupera una lista de todos los documentos procesados."""
    documents = get_documents(db, skip=skip, limit=limit)
    return documents

@app.get("/library", response_class=HTMLResponse)
async def read_library(request: Request):
    """Sirve la página de la biblioteca de documentos."""
    return templates.TemplateResponse("library.html", {"request": request})

@app.get("/query", response_class=HTMLResponse)
async def read_query_page(request: Request):
    """Sirve la página de consulta RAG."""
    return templates.TemplateResponse("query.html", {"request": request})

@app.post("/query/", response_model=QueryResponse)
def handle_query(query_request: QueryRequest):
    """Maneja una consulta RAG del usuario."""
    try:
        return perform_rag_query(query=query_request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar la consulta: {e}")