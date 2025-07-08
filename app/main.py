from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from sqlalchemy.orm import Session

from app.services.pipeline import process_pdf_pipeline
from app.schemas.document import DocumentResponse, LanguageEnum
from app.db.crud import create_document, update_document_processing_results
from app.db.session import get_db, engine
from app.models.document import Document
from app.db.base_class import Base


Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Sistema RAG de Vigilancia e Inteligencia",
    description="API para procesar documentos y alimentar la base de conocimiento.",
    version="0.1.0"
)

@app.post("/upload-document/", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(..., description="El archivo PDF a analizar."),
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
        "source_url": source_url, 
        "publisher": publisher, 
        "publication_year": publication_year, 
        "language": language.value # Usamos .value para obtener el string "es", "en", etc.
    }
    
    # 1. Crear la entrada inicial en la base de datos
    db_document = create_document(db=db, document_data=metadata)
    
    # 2. Procesar el PDF y obtener el resultado
    result = process_pdf_pipeline(file_bytes=file_bytes, document_id=db_document.id)
    
    # 3. Actualizar el registro en la DB con los resultados
    updated_document = update_document_processing_results(db=db, document_id=db_document.id, results=result)

    return updated_document
