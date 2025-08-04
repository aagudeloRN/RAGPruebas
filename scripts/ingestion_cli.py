import argparse
import asyncio
import logging
import os
import sys

# Añadir el directorio raíz del proyecto al sys.path para importaciones relativas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import settings
from app.db.session import SessionLocal, get_db # Importar SessionLocal para uso independiente
from app.modules.data_ingestion.ingestion_service import handle_upload_document, handle_process_document
from app.modules.document_management.document_service import delete_document
from app.schemas.document import DocumentCreateRequest
from pinecone import Pinecone

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar Pinecone (necesario para listar/eliminar documentos)
pc = Pinecone(api_key=settings.PINECONE_API_KEY)

from app.modules.data_ingestion.pipeline import process_document_for_rag
from app.modules.document_management.document_service import delete_document as service_delete_document

async def ingest_document_cli(file_path: str, namespace: str):
    logger.info(f"Iniciando ingesta de documento: {file_path} en namespace: {namespace}")
    
    class MockUploadFile:
        def __init__(self, path):
            self.file = open(path, "rb")
            self.filename = os.path.basename(path)
            self.content_type = "application/pdf"
        
        async def __aenter__(self):
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self.file.close()

    db = SessionLocal()
    try:
        async with MockUploadFile(file_path) as mock_file:
            upload_result = await handle_upload_document(file=mock_file, db=db)
            document_id = upload_result.get("document_id")
            initial_metadata = upload_result.get("metadata", {})
            
            if document_id:
                logger.info(f"Documento subido temporalmente con ID: {document_id}. Recopilando metadatos...")
                
                # Recopilar metadatos del usuario
                title = input(f"Título del documento (default: {initial_metadata.get('title')}): ") or initial_metadata.get('title')
                publisher = input(f"Publicador (default: {initial_metadata.get('publisher')}): ") or initial_metadata.get('publisher')
                publication_year_str = input(f"Año de publicación (default: {initial_metadata.get('publication_year')}): ") or str(initial_metadata.get('publication_year', ''))
                publication_year = int(publication_year_str) if publication_year_str.isdigit() else None
                summary = input(f"Resumen (default: {initial_metadata.get('summary')}): ") or initial_metadata.get('summary')
                keywords_str = input(f"Palabras clave (separadas por coma, default: {initial_metadata.get('keywords')}): ") or ", ".join(initial_metadata.get('keywords', []))
                keywords = [k.strip() for k in keywords_str.split(',')] if keywords_str else []
                source_url = input(f"URL de origen (default: {initial_metadata.get('source_url', f'file://{file_path}')}): ") or initial_metadata.get('source_url', f'file://{file_path}')
                language = input(f"Idioma (default: {initial_metadata.get('language', 'es')}): ") or initial_metadata.get('language', 'es')
                preview_image_url = initial_metadata.get('cover_image_url') # Usar la URL de la imagen de portada generada

                document_data = DocumentCreateRequest(
                    title=title,
                    publisher=publisher,
                    publication_year=publication_year,
                    summary=summary,
                    keywords=keywords,
                    source_url=source_url,
                    language=language,
                    status="processing", # Se actualizará a 'completed' o 'failed' por process_document_for_rag
                    preview_image_url=preview_image_url # Incluir la URL de la imagen de portada
                )
                
                logger.info(f"Iniciando procesamiento RAG para documento ID: {document_id}...")
                temp_file_path = os.path.join(TEMP_FILE_DIR, f"{document_id}.pdf")
                
                await process_document_for_rag(
                    document_id=document_id,
                    namespace=namespace,
                    temp_file_path=temp_file_path,
                    validated_data=document_data.model_dump() # Pasa todos los datos validados
                )
                logger.info(f"Procesamiento RAG para documento ID: {document_id} completado (o fallido). Revise el estado en la DB.")
            else:
                logger.error("Fallo al subir el documento temporalmente.")
    except Exception as e:
        logger.error(f"Error durante la ingesta: {e}", exc_info=True)
        # Intentar actualizar el estado del documento a 'failed' si existe
        if 'document_id' in locals() and document_id:
            from app.db.crud import update_document_processing_results
            update_document_processing_results(db, document_id, {"status": "failed", "processing_error": f"Error CLI: {e}"})
            db.commit()
    finally:
        db.close()

async def list_pinecone_namespaces_cli():
    logger.info("Listando namespaces de Pinecone...")
    try:
        # La API de Pinecone no tiene un método directo para listar todos los namespaces.
        # Una forma de "descubrir" namespaces es iterar sobre índices y ver los que tienen datos.
        # Esto es una limitación de la API de Pinecone.
        # Si tuvieras un control plane que gestiona los namespaces, lo consultarías aquí.
        logger.info("La API de Pinecone no ofrece un método directo para listar todos los namespaces.")
        logger.info(f"El índice configurado es: {settings.PINECONE_INDEX_NAME}")
        logger.info("Puedes listar los contenidos de un namespace específico usando 'list-documents'.")
    except Exception as e:
        logger.error(f"Error al listar namespaces de Pinecone: {e}", exc_info=True)

async def list_pinecone_documents_cli(namespace: str, top_k: int = 100):
    logger.info(f"Listando los primeros {top_k} documentos en el namespace '{namespace}' de Pinecone...")
    try:
        index = pc.Index(settings.PINECONE_INDEX_NAME)
        # Fetch all vectors (or a large number) to get their metadata
        # Esto puede ser costoso para índices muy grandes.
        # Una mejor aproximación sería usar `query` con un vector de ceros o un vector aleatorio
        # para obtener una muestra, o si Pinecone tuviera una API de listado.
        
        # Usamos una consulta con un vector de ceros para obtener una muestra de IDs y metadatos
        # Esto no es una lista exhaustiva, sino una forma de obtener algunos ejemplos.
        # La API de Pinecone no tiene un método "list all vectors".
        dummy_query_vector = [0.0] * 1536 # Dimensión de text-embedding-3-small
        
        response = index.query(
            vector=dummy_query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        if not response.matches:
            logger.info(f"No se encontraron documentos en el namespace '{namespace}'.")
            return
            
        logger.info(f"Documentos encontrados en '{namespace}':")
        for match in response.matches:
            doc_id = match.metadata.get("document_id", "N/A")
            title = match.metadata.get("title", "N/A")
            logger.info(f"  - ID: {match.id}, Document ID: {doc_id}, Título: {title[:70]}..., Score: {match.score:.4f}")
            
    except Exception as e:
        logger.error(f"Error al listar documentos de Pinecone en namespace '{namespace}': {e}", exc_info=True)

async def delete_document_cli(document_id: int, namespace: str):
    logger.info(f"Eliminando documento ID {document_id} de la base de datos y Pinecone en namespace: {namespace}...")
    db = SessionLocal()
    try:
        success = service_delete_document(db=db, document_id=document_id, namespace=namespace)
        if success:
            logger.info(f"Documento ID {document_id} eliminado exitosamente.")
        else:
            logger.warning(f"No se pudo eliminar el documento ID {document_id}. Puede que no exista o haya ocurrido un error.")
    except Exception as e:
        logger.error(f"Error al eliminar documento ID {document_id}: {e}", exc_info=True)
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description="Herramienta CLI para la ingesta y gestión de documentos RAG.")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles")

    # Comando para ingestar un documento
    ingest_parser = subparsers.add_parser("ingest", help="Ingesta un documento PDF.")
    ingest_parser.add_argument("file_path", type=str, help="Ruta al archivo PDF a ingestar.")
    ingest_parser.add_argument("--namespace", type=str, default="default", help="Namespace de Pinecone donde ingestar el documento.")

    # Comando para listar namespaces de Pinecone
    list_namespaces_parser = subparsers.add_parser("list-namespaces", help="Lista los namespaces de Pinecone (limitado por la API de Pinecone).")

    # Comando para listar documentos en un namespace de Pinecone
    list_documents_parser = subparsers.add_parser("list-documents", help="Lista documentos en un namespace específico de Pinecone.")
    list_documents_parser.add_argument("namespace", type=str, help="Namespace de Pinecone a listar.")
    list_documents_parser.add_argument("--top_k", type=int, default=100, help="Número máximo de documentos a listar.")

    # Comando para eliminar un documento
    delete_parser = subparsers.add_parser("delete-document", help="Elimina un documento por su ID de la DB y Pinecone.")
    delete_parser.add_argument("document_id", type=int, help="ID del documento a eliminar.")
    delete_parser.add_argument("--namespace", type=str, default="default", help="Namespace de Pinecone del documento a eliminar.")

    args = parser.parse_args()

    if args.command == "ingest":
        asyncio.run(ingest_document_cli(args.file_path, args.namespace))
    elif args.command == "list-namespaces":
        asyncio.run(list_pinecone_namespaces_cli())
    elif args.command == "list-documents":
        asyncio.run(list_pinecone_documents_cli(args.namespace, args.top_k))
    elif args.command == "delete-document":
        asyncio.run(delete_document_cli(args.document_id, args.namespace))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
