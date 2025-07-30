import sys
import os
import asyncio
import logging
from sqlalchemy.orm import Session
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.session import SessionLocal
from app.modules.document_management.document_service import get_abandoned_documents, delete_document

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def cleanup_abandoned_documents():
    """
    Finds and deletes documents that have been in the 'awaiting_validation'
    state for more than 24 hours.
    """
    db: Session = SessionLocal()
    try:
        logging.info("Starting cleanup of abandoned documents...")
        
        # We need a default namespace for deletion, this might need to be configured
        # if you use multiple namespaces dynamically. For now, we assume a default.
        default_namespace = ""

        abandoned_docs = get_abandoned_documents(db, older_than_hours=24)
        
        if not abandoned_docs:
            logging.info("No abandoned documents found.")
            return

        logging.info(f"Found {len(abandoned_docs)} abandoned document(s) to delete.")

        for doc in abandoned_docs:
            logging.info(f"Deleting document ID: {doc.id}, Filename: {doc.filename}")
            try:
                # The delete_document function now requires a namespace
                success = delete_document(db=db, document_id=doc.id, namespace=default_namespace)
                if success:
                    logging.info(f"Successfully deleted document ID: {doc.id} and its associated files/vectors.")
                else:
                    logging.warning(f"Failed to delete document ID: {doc.id}. Check logs for details.")
            except Exception as e:
                logging.error(f"An error occurred while deleting document ID {doc.id}: {e}", exc_info=True)
                # Rollback any partial changes for this specific document if the session is per-document
                db.rollback()

    finally:
        db.close()
        logging.info("Cleanup process finished.")

if __name__ == "__main__":
    asyncio.run(cleanup_abandoned_documents())
