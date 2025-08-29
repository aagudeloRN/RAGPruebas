
# app/modules/document_processing/table_extractor.py
import fitz  # PyMuPDF
import logging
from typing import List
import pandas as pd

logger = logging.getLogger(__name__)

def extract_and_format_tables(doc_path: str) -> List[str]:
    """
    Extrae todas las tablas de un documento PDF y las formatea como strings de Markdown.
    """
    try:
        doc = fitz.open(doc_path)
    except Exception as e:
        logger.error(f"Error al abrir el documento {doc_path} para extraer tablas: {e}")
        return []

    table_chunks = []
    for page_num, page in enumerate(doc):
        tables = page.find_tables()
        if not tables.tables:
            continue

        logger.info(f"Se encontraron {len(tables.tables)} tablas en la página {page_num + 1}")

        for i, table in enumerate(tables):
            df = table.to_pandas()
            if df.empty:
                continue
            markdown_table = df.to_markdown(index=False, tablefmt="grid")
            table_title = f"Tabla {i+1} de la página {page_num + 1}"
            table_chunk = f"### {table_title}\n\n{markdown_table}"
            table_chunks.append(table_chunk)

    logger.info(f"Se extrajeron y formatearon {len(table_chunks)} tablas en total.")
    return table_chunks
