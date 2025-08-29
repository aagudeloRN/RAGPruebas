
# app/modules/document_processing/text_extractor.py
import fitz  # PyMuPDF
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def extract_and_sort_blocks(doc_path: str) -> List[Dict[str, Any]]:
    # ... (código de la función)
    try:
        doc = fitz.open(doc_path)
    except Exception as e:
        logger.error(f"Error al abrir el documento {doc_path}: {e}")
        return []

    all_blocks = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if b['type'] == 0:
                for line in b["lines"]:
                    for span in line["spans"]:
                        all_blocks.append({
                            "page": page_num,
                            "bbox": span["bbox"],
                            "text": span["text"].strip()
                        })
    
    sorted_blocks = sorted(all_blocks, key=lambda b: (b["page"], b["bbox"][1], b["bbox"][0]))
    logger.info(f"Se extrajeron y ordenaron {len(sorted_blocks)} bloques de texto de {doc_path}")
    return sorted_blocks

def group_blocks_into_chunks(sorted_blocks: List[Dict[str, Any]], tolerance: int = 10) -> List[str]:
    # ... (código de la función)
    if not sorted_blocks:
        return []

    chunks = []
    current_chunk = ""
    last_bbox = sorted_blocks[0]["bbox"]
    last_page = sorted_blocks[0]["page"]

    for block in sorted_blocks:
        current_page = block["page"]
        current_bbox = block["bbox"]
        text = block["text"]

        if not text:
            continue

        vertical_distance = current_bbox[1] - last_bbox[3]

        if current_page != last_page or vertical_distance > tolerance:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = text
        else:
            if not current_chunk.endswith(' '):
                current_chunk += " "
            current_chunk += text

        last_bbox = current_bbox
        last_page = current_page

    if current_chunk:
        chunks.append(current_chunk)

    logger.info(f"Se agruparon los bloques en {len(chunks)} chunks semánticos.")
    return chunks

def process_text_document(doc_path: str) -> List[str]:
    """Función principal que orquesta la extracción y agrupación de texto."""
    sorted_blocks = extract_and_sort_blocks(doc_path)
    text_chunks = group_blocks_into_chunks(sorted_blocks)
    return text_chunks
