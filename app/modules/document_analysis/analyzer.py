
import fitz  # PyMuPDF
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def analyze_pdf_structure(doc_path: str, analysis_pages: int = 5) -> Dict[str, Any]:
    """
    Realiza un análisis estructural rápido de un PDF para detectar complejidad.

    Args:
        doc_path: La ruta al archivo PDF.
        analysis_pages: El número de páginas a analizar para tomar una decisión.

    Returns:
        Un diccionario con los resultados del análisis y una recomendación.
    """
    results = {
        "page_count": 0,
        "has_tables": False,
        "has_images": False,
        "recommendation": "V1"
    }

    try:
        doc = fitz.open(doc_path)
        results["page_count"] = doc.page_count

        # Analizar solo un subconjunto de páginas para ser eficientes
        pages_to_check = min(doc.page_count, analysis_pages)

        for i in range(pages_to_check):
            page = doc.load_page(i)
            
            # Comprobar si hay tablas
            if page.find_tables():
                logger.info(f"Tabla detectada en la página {i+1}")
                results["has_tables"] = True
            
            # Comprobar si hay imágenes
            if page.get_images():
                logger.info(f"Imagen detectada en la página {i+1}")
                results["has_images"] = True
            
            # Si ya hemos encontrado ambos, no es necesario seguir buscando
            if results["has_tables"] and results["has_images"]:
                break
        
        # Generar recomendación
        if results["has_tables"] or results["has_images"]:
            results["recommendation"] = "V2"
            
    except Exception as e:
        logger.error(f"Error durante el análisis estructural de {doc_path}: {e}", exc_info=True)
        # En caso de error, se recomienda V1 como opción segura
        results["recommendation"] = "V1"

    logger.info(f"Análisis estructural para {doc_path}: {results}")
    return results
