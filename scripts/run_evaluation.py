
import asyncio
import json
import logging
from typing import List, Dict, Any

import httpx

# --- Configuración ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_ENDPOINT = "http://127.0.0.1:8000/chat/"
GOLDEN_DATASET_PATH = "evaluation_dataset.jsonl"
RESULTS_OUTPUT_PATH = "evaluation_results_topk20.jsonl"

# --- Funciones Principales ---

def load_golden_dataset(path: str) -> List[Dict[str, Any]]:
    """Carga el dataset de evaluación desde un archivo JSONL."""
    logging.info(f"Cargando Golden Dataset desde '{path}'...")
    dataset = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line))
        logging.info(f"Se cargaron {len(dataset)} preguntas de evaluación.")
        return dataset
    except FileNotFoundError:
        logging.error(f"Error: No se encontró el archivo del dataset en '{path}'. Abortando.")
        exit(1)
    except json.JSONDecodeError as e:
        logging.error(f"Error al decodificar JSON en '{path}': {e}. Abortando.")
        exit(1)

async def run_single_evaluation(client: httpx.AsyncClient, question: str) -> Dict[str, Any]:
    """Envía una única pregunta al endpoint del chat y devuelve la respuesta."""
    payload = {
        "query": question,
        "history": []  # Enviamos un historial vacío para una pregunta nueva
    }
    try:
        response = await client.post(API_ENDPOINT, json=payload, timeout=120.0) # Timeout de 2 minutos
        response.raise_for_status() # Lanza una excepción para respuestas 4xx/5xx
        return response.json()
    except httpx.HTTPStatusError as e:
        logging.error(f"Error de estado HTTP para la pregunta '{question[:50]}...': {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logging.error(f"Error de conexión para la pregunta '{question[:50]}...': {e}")
    return {"answer": "ERROR_EN_EVALUACION", "sources": []}

async def main():
    """Función principal para orquestar el proceso de evaluación."""
    golden_dataset = load_golden_dataset(GOLDEN_DATASET_PATH)
    
    if not golden_dataset:
        return

    logging.info("Iniciando la ejecución de la evaluación...")
    results = []

    async with httpx.AsyncClient() as client:
        for i, item in enumerate(golden_dataset):
            question = item["question"]
            logging.info(f"Procesando pregunta {i+1}/{len(golden_dataset)}: '{question[:70]}...'")
            
            generated_response = await run_single_evaluation(client, question)
            
            result_entry = {
                "question": question,
                "ideal_answer": item["ideal_answer"],
                "ideal_context_sources": item["ideal_context_sources"],
                "generated_answer": generated_response.get("answer", "SIN_RESPUESTA"),
                "retrieved_sources": generated_response.get("sources", [])
            }
            results.append(result_entry)

    # Guardar los resultados en un archivo
    logging.info(f"Guardando {len(results)} resultados en '{RESULTS_OUTPUT_PATH}'...")
    with open(RESULTS_OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    logging.info("¡Evaluación completada!")
    logging.info(f"Los resultados se han guardado en '{RESULTS_OUTPUT_PATH}'.")
    logging.info("Próximo paso: Revisar manualmente el archivo de resultados y puntuar cada entrada.")

if __name__ == "__main__":
    asyncio.run(main())
