import asyncio
import os
import sys
import logging
from typing import List, Dict, Any

# Añadir el directorio raíz del proyecto al sys.path para importaciones relativas
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config import settings
from app.db.session import SessionLocal, get_db
from app.modules.rag_chat.query_orchestrator import QueryOrchestrator
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.document import QueryRequest, QueryResponse

from ragas import evaluate
from datasets import Dataset

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Inicializar el orquestador de consultas
query_orchestrator = QueryOrchestrator()

# --- Conjunto de Datos de Evaluación de Ejemplo ---
# Este es un conjunto de datos muy pequeño. Para una evaluación significativa,
# necesitarías un conjunto de datos mucho más grande y diverso.
# Cada entrada debe tener: question, ground_truth, y opcionalmente ground_truth_contexts.
# Los ground_truth_contexts son los fragmentos de texto reales que contienen la respuesta.

eval_dataset_examples = [
    {
        "question": "¿Cuál es el propósito principal de la Corporación Ruta N?",
        "ground_truth": "El propósito principal de la Corporación Ruta N es contribuir al mejoramiento de la calidad de vida de los habitantes de Medellín a través del fomento de la innovación y el desarrollo de negocios de base tecnológica.",
        "ground_truth_contexts": [
            "La Corporación Ruta N es una corporación sin ánimo de lucro creada por la Alcaldía de Medellín, UNE y EPM, cuyo propósito principal es contribuir al mejoramiento de la calidad de vida de los habitantes de Medellín a través del fomento de la innovación y el desarrollo de negocios de base tecnológica."
        ]
    },
    {
        "question": "¿Qué tipo de empresas apoya la Corporación Ruta N?",
        "ground_truth": "La Corporación Ruta N apoya empresas de base tecnológica y de innovación.",
        "ground_truth_contexts": [
            "La Corporación Ruta N... contribuye al mejoramiento de la calidad de vida de los habitantes de Medellín a través del fomento de la innovación y el desarrollo de negocios de base tecnológica."
        ]
    },
    {
        "question": "¿Quiénes crearon la Corporación Ruta N?",
        "ground_truth": "La Corporación Ruta N fue creada por la Alcaldía de Medellín, UNE y EPM.",
        "ground_truth_contexts": [
            "La Corporación Ruta N es una corporación sin ánimo de lucro creada por la Alcaldía de Medellín, UNE y EPM."
        ]
    },
    {
        "question": "¿Qué es el GEIAL y cuál es su objetivo?",
        "ground_truth": "El Grupo de Ecosistemas Inteligentes de América Latina (GEIAL) es una iniciativa que busca fomentar la colaboración y el desarrollo de startups en la región.",
        "ground_truth_contexts": [
            "El Grupo de Ecosistemas Inteligentes de América Latina (GEIAL) presenta el informe 2024 que analiza las condiciones sistémicas para el emprendimiento dinámico en diversas ciudades latinoamericanas."
        ]
    },
    {
        "question": "¿Cuál es el enfoque del reporte de KPMG sobre Colombia?",
        "ground_truth": "El reporte de KPMG sobre Colombia se enfoca en el panorama de la tecnología y la innovación en el país, destacando oportunidades y desafíos.",
        "ground_truth_contexts": [
            "El Colombia Tech Report 2024-2025 destaca el crecimiento del ecosistema tecnológico colombiano, con 2.126 startups registradas y un enfoque en tendencias globales, top 10 de startups y talento del futuro."
        ]
    },
    {
        "question": "¿Qué es un sandbox regulatorio según el Decreto 1732 de 2021?",
        "ground_truth": "Un sandbox regulatorio es un ambiente especial de vigilancia y control para promover la innovación, el emprendimiento y la formalización empresarial en industrias reguladas.",
        "ground_truth_contexts": [
            "El Decreto 1732 de 2021 reglamenta el artículo 5 de la Ley 2069 de 2020 en Colombia, estableciendo un mecanismo exploratorio de regulación para modelos de negocio innovadores en industrias reguladas."
        ]
    },
    {
        "question": "¿Cuál es el objetivo de la Política Nacional de Inteligencia Artificial en Colombia?",
        "ground_truth": "El objetivo de la Política Nacional de Inteligencia Artificial en Colombia es desarrollar capacidades para la investigación, desarrollo, adopción y aprovechamiento ético y sostenible de sistemas de IA.",
        "ground_truth_contexts": [
            "El documento CONPES establece la Política Nacional de Inteligencia Artificial en Colombia con el objetivo de desarrollar capacidades para la investigación, desarrollo, adopción y aprovechamiento ético y sostenible de sistemas de IA."
        ]
    }
]

async def run_rag_pipeline_for_evaluation(question: str, kb_id: str) -> Dict[str, Any]:
    """
    Ejecuta el pipeline RAG para una pregunta dada y extrae la respuesta y el contexto.
    """
    db = SessionLocal()
    print(f"DEBUG: run_rag_pipeline_for_evaluation called for question: {question}")
    try:
        query_request = QueryRequest(query=question)
        # Usamos decide_and_execute directamente para obtener la respuesta y las fuentes
        response: QueryResponse = await query_orchestrator.decide_and_execute(query=query_request.query, kb_id=kb_id, db=db)
        
        answer = response.answer
        # Extraer los textos de las fuentes recuperadas
        contexts = response.context_chunks
        
        print(f"DEBUG: RAG pipeline response for '{question}': Answer='{answer[:100]}...', Contexts={len(contexts)} sources")
        
        return {"answer": answer, "contexts": contexts}
    except Exception as e:
        logger.error(f"Error al ejecutar el pipeline RAG para la pregunta \"{question}\": {e}", exc_info=True)
        return {"answer": "Error al generar respuesta.", "contexts": []}
    finally:
        db.close()

async def evaluate_rag_system():
    logger.info("Iniciando evaluación del sistema RAG...")
    print("DEBUG: evaluate_rag_system started.")
    
    # Asumimos que estamos evaluando la base de conocimiento 'default'
    kb_to_evaluate = "default"

    # Preparar los datos para Ragas
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "ground_truth_contexts": []
    }

    for example in eval_dataset_examples:
        print(f"DEBUG: Processing example: {example['question']}")
        rag_output = await run_rag_pipeline_for_evaluation(example["question"], kb_to_evaluate)
        
        data["question"].append(example["question"])
        data["answer"].append(rag_output["answer"])
        data["contexts"].append(rag_output["contexts"])
        data["ground_truth"].append(example["ground_truth"])
        data["ground_truth_contexts"].append(example["ground_truth_contexts"])

    print(f"DEBUG: Data prepared for Ragas: {data}")
    # Crear el Dataset de Ragas
    dataset = Dataset.from_dict(data)
    
    # Definir las métricas a evaluar
    # faithfulness: Mide si la respuesta se basa en el contexto proporcionado.
    # answer_relevance: Mide si la respuesta es relevante para la pregunta.
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
    metrics = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()]

    logger.info("Calculando métricas con Ragas...")
    print("DEBUG: Calling Ragas evaluate...")
    try:
        result = evaluate(dataset, metrics)
        print("DEBUG: Ragas evaluate returned.")
        logger.info("Resultados de la evaluación:")
        logger.info(result)
        logger.info("Métricas por ejemplo:")
        logger.info(result.to_pandas())
    except Exception as e:
        logger.error(f"Error durante la evaluación con Ragas: {e}", exc_info=True)

if __name__ == "__main__":
    # Asegurarse de que las variables de entorno estén cargadas si es necesario
    # from dotenv import load_dotenv
    # load_dotenv()
    asyncio.run(evaluate_rag_system())