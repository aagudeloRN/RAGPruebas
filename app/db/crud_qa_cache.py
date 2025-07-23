from sqlalchemy.orm import Session
from app.models.qa_cache import QACache
from app.schemas.qa_cache import QACacheCreate, QACache as QACacheSchema
import openai
import json
from app.core.config import settings
import math
from typing import List
from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

client_openai = AsyncOpenAI()

async def get_embedding(text: str) -> List[float]:
    logger.info(f"Generando embedding para texto: '{text[:50]}...'")
    response = await client_openai.embeddings.create(
        input=text,
        model=settings.OPENAI_EMBEDDING_MODEL
    )
    embedding = response.data[0].embedding
    logger.info(f"Embedding generado (primeros 5 valores): {embedding[:5]}")
    return embedding

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(v1**2 for v1 in vec1))
    magnitude2 = math.sqrt(sum(v2**2 for v2 in vec2))
    if not magnitude1 or not magnitude2:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)

async def get_qa_cache_by_question(db: Session, *, question: str) -> QACache | None:
    logger.info(f"Buscando en caché por pregunta: '{question}'")
    query_embedding = await get_embedding(question)

    all_cached_qas = db.query(QACache).filter(QACache.embedding.isnot(None)).all()
    logger.info(f"Recuperadas {len(all_cached_qas)} entradas de caché con embeddings.")

    best_match = None
    highest_similarity = -1
    SIMILARITY_THRESHOLD = 0.75

    for qa_entry in all_cached_qas:
        try:
            cached_embedding = json.loads(qa_entry.embedding)
            similarity = cosine_similarity(query_embedding, cached_embedding)
            logger.info(f"Comparando con caché ID {qa_entry.id} (pregunta: '{qa_entry.question[:50]}...'): Similitud = {similarity:.4f}")
            if similarity > highest_similarity and similarity >= SIMILARITY_THRESHOLD:
                highest_similarity = similarity
                best_match = qa_entry
        except Exception as e:
            logger.error(f"Error al procesar embedding de caché {qa_entry.id}: {e}")
            continue

    if best_match:
        logger.info(f"Cache HIT! Mejor coincidencia (ID: {best_match.id}) con similitud: {highest_similarity:.4f}")
        # Incrementar hit_count
        best_match.hit_count += 1
        db.add(best_match)
        db.commit()
        db.refresh(best_match)
    else:
        logger.info("Cache MISS. No se encontró una coincidencia suficientemente similar.")

    return best_match

async def create_qa_cache(db: Session, *, qa_in: QACacheCreate) -> QACache:
    logger.info(f"Intentando guardar en caché la pregunta: '{qa_in.question}'")
    question_embedding = await get_embedding(qa_in.question)

    db_obj = QACache(
        question=qa_in.question,
        answer=qa_in.answer,
        context=qa_in.context,
        embedding=json.dumps(question_embedding),
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
        hit_count=0 # Inicializar hit_count a 0
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    logger.info(f"Pregunta '{qa_in.question}' guardada en caché con ID: {db_obj.id}")
    return db_obj

def get_top_qa_cache(db: Session, limit: int = 5) -> List[QACache]:
    """Recupera las entradas de caché de Q&A con mayor hit_count."""
    return db.query(QACache).order_by(QACache.hit_count.desc()).limit(limit).all()
