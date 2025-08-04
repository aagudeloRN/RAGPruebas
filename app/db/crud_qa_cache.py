# app/db/crud_qa_cache.py
from sqlalchemy.orm import Session
from app.models.qa_cache import QACache
from app.schemas.qa_cache import QACacheCreate
from app.core.config import settings
from typing import List
from openai import AsyncOpenAI
import logging

logger = logging.getLogger(__name__)

client_openai = AsyncOpenAI()

async def get_embedding(text: str) -> List[float]:
    response = await client_openai.embeddings.create(
        input=text,
        model=settings.OPENAI_EMBEDDING_MODEL
    )
    return response.data[0].embedding

async def _get_canonical_question(question: str) -> str:
    prompt = (
        "Eres un asistente experto en lenguaje. Dada una pregunta de usuario, "
        "reescríbela para que sea gramaticalmente correcta, clara y concisa, "
        "sin cambiar su significado original. Corrige errores de ortografía o tipográficos. "
        "Devuelve solo la pregunta reescrita, sin comentarios adicionales."
        f"\n\nPregunta original: {question}"
        "\nPregunta canónica:"
    )
    try:
        response = await client_openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.0,
        )
        canonical_question = response.choices[0].message.content.strip()
        logger.info(f"Pregunta original: '{question}' -> Pregunta canónica: '{canonical_question}'")
        return canonical_question
    except Exception as e:
        logger.error(f"Error al generar pregunta canónica: {e}", exc_info=True)
        return question

async def get_qa_cache_by_question(db: Session, *, question: str, kb_id: str) -> QACache | None:
    logger.info(f"Buscando en caché para KB '{kb_id}' por pregunta: '{question}'")
    canonical_query = await _get_canonical_question(question)
    query_embedding = await get_embedding(canonical_query)

    SIMILARITY_THRESHOLD = 0.15

    results = db.query(QACache, QACache.embedding.cosine_distance(query_embedding).label("distance")) \
                .filter(QACache.kb_id == kb_id) \
                .order_by("distance").limit(1).all()

    if not results:
        logger.info(f"Cache MISS para KB '{kb_id}'. No se encontró ninguna coincidencia.")
        return None

    best_match, distance = results[0]
    similarity = 1 - distance

    if similarity >= (1 - SIMILARITY_THRESHOLD):
        logger.info(f"Cache HIT para KB '{kb_id}'! ID: {best_match.id}, Similitud: {similarity:.4f}")
        best_match.hit_count += 1
        db.commit()
        db.refresh(best_match)
        return best_match
    else:
        logger.info(f"Cache MISS para KB '{kb_id}'. Similitud: {similarity:.4f} no supera el umbral.")
        return None

async def create_qa_cache(db: Session, *, qa_in: QACacheCreate, kb_id: str) -> QACache:
    logger.info(f"Guardando en caché para KB '{kb_id}' la pregunta: '{qa_in.question}'")
    canonical_question = await _get_canonical_question(qa_in.question)
    question_embedding = await get_embedding(canonical_question)

    db_obj = QACache(
        kb_id=kb_id,
        question=canonical_question,
        answer=qa_in.answer,
        context=qa_in.context,
        context_chunks=qa_in.context_chunks,
        embedding=question_embedding,
        embedding_model=settings.OPENAI_EMBEDDING_MODEL,
        hit_count=1 # Inicia en 1 ya que se acaba de usar
    )
    db.add(db_obj)
    db.commit()
    db.refresh(db_obj)
    logger.info(f"Pregunta guardada en caché para KB '{kb_id}' con ID: {db_obj.id}")
    
    await _evict_cache_entries(db, kb_id=kb_id)
    
    return db_obj

async def _evict_cache_entries(db: Session, kb_id: str, max_size: int = 100):
    """Elimina las entradas de caché más antiguas para una KB específica si excede el tamaño."""
    try:
        current_cache_size = db.query(QACache).filter(QACache.kb_id == kb_id).count()
        if current_cache_size > max_size:
            entries_to_evict_count = current_cache_size - max_size
            entries_to_evict = db.query(QACache) \
                .filter(QACache.kb_id == kb_id) \
                .order_by(QACache.hit_count.asc(), QACache.created_at.asc()) \
                .limit(entries_to_evict_count).all()
            
            for entry in entries_to_evict:
                db.delete(entry)
            
            db.commit()
            logger.info(f"Desalojo de caché para KB '{kb_id}' completado. {len(entries_to_evict)} entradas eliminadas.")
    except Exception as e:
        db.rollback()
        logger.error(f"Error durante el desalojo de caché para KB '{kb_id}': {e}", exc_info=True)

def get_top_qa_cache(db: Session, kb_id: str, limit: int = 5) -> List[QACache]:
    """Recupera las entradas de caché más populares para una KB específica."""
    return db.query(QACache).filter(QACache.kb_id == kb_id).order_by(QACache.hit_count.desc()).limit(limit).all()
