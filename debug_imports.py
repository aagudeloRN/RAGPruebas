
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Intentando importar la aplicación...")

try:
    from app import main
    logger.info("¡La importación de la aplicación fue exitosa!")
except Exception as e:
    logger.error("Ocurrió un error durante la importación:", exc_info=True)
