from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

# Cargar explícitamente el archivo .env desde la raíz del proyecto
# Esto asegura que se encuentre sin importar desde dónde se inicie el servidor.
# El argumento `override=True` fuerza a que los valores del .env reemplacen
# a cualquier variable de entorno del sistema con el mismo nombre.
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path=dotenv_path, override=True)

class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str
    COHERE_API_KEY: str

    

    # --- Variables con valores por defecto ---
    # Estos valores pueden ser sobrescritos por el archivo .env
    PINECONE_INDEX_NAME: str = "vigilancia-dev-index"
    PINECONE_BATCH_SIZE: int = 100
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_LLM_MODEL: str = "gpt-3.5-turbo"

    # Le decimos a Pydantic que cargue desde .env y que ignore cualquier variable extra que encuentre.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()


