from pydantic_settings import BaseSettings, SettingsConfigDict

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


