from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    DATABASE_URL: str
    OPENAI_API_KEY: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    CLOUDINARY_CLOUD_NAME: str
    CLOUDINARY_API_KEY: str
    CLOUDINARY_API_SECRET: str

    # Le decimos a Pydantic que cargue desde .env y que ignore cualquier variable extra que encuentre.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()


