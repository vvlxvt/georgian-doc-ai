from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    database_url: str = "sqlite:///data/app.db"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    embedding_model: str = "BAAI/bge-m3"
    embedding_size: int = 1024

    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"

    tesseract_languages: str = "kat+eng"


settings = Settings()
