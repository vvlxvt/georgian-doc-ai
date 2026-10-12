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
    qdrant_api_key: str = ""
    qdrant_collection: str = "documents"

    embedding_model: str = "BAAI/bge-m3"
    embedding_size: int = 1024

    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"

    # LLM provider configuration
    llm_provider: str = "openrouter"  # one of: 'openrouter', 'ollama'

    # OpenRouter (optional; used when llm_provider == 'openrouter')
    openrouter_url: str = "https://api.openrouter.ai"
    openrouter_api_key: str = ""
    openrouter_model: str = "gpt-4o-mini"  # default example model
    openrouter_timeout: float = 120.0

    tesseract_languages: str = "kat+eng"

    chunk_max_length: int = 1000
    chunk_overlap: int = 100
    min_text_length: int = 20


settings = Settings()
