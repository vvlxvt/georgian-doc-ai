from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    embedding_model: str = "BAAI/bge-m3"
    embedding_size: int = 1024

    ollama_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434"


settings = Settings()