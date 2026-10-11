import httpx
import pytesseract
from qdrant_client import QdrantClient

from app.config import Settings


class PreflightError(RuntimeError):
    pass


def validate_qdrant(config: Settings) -> None:
    try:
        QdrantClient(url=config.qdrant_url).get_collections()
    except Exception as exc:
        raise PreflightError(
            f"Qdrant is unavailable at {config.qdrant_url}: {exc}"
        ) from exc


def validate_ollama(config: Settings) -> None:
    try:
        response = httpx.get(f"{config.ollama_url}/api/tags", timeout=5.0)
        response.raise_for_status()
        models = {model["name"] for model in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        raise PreflightError(
            f"Ollama is unavailable at {config.ollama_url}: {exc}"
        ) from exc

    if config.ollama_model not in models:
        raise PreflightError(f"Ollama model '{config.ollama_model}' is not installed")


def validate_tesseract(config: Settings) -> None:
    try:
        installed_languages = set(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as exc:
        raise PreflightError(
            "Tesseract executable is not installed or not on PATH"
        ) from exc

    missing_languages = set(config.tesseract_languages.split("+")) - installed_languages
    if missing_languages:
        missing = ", ".join(sorted(missing_languages))
        raise PreflightError(f"Tesseract language data is missing: {missing}")


def validate_runtime(config: Settings) -> None:
    validate_qdrant(config)
    validate_ollama(config)
    validate_tesseract(config)
