from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from starlette.responses import PlainTextResponse

from app.database import SessionLocal
from app.web_routes import router

STATIC_DIRECTORY = Path(__file__).parent / "static"


def create_app(
    session_factory: Callable[[], Session] | None = None,
) -> FastAPI:
    application = FastAPI(
        title="Georgian Document RAG",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    application.state.session_factory = session_factory or SessionLocal
    application.include_router(router)
    application.mount(
        "/static",
        StaticFiles(directory=STATIC_DIRECTORY, check_dir=False),
        name="static",
    )

    @application.middleware("http")
    async def protect_form_posts(request, call_next):
        is_document_delete = request.url.path.startswith(
            "/documents/"
        ) and request.url.path.endswith("/delete")
        is_conversation_mutating = request.url.path.startswith("/conversations/") and (
            request.url.path.endswith("/messages")
            or request.url.path.endswith("/delete")
        )
        if request.method == "POST" and (
            request.url.path in {"/upload", "/ask"}
            or is_document_delete
            or is_conversation_mutating
        ):
            origin = request.headers.get("origin")
            if origin and not _is_same_origin(origin, str(request.base_url)):
                return PlainTextResponse(
                    "Запрос отклонён: источник не совпадает с адресом приложения.",
                    status_code=403,
                )
        return await call_next(request)

    return application


def _is_same_origin(origin: str, base_url: str) -> bool:
    try:
        origin_parts = urlsplit(origin)
        base_parts = urlsplit(base_url)
        origin_port = origin_parts.port or (
            443 if origin_parts.scheme == "https" else 80
        )
        base_port = base_parts.port or (443 if base_parts.scheme == "https" else 80)
    except ValueError:
        return False

    return (
        origin_parts.scheme in {"http", "https"}
        and origin_parts.scheme.lower() == base_parts.scheme.lower()
        and origin_parts.hostname is not None
        and origin_parts.hostname.lower() == (base_parts.hostname or "").lower()
        and origin_port == base_port
        and not origin_parts.path
        and not origin_parts.query
        and not origin_parts.fragment
    )


app = create_app()
