import hashlib
import logging
import shutil
from datetime import datetime
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.chat import ChatService
from app.llm import LLMError, LLMErrorCategory
from app.models import (
    Chat,
    Document,
    DocumentPage,
    PageStatus,
)
from app.repositories import ChatRepository
from app.web_dependencies import get_session

PAGE_SIZE = 20
PAGE_STATUS_LABELS = {
    PageStatus.PENDING: "Ожидает обработки",
    PageStatus.PARSED: "Текст извлечён",
    PageStatus.OCR: "Распознано OCR",
    PageStatus.OCR_FAILED: "Ошибка OCR",
}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_QUESTION_CHARS = 2000


def _format_datetime(value):
    """Format a datetime or already-string timestamp for display."""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    return str(value)[:16]


def _messages_for_display(messages):
    """Attach formatted timestamps to messages."""
    for message in messages:
        message.created_at_display = _format_datetime(message.created_at)
    return messages


ORIGINAL_DOCUMENT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "documents"
INPUT_DOCUMENT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "input"
logger = logging.getLogger(__name__)
router = APIRouter()
TEMPLATE_DIRECTORY = Path(__file__).parent / "templates"


def _chat_sidebar_context(request: Request) -> dict:
    if not request.url.path.startswith("/chats"):
        return {"sidebar_chats": []}

    session = request.app.state.session_factory()
    try:
        chats = session.execute(
            select(Chat.id, Chat.title).order_by(Chat.updated_at.desc(), Chat.id.desc())
        ).all()
        return {
            "sidebar_chats": [
                {"id": chat_id, "title": title} for chat_id, title in chats
            ]
        }
    finally:
        session.close()


templates = Jinja2Templates(
    directory=TEMPLATE_DIRECTORY,
    context_processors=[_chat_sidebar_context],
)


def _llm_failure_message(error: LLMError) -> str:
    if error.category is LLMErrorCategory.AUTHENTICATION:
        return "Проверьте настройку ключа доступа к выбранному LLM-провайдеру."
    if error.category is LLMErrorCategory.INSUFFICIENT_CREDITS:
        return "Проверьте баланс и лимиты аккаунта LLM-провайдера."
    if error.category is LLMErrorCategory.RATE_LIMIT:
        return "Провайдер ограничил частоту запросов. Повторите попытку позже."
    if error.category is LLMErrorCategory.CONTEXT_LENGTH:
        return "Сократите вопрос или уменьшите объём найденного контекста."
    if error.category is LLMErrorCategory.OUTCOME_UNKNOWN:
        return (
            "Результат запроса не подтверждён: провайдер мог его обработать. "
            "Проверьте лимит расходов перед повторной отправкой."
        )
    if error.category is LLMErrorCategory.TIMEOUT:
        return "Превышен лимит времени запроса. Проверьте доступность провайдера."
    if error.category is LLMErrorCategory.CONNECTION:
        return "Не удалось подключиться к LLM-провайдеру. Проверьте его доступность."
    if error.category in {
        LLMErrorCategory.INVALID_REQUEST,
        LLMErrorCategory.UNSUPPORTED_CAPABILITY,
    }:
        return "Проверьте модель и параметры выбранного LLM-провайдера."
    if error.category in {
        LLMErrorCategory.INVALID_RESPONSE,
        LLMErrorCategory.REFUSAL,
        LLMErrorCategory.VALIDATION,
    }:
        return (
            "Ответ провайдера не прошёл проверку. Попробуйте переформулировать вопрос."
        )
    return "Не удалось получить ответ от LLM-провайдера. Проверьте его доступность."


@router.get("/", name="home")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="home.html",
        context={},
    )


@router.get("/documents", name="documents")
def documents(
    request: Request,
    session: Session = Depends(get_session),
    page: int = Query(default=1, ge=1),
    deleted: bool = Query(default=False),
    cleanup_warning: bool = Query(default=False),
):
    return _render_documents(
        request,
        session,
        page=page,
        deleted=deleted,
        cleanup_warning=cleanup_warning,
    )


def _render_documents(
    request: Request,
    session: Session,
    page: int = 1,
    *,
    error: str | None = None,
    deleted: bool = False,
    cleanup_warning: bool = False,
    status_code: int = 200,
):
    total = session.scalar(select(func.count()).select_from(Document)) or 0
    page_count = max(1, ceil(total / PAGE_SIZE))
    page = min(page, page_count)
    records = session.scalars(
        select(Document)
        .options(selectinload(Document.pages))
        .order_by(Document.created_at.desc(), Document.filename)
        .offset((page - 1) * PAGE_SIZE)
        .limit(PAGE_SIZE)
    ).all()
    documents_with_status = []

    for document in records:
        counts = {}
        for document_page in document.pages:
            counts[document_page.status] = counts.get(document_page.status, 0) + 1

        statuses = [
            {
                "label": PAGE_STATUS_LABELS.get(status, "Неизвестный статус"),
                "count": count,
            }
            for status, count in counts.items()
        ]
        documents_with_status.append(
            {
                "document": document,
                "page_count": len(document.pages),
                "original_available": _original_document_path(document) is not None,
                "statuses": statuses,
            }
        )

    return templates.TemplateResponse(
        request=request,
        name="documents.html",
        context={
            "documents": documents_with_status,
            "page": page,
            "page_count": page_count,
            "total": total,
            "error": error,
            "deleted": deleted,
            "cleanup_warning": cleanup_warning,
        },
        status_code=status_code,
    )


def _original_document_path(document: Document) -> Path | None:
    if len(document.file_hash) != 64 or any(
        character not in "0123456789abcdef" for character in document.file_hash
    ):
        return None

    candidates = [
        (
            ORIGINAL_DOCUMENT_DIRECTORY,
            f"{document.file_hash}.pdf",
        )
    ]
    filename = Path(document.filename)
    if filename.name == document.filename and filename.suffix.lower() == ".pdf":
        candidates.append((INPUT_DOCUMENT_DIRECTORY, document.filename))

    for directory, filename in candidates:
        storage_directory = directory.resolve()
        original_path = directory / filename
        if (
            not original_path.is_symlink()
            and original_path.resolve().parent == storage_directory
            and original_path.is_file()
        ):
            return original_path
    return None


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(UPLOAD_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest()


@router.get("/documents/{document_id}/original", name="document_original")
def document_original(
    document_id: str,
    session: Session = Depends(get_session),
):
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    original_path = _original_document_path(document)
    if original_path is None or _file_sha256(original_path) != document.file_hash:
        raise HTTPException(status_code=404, detail="Original PDF not found")

    return FileResponse(
        original_path,
        media_type="application/pdf",
        filename=document.filename,
        content_disposition_type="inline",
    )


@router.post("/documents/{document_id}/delete", name="delete_document")
def delete_document(
    document_id: str,
    request: Request,
    session: Session = Depends(get_session),
):
    document = session.scalar(
        select(Document)
        .options(selectinload(Document.pages).selectinload(DocumentPage.chunks))
        .where(Document.id == document_id)
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        from app.dependencies import get_vector_store

        get_vector_store().delete_by_document(document.id)
    except Exception:
        logger.exception("Failed to delete document vectors")
        return _render_documents(
            request,
            session,
            error=(
                "Не удалось удалить векторы документа. Запись в базе данных "
                "оставлена без изменений."
            ),
            status_code=503,
        )

    document_hash = document.file_hash
    stored_original = _stored_original_path(document)
    session.delete(document)
    try:
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Failed to delete document database records")
        return _render_documents(
            request,
            session,
            error=(
                "Векторы документа удалены, но запись в базе данных удалить "
                "не удалось. Проверьте журнал приложения."
            ),
            status_code=503,
        )

    cleanup_warning = False
    if stored_original is not None:
        try:
            if _file_sha256(stored_original) != document_hash:
                logger.warning("Stored original PDF hash did not match document")
                cleanup_warning = True
            else:
                stored_original.unlink()
        except OSError:
            logger.exception("Failed to remove stored original PDF")
            cleanup_warning = True

    query = "cleanup_warning=1" if cleanup_warning else "deleted=1"
    return RedirectResponse(url=f"/documents?{query}", status_code=303)


def _stored_original_path(document: Document) -> Path | None:
    if len(document.file_hash) != 64 or any(
        character not in "0123456789abcdef" for character in document.file_hash
    ):
        return None

    original_path = ORIGINAL_DOCUMENT_DIRECTORY / f"{document.file_hash}.pdf"
    if (
        original_path.is_symlink()
        or original_path.resolve().parent != ORIGINAL_DOCUMENT_DIRECTORY.resolve()
        or not original_path.is_file()
    ):
        return None
    return original_path


@router.get("/upload", name="upload_form")
def upload_form(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={"max_upload_mib": MAX_UPLOAD_BYTES // (1024 * 1024)},
    )


@router.post("/upload", name="upload_document")
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    filename = (file.filename or "").replace("\\", "/")
    filename = Path(filename).name.strip()
    error = None
    status_code = 200
    document_id = None

    if not filename or Path(filename).suffix.lower() != ".pdf":
        error = "Выберите файл с расширением .pdf."
        status_code = 422
        file.file.close()
    else:
        try:
            with TemporaryDirectory(prefix="georgian-doc-") as temp_directory:
                temp_path = Path(temp_directory) / filename
                total_bytes = 0
                header = b""
                file_hash = hashlib.sha256()

                with temp_path.open("wb") as temp_file:
                    while chunk := file.file.read(UPLOAD_CHUNK_BYTES):
                        total_bytes += len(chunk)
                        if total_bytes > MAX_UPLOAD_BYTES:
                            error = (
                                "Файл превышает допустимый размер "
                                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} МиБ."
                            )
                            status_code = 413
                            break
                        if not header:
                            header = chunk[:5]
                        temp_file.write(chunk)
                        file_hash.update(chunk)

                if error is None and total_bytes == 0:
                    error = "Файл пуст. Выберите PDF с содержимым."
                    status_code = 422
                elif error is None and header != b"%PDF-":
                    error = "Файл не похож на PDF. Проверьте выбранный документ."
                    status_code = 422

                if error is None:
                    from app.dependencies import create_document_pipeline

                    document_id = create_document_pipeline(session).process(
                        str(temp_path)
                    )
                    ORIGINAL_DOCUMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
                    original_path = (
                        ORIGINAL_DOCUMENT_DIRECTORY / f"{file_hash.hexdigest()}.pdf"
                    )
                    with TemporaryDirectory(
                        prefix="georgian-original-",
                        dir=ORIGINAL_DOCUMENT_DIRECTORY,
                    ) as staging_directory:
                        staged_path = Path(staging_directory) / "original.pdf"
                        shutil.copyfile(temp_path, staged_path)
                        staged_path.replace(original_path)
        except Exception:
            logger.exception("PDF upload and indexing failed")
            error = (
                "Не удалось обработать документ. Проверьте доступность служб "
                "и повторите попытку."
            )
            status_code = 503
        finally:
            file.file.close()

    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={
            "error": error,
            "document_id": document_id,
            "filename": filename,
            "max_upload_mib": MAX_UPLOAD_BYTES // (1024 * 1024),
        },
        status_code=status_code,
    )


@router.get("/ask", name="ask_form")
def ask_form(
    request: Request,
    session: Session = Depends(get_session),
):
    from app.dependencies import get_llm_service

    documents = _get_question_documents(session)
    return templates.TemplateResponse(
        request=request,
        name="ask.html",
        context={
            "answer": None,
            "documents": documents,
            "llm_model_name": get_llm_service().model,
            "max_question_chars": MAX_QUESTION_CHARS,
            "question": "",
            "selected_document_ids": [],
        },
    )


def _get_question_documents(session: Session) -> list[Document]:
    return session.scalars(
        select(Document).order_by(Document.created_at.desc(), Document.filename)
    ).all()


@router.post("/ask", name="ask_question")
def ask_question(
    request: Request,
    question: str = Form(...),
    document_ids: list[str] = Form(default=[]),
    session: Session = Depends(get_session),
):
    from app.dependencies import get_llm_service

    question = question.strip()
    answer = None
    error = None
    status_code = 200
    documents = _get_question_documents(session)
    available_document_ids = {document.id for document in documents}
    selected_document_ids = list(dict.fromkeys(document_ids))
    llm_model_name = get_llm_service().model

    if not question:
        error = "Введите вопрос."
        status_code = 422
    elif len(question) > MAX_QUESTION_CHARS:
        error = f"Вопрос не должен превышать {MAX_QUESTION_CHARS} символов."
        status_code = 422
    elif not set(selected_document_ids).issubset(available_document_ids):
        error = "Выберите документы из списка и повторите попытку."
        status_code = 422
    else:
        try:
            from app.dependencies import create_rag_pipeline

            rag_pipeline = create_rag_pipeline()
            if selected_document_ids:
                answer = rag_pipeline.answer(
                    question=question,
                    document_ids=selected_document_ids,
                )
            else:
                answer = rag_pipeline.answer(question=question)
        except LLMError as exc:
            logger.warning(
                "LLM question processing failed",
                extra={
                    "llm_provider": exc.provider,
                    "llm_model": exc.model,
                    "llm_category": exc.category.value,
                    "llm_attempts": exc.attempts,
                    "llm_duration_seconds": exc.duration_seconds,
                    "llm_request_id": exc.provider_request_id,
                },
            )
            error = _llm_failure_message(exc)
            status_code = 503
        except Exception:
            logger.exception("RAG question processing failed")
            error = "Не удалось получить ответ. Проверьте доступность служб."
            status_code = 503

    if answer is not None:
        llm_model_name = getattr(answer, "model_name", None) or llm_model_name

    return templates.TemplateResponse(
        request=request,
        name="ask.html",
        context={
            "answer": answer,
            "documents": documents,
            "error": error,
            "llm_model_name": llm_model_name,
            "max_question_chars": MAX_QUESTION_CHARS,
            "question": question,
            "selected_document_ids": selected_document_ids,
        },
        status_code=status_code,
    )


@router.get("/chats", name="chats")
def chats_list(
    request: Request,
    session: Session = Depends(get_session),
):
    repository = ChatRepository(session)
    chats = repository.list_all()
    documents = session.scalars(
        select(Document).order_by(Document.filename, Document.id)
    ).all()
    for chat in chats:
        chat.created_at_display = _format_datetime(chat.created_at)
        chat.updated_at_display = _format_datetime(chat.updated_at)
    return templates.TemplateResponse(
        request=request,
        name="chats.html",
        context={"chats": chats, "documents": documents},
    )


@router.post("/chats", name="create_chat")
def create_chat(
    request: Request,
    document_ids: list[str] = Form(default=[]),
    session: Session = Depends(get_session),
):
    repository = ChatRepository(session)
    if not document_ids:
        raise HTTPException(
            status_code=422,
            detail="Выберите хотя бы один документ для чата",
        )
    try:
        chat = repository.create(document_ids=document_ids)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Документ не найден") from exc
    session.commit()
    return RedirectResponse(
        url=router.url_path_for("chat_view", chat_id=chat.id),
        status_code=303,
    )


@router.get("/chats/{chat_id}", name="chat_view")
def chat_view(
    request: Request,
    chat_id: str,
    session: Session = Depends(get_session),
):
    repository = ChatRepository(session)
    chat = repository.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Чат не найден")
    messages = _messages_for_display(chat.messages)
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "chat": chat,
            "messages": messages,
            "error": None,
            "content": "",
            "max_question_chars": MAX_QUESTION_CHARS,
        },
    )


@router.post("/chats/{chat_id}/messages", name="send_message")
def send_message(
    request: Request,
    chat_id: str,
    content: str = Form(...),
    session: Session = Depends(get_session),
):
    from app.dependencies import create_rag_pipeline

    repository = ChatRepository(session)
    chat = repository.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Чат не найден")

    content_stripped = content.strip()
    error = None
    rag_result = None
    status_code = 200

    if not content_stripped:
        error = "Введите сообщение."
        status_code = 422
    elif len(content_stripped) > MAX_QUESTION_CHARS:
        error = f"Сообщение не должно превышать {MAX_QUESTION_CHARS} символов."
        status_code = 422
    else:
        try:
            chat_service = ChatService(
                session=session,
                chat_repository=repository,
                rag_pipeline=create_rag_pipeline(),
            )
            rag_result = chat_service.send_message(
                chat_id=chat_id,
                content=content_stripped,
            )
            if rag_result.error is not None:
                error = rag_result.error
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Chat message processing failed")
            error = "Не удалось отправить сообщение. Проверьте доступность служб."
            rag_result = None
            status_code = 503

    # Re-fetch chat and messages after possible state change
    chat = repository.get(chat_id)
    messages = _messages_for_display(chat.messages) if chat else []

    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={
            "chat": chat,
            "messages": messages,
            "error": error,
            "content": content if status_code == 422 else "",
            "max_question_chars": MAX_QUESTION_CHARS,
        },
        status_code=status_code,
    )


@router.post("/chats/{chat_id}/delete", name="delete_chat")
def delete_chat(
    request: Request,
    chat_id: str,
    session: Session = Depends(get_session),
):
    repository = ChatRepository(session)
    chat = repository.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Чат не найден")
    repository.delete(chat)
    session.commit()
    return RedirectResponse(url=router.url_path_for("chats"), status_code=303)


@router.post("/chats/delete-selected", name="delete_selected_chats")
def delete_selected_chats(
    chat_ids: list[str] = Form(default=[]),
    session: Session = Depends(get_session),
):
    ChatRepository(session).delete_many(chat_ids)
    session.commit()
    return RedirectResponse(url=router.url_path_for("chats"), status_code=303)


@router.get("/health", name="health")
def health() -> dict[str, str]:
    return {"status": "ok"}
