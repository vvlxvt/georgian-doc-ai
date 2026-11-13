import asyncio
import hashlib
import json
import logging
import queue
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
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
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
PAGE_STATUS_CLASSES = {
    PageStatus.PENDING: "pending",
    PageStatus.PARSED: "parsed",
    PageStatus.OCR: "ocr",
    PageStatus.OCR_FAILED: "error",
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
    if not request.url.path.startswith("/chats/"):
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
                "class_name": PAGE_STATUS_CLASSES.get(status, "unknown"),
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
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
):
    filename = (file.filename or "").replace("\\", "/")
    filename = Path(filename).name.strip()
    error = None
    status_code = 200
    document_id = None

    if not filename or Path(filename).suffix.lower() != ".pdf":
        error = "Выберите файл с расширением .pdf."
        status_code = 422
        await file.close()
    else:
        temporary_upload = TemporaryDirectory(prefix="georgian-doc-")
        temp_path = Path(temporary_upload.name) / filename
        try:
            total_bytes, header, file_digest = await asyncio.to_thread(
                _copy_upload_to_temp,
                file.file,
                temp_path,
            )
        except OSError:
            temporary_upload.cleanup()
            logger.exception("Failed to stage uploaded PDF")
            error = "Не удалось принять файл. Повторите попытку."
            status_code = 503
        finally:
            await file.close()

        if error is None and total_bytes > MAX_UPLOAD_BYTES:
            error = (
                "Файл превышает допустимый размер "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} МиБ."
            )
            status_code = 413
        elif error is None and total_bytes == 0:
            error = "Файл пуст. Выберите PDF с содержимым."
            status_code = 422
        elif error is None and header != b"%PDF-":
            error = "Файл не похож на PDF. Проверьте выбранный документ."
            status_code = 422

        if error is not None:
            temporary_upload.cleanup()
        else:
            accept_header = request.headers.get("accept", "").lower()
            wants_html_response = (
                "text/html" in accept_header
                and "text/event-stream" not in accept_header
            )
            event_queue: queue.Queue[dict[str, object]] = queue.Queue(maxsize=128)
            active_step = "extraction"
            ocr_failed = False

            def enqueue_event(event: dict[str, object]) -> None:
                try:
                    event_queue.put_nowait(event)
                except queue.Full:
                    try:
                        event_queue.get_nowait()
                    except queue.Empty:
                        pass
                    event_queue.put_nowait(event)

            def report_progress(step: str, step_status: str) -> None:
                nonlocal active_step, ocr_failed
                if step_status in {"processing", "failed"}:
                    active_step = step
                event = {
                    "event": "progress",
                    "step": step,
                    "status": step_status,
                }
                if step == "ocr" and step_status == "failed":
                    ocr_failed = True
                    event["message"] = (
                        "Не удалось распознать страницу; остальные страницы "
                        "обрабатываются."
                    )
                enqueue_event(event)

            def report_page_progress(remaining_pages: int, total_pages: int) -> None:
                enqueue_event(
                    {
                        "event": "page_progress",
                        "remaining_pages": remaining_pages,
                        "total_pages": total_pages,
                    }
                )

            if not wants_html_response:
                enqueue_event(
                    {
                        "event": "progress",
                        "step": "upload",
                        "status": "completed",
                    }
                )

            def process_upload() -> dict[str, str] | None:
                nonlocal active_step
                session = None
                terminal_event = None
                try:
                    from app.dependencies import create_document_pipeline

                    session = request.app.state.session_factory()
                    document_id = create_document_pipeline(
                        session,
                        progress_callback=(
                            None if wants_html_response else report_progress
                        ),
                        page_progress_callback=(
                            None if wants_html_response else report_page_progress
                        ),
                    ).process(str(temp_path))

                    active_step = "storage"
                    ORIGINAL_DOCUMENT_DIRECTORY.mkdir(parents=True, exist_ok=True)
                    original_path = ORIGINAL_DOCUMENT_DIRECTORY / f"{file_digest}.pdf"
                    with TemporaryDirectory(
                        prefix="georgian-original-",
                        dir=ORIGINAL_DOCUMENT_DIRECTORY,
                    ) as staging_directory:
                        staged_path = Path(staging_directory) / "original.pdf"
                        shutil.copyfile(temp_path, staged_path)
                        staged_path.replace(original_path)

                    terminal_event = {
                        "event": "completed",
                        "status": "completed",
                        "document_id": document_id,
                        "filename": filename,
                        "message": (
                            "Документ обработан, но часть страниц не распознана."
                            if ocr_failed
                            else "Документ обработан и проиндексирован."
                        ),
                    }
                except Exception:
                    logger.exception("PDF upload and indexing failed")
                    terminal_event = {
                        "event": "failed",
                        "step": active_step,
                        "status": "failed",
                        "message": (
                            "Не удалось обработать документ. Проверьте "
                            "доступность служб и повторите попытку."
                        ),
                    }
                finally:
                    if session is not None:
                        try:
                            session.close()
                        except Exception:
                            logger.exception("Failed to close upload database session")
                    try:
                        temporary_upload.cleanup()
                    except OSError:
                        logger.exception("Failed to clean up temporary PDF upload")
                    if terminal_event is not None and not wants_html_response:
                        enqueue_event(terminal_event)
                return terminal_event

            async def run_upload() -> None:
                await asyncio.to_thread(process_upload)

            if wants_html_response:
                result = await asyncio.to_thread(process_upload)
                if result is None:
                    raise RuntimeError("Upload worker ended without a terminal result")
                if result["event"] == "completed":
                    document_id = result["document_id"]
                else:
                    error = result["message"]
                    status_code = 503
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

            upload_task = asyncio.create_task(run_upload())
            request.app.state.upload_tasks.add(upload_task)
            upload_task.add_done_callback(request.app.state.upload_tasks.discard)

            async def stream_events():
                while True:
                    try:
                        event = event_queue.get_nowait()
                    except queue.Empty:
                        if upload_task.done():
                            break
                        await asyncio.sleep(0.05)
                        continue

                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    if event["event"] in {"completed", "failed"}:
                        break

            return StreamingResponse(
                stream_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

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


def _copy_upload_to_temp(file, temp_path: Path) -> tuple[int, bytes, str]:
    total_bytes = 0
    header = b""
    file_hash = hashlib.sha256()

    with temp_path.open("wb") as temp_file:
        while chunk := file.read(UPLOAD_CHUNK_BYTES):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                break
            if not header:
                header = chunk[:5]
            temp_file.write(chunk)
            file_hash.update(chunk)

    return total_bytes, header, file_hash.hexdigest()


@router.get("/chats", name="chats")
def chats_list(
    request: Request,
    session: Session = Depends(get_session),
):
    repository = ChatRepository(session)
    chats = repository.list_all()
    for chat in chats:
        chat.created_at_display = _format_datetime(chat.created_at)
        chat.updated_at_display = _format_datetime(chat.updated_at)
    return templates.TemplateResponse(
        request=request,
        name="chats.html",
        context={"chats": chats},
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
    from app.dependencies import get_llm_service

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
            "llm_model_name": get_llm_service().model,
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
    from app.dependencies import create_rag_pipeline, get_llm_service

    repository = ChatRepository(session)
    chat = repository.get(chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Чат не найден")
    llm_model_name = get_llm_service().model

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
                if isinstance(rag_result.error, LLMError):
                    llm_error = rag_result.error
                    logger.warning(
                        "Chat message LLM processing failed",
                        extra={
                            "llm_provider": llm_error.provider,
                            "llm_model": llm_error.model,
                            "llm_category": llm_error.category.value,
                            "llm_attempts": llm_error.attempts,
                            "llm_duration_seconds": llm_error.duration_seconds,
                            "llm_request_id": llm_error.provider_request_id,
                        },
                    )
                    error = _llm_failure_message(llm_error)
                else:
                    logger.error(
                        "Chat message processing failed",
                        extra={"error_type": type(rag_result.error).__name__},
                    )
                    error = "Не удалось получить ответ. Проверьте доступность служб."
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
            "llm_model_name": llm_model_name,
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
