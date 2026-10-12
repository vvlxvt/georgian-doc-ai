import logging
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session, selectinload

from app.models import Document, PageStatus
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
logger = logging.getLogger(__name__)
router = APIRouter()
TEMPLATE_DIRECTORY = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=TEMPLATE_DIRECTORY)


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
        },
    )


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
    documents = _get_question_documents(session)
    return templates.TemplateResponse(
        request=request,
        name="ask.html",
        context={
            "answer": None,
            "documents": documents,
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
    question = question.strip()
    answer = None
    error = None
    status_code = 200
    documents = _get_question_documents(session)
    available_document_ids = {document.id for document in documents}
    selected_document_ids = list(dict.fromkeys(document_ids))

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
        except Exception:
            logger.exception("RAG question processing failed")
            error = "Не удалось получить ответ. Проверьте доступность служб."
            status_code = 503

    return templates.TemplateResponse(
        request=request,
        name="ask.html",
        context={
            "answer": answer,
            "documents": documents,
            "error": error,
            "max_question_chars": MAX_QUESTION_CHARS,
            "question": question,
            "selected_document_ids": selected_document_ids,
        },
        status_code=status_code,
    )


@router.get("/health", name="health")
def health() -> dict[str, str]:
    return {"status": "ok"}
