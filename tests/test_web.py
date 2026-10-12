from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.requests import Request

from app.models import Base, Document, DocumentPage, PageStatus
from app.web import app, create_app
from app.web_dependencies import get_session


@pytest.fixture
def web_session_factory():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)
    engine.dispose()


def add_web_document(session_factory, filename="document.pdf", file_hash="a" * 64):
    with session_factory() as session:
        document = Document(filename=filename, file_hash=file_hash)
        session.add(document)
        session.commit()
        return document.id


def test_health_route_reports_web_process_liveness():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_home_page_renders_shared_layout_and_russian_content():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert '<html lang="ru">' in response.text
    assert "<title>Главная — Georgian Document RAG</title>" in response.text
    assert "Находите ответы в документах" in response.text
    assert "/static/css/main.css" in response.text


def test_static_stylesheet_is_mounted():
    with TestClient(app) as client:
        response = client.get("/static/css/main.css")

    assert response.status_code == 200
    assert "focus-visible" in response.text


def test_static_pending_script_is_mounted():
    with TestClient(app) as client:
        response = client.get("/static/js/form-pending.js")

    assert response.status_code == 200
    assert 'form.addEventListener("submit"' in response.text


@pytest.mark.parametrize(
    ("path", "pending_message", "pending_label"),
    [
        (
            "/upload",
            "Файл загружается и индексируется",
            "Идёт индексация...",
        ),
        (
            "/ask",
            "Ищем информацию в документах",
            "Ищем ответ...",
        ),
    ],
)
def test_processing_forms_render_accessible_pending_status(
    path,
    pending_message,
    pending_label,
):
    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert pending_message in response.text
    assert pending_label in response.text
    assert 'data-pending-status role="status" aria-live="polite" hidden' in (
        response.text
    )
    assert "form-pending.js" in response.text


def test_documents_page_renders_empty_state(web_session_factory):
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.get("/documents")

    assert response.status_code == 200
    assert "Пока нет документов" in response.text
    assert "не проверяет наличие векторов в Qdrant" in response.text


def test_documents_page_renders_database_data_escaped(web_session_factory):
    with web_session_factory() as session:
        document = Document(
            filename="<script>alert(1)</script>.pdf",
            file_hash="a" * 64,
        )
        session.add(document)
        session.flush()
        session.add(
            DocumentPage(
                document_id=document.id,
                page_number=1,
                content_hash="b" * 64,
                status=PageStatus.OCR_FAILED,
            )
        )
        session.commit()

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.get("/documents")

    assert response.status_code == 200
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.pdf" in response.text
    assert "<script>alert(1)</script>.pdf" not in response.text
    assert "Ошибка OCR: 1" in response.text


def test_upload_indexes_valid_pdf_and_cleans_temporary_file(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies

    processed_paths = []

    class FakeDocumentPipeline:
        def __init__(self, session):
            pass

        def process(self, path):
            processed_paths.append(path)
            assert Path(path).read_bytes().startswith(b"%PDF-")
            return "document-123"

    monkeypatch.setattr(
        dependencies,
        "create_document_pipeline",
        FakeDocumentPipeline,
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            "/upload",
            files={"file": ("valid.pdf", b"%PDF-1.7 test", "application/pdf")},
        )

    assert response.status_code == 200
    assert "Документ проиндексирован" in response.text
    assert "valid.pdf" in response.text
    assert processed_paths
    assert not Path(processed_paths[0]).exists()


@pytest.mark.parametrize(
    ("filename", "content", "expected_status", "message"),
    [
        ("notes.txt", b"%PDF-1.7 data", 422, "расширением .pdf"),
        ("empty.pdf", b"", 422, "Файл пуст"),
        ("not-a-pdf.pdf", b"not pdf", 422, "не похож на PDF"),
    ],
)
def test_upload_rejects_invalid_files(
    filename,
    content,
    expected_status,
    message,
    web_session_factory,
    monkeypatch,
):
    from app import dependencies

    def unexpected_pipeline(session):
        pytest.fail("Invalid upload must not reach ingestion")

    monkeypatch.setattr(
        dependencies,
        "create_document_pipeline",
        unexpected_pipeline,
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            "/upload",
            files={"file": (filename, content, "application/pdf")},
        )

    assert response.status_code == expected_status
    assert message in response.text


def test_upload_enforces_size_limit_and_reports_pipeline_errors(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies, web_routes

    monkeypatch.setattr(web_routes, "MAX_UPLOAD_BYTES", 4)
    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        too_large = client.post(
            "/upload",
            files={"file": ("large.pdf", b"%PDF-123456", "application/pdf")},
        )

    assert too_large.status_code == 413
    assert "превышает допустимый размер" in too_large.text

    failed_paths = []

    class BrokenPipeline:
        def __init__(self, session):
            pass

        def process(self, path):
            failed_paths.append(path)
            raise RuntimeError("private internal details")

    monkeypatch.setattr(
        dependencies,
        "create_document_pipeline",
        BrokenPipeline,
    )
    monkeypatch.setattr(web_routes, "MAX_UPLOAD_BYTES", 50 * 1024 * 1024)

    with TestClient(application) as client:
        failed = client.post(
            "/upload",
            files={"file": ("valid.pdf", b"%PDF-1.7 test", "application/pdf")},
        )

    assert failed.status_code == 503
    assert "Не удалось обработать документ" in failed.text
    assert "private internal details" not in failed.text
    assert failed_paths
    assert not Path(failed_paths[0]).exists()


def test_question_form_uses_rag_pipeline_and_escapes_answer(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies

    document_ids = [
        add_web_document(
            web_session_factory,
            filename="document-one.pdf",
            file_hash="a" * 64,
        ),
        add_web_document(
            web_session_factory,
            filename="document-two.pdf",
            file_hash="b" * 64,
        ),
    ]
    received_questions = []

    class FakeRAGPipeline:
        def answer(self, question, document_ids=None):
            received_questions.append(question)
            assert document_ids == document_ids_expected
            return "<script>alert(1)</script> Ответ. [стр. 2]"

    document_ids_expected = document_ids
    monkeypatch.setattr(
        dependencies,
        "create_rag_pipeline",
        lambda: FakeRAGPipeline(),
    )

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.post(
            "/ask",
            data={
                "question": "  Что указано?  ",
                "document_ids": document_ids,
            },
        )

    assert response.status_code == 200
    assert received_questions == ["Что указано?"]
    for document_id in document_ids:
        assert f'value="{document_id}"' in response.text
    assert response.text.count("checked") == len(document_ids)
    assert "&lt;script&gt;alert(1)&lt;/script&gt; Ответ. [стр. 2]" in response.text
    assert "<script>alert(1)</script>" not in response.text


def test_question_form_initial_state_lists_available_documents(web_session_factory):
    document_id = add_web_document(web_session_factory)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.get("/ask")

    assert response.status_code == 200
    assert "Результат" not in response.text
    assert "Искать в документах" in response.text
    assert "document.pdf" in response.text
    assert f'value="{document_id}"' in response.text


def test_question_form_displays_insufficient_information_answer(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies

    class FakeRAGPipeline:
        def answer(self, question):
            return "В предоставленных документах недостаточно информации для ответа."

    monkeypatch.setattr(
        dependencies,
        "create_rag_pipeline",
        lambda: FakeRAGPipeline(),
    )

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.post("/ask", data={"question": "Несвязанный вопрос"})

    assert response.status_code == 200
    assert "В предоставленных документах недостаточно информации" in response.text


@pytest.mark.parametrize(
    ("question", "error"),
    [
        ("   ", "Введите вопрос"),
        ("x" * 2001, "не должен превышать 2000 символов"),
    ],
)
def test_question_form_rejects_invalid_question(
    question,
    error,
    web_session_factory,
    monkeypatch,
):
    from app import dependencies

    def unexpected_pipeline():
        pytest.fail("Invalid question must not reach RAG")

    monkeypatch.setattr(dependencies, "create_rag_pipeline", unexpected_pipeline)

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.post("/ask", data={"question": question})

    assert response.status_code == 422
    assert error in response.text


def test_question_form_reports_rag_failure_without_internal_details(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies

    class BrokenRAGPipeline:
        def answer(self, question):
            raise RuntimeError("private internal details")

    monkeypatch.setattr(
        dependencies,
        "create_rag_pipeline",
        lambda: BrokenRAGPipeline(),
    )

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.post("/ask", data={"question": "Каков срок?"})

    assert response.status_code == 503
    assert "Не удалось получить ответ" in response.text
    assert "private internal details" not in response.text


def test_question_form_maps_llm_failure_and_logs_only_safe_metadata(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies, web_routes
    from app.llm import LLMError, LLMErrorCategory

    log_records = []

    class BrokenRAGPipeline:
        def answer(self, question):
            raise LLMError(
                LLMErrorCategory.RATE_LIMIT,
                "raw provider body and secret token",
                provider="openrouter",
                model="test-model",
                provider_request_id="request-123",
                attempts=2,
                duration_seconds=1.25,
            )

    monkeypatch.setattr(
        dependencies,
        "create_rag_pipeline",
        lambda: BrokenRAGPipeline(),
    )
    monkeypatch.setattr(
        web_routes.logger,
        "warning",
        lambda message, *, extra: log_records.append((message, extra)),
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            "/ask",
            data={"question": "private question text"},
        )

    assert response.status_code == 503
    assert "ограничил частоту запросов" in response.text
    assert "raw provider body" not in response.text
    assert log_records == [
        (
            "LLM question processing failed",
            {
                "llm_provider": "openrouter",
                "llm_model": "test-model",
                "llm_category": "rate_limit",
                "llm_attempts": 2,
                "llm_duration_seconds": 1.25,
                "llm_request_id": "request-123",
            },
        )
    ]
    assert "private question text" not in repr(log_records)
    assert "raw provider body" not in repr(log_records)
    assert "secret token" not in repr(log_records)


def test_question_form_warns_before_retrying_unknown_outcome(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies
    from app.llm import LLMError, LLMErrorCategory

    class BrokenRAGPipeline:
        def answer(self, question):
            raise LLMError(
                LLMErrorCategory.OUTCOME_UNKNOWN,
                "private transport detail",
                provider="openrouter",
                model="test-model",
            )

    monkeypatch.setattr(
        dependencies,
        "create_rag_pipeline",
        lambda: BrokenRAGPipeline(),
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post("/ask", data={"question": "Каков срок?"})

    assert response.status_code == 503
    assert "провайдер мог его обработать" in response.text
    assert "Проверьте лимит расходов" in response.text
    assert "private transport detail" not in response.text


def test_question_form_rejects_document_ids_not_in_database(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies

    def unexpected_pipeline():
        pytest.fail("Unknown document IDs must not reach RAG")

    monkeypatch.setattr(dependencies, "create_rag_pipeline", unexpected_pipeline)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            "/ask",
            data={
                "question": "Каков срок?",
                "document_ids": ["not-a-document"],
            },
        )

    assert response.status_code == 422
    assert "Выберите документы из списка" in response.text


@pytest.mark.parametrize(
    ("path", "data"),
    [
        ("/ask", {"question": "Каков срок?"}),
        ("/upload", {}),
    ],
)
def test_state_changing_forms_reject_cross_origin_requests(path, data):
    with TestClient(app) as client:
        response = client.post(
            path,
            data=data,
            headers={"Origin": "https://untrusted.example"},
        )

    assert response.status_code == 403
    assert "источник не совпадает" in response.text


def test_create_app_accepts_custom_session_factory():
    application = create_app(session_factory=lambda: None)

    assert application.state.session_factory() is None


def test_get_session_closes_session_after_request():
    session = SimpleNamespace(closed=False)
    session.close = lambda: setattr(session, "closed", True)
    application = create_app(session_factory=lambda: session)
    request = Request({"type": "http", "app": application})
    session_generator = get_session(request)

    assert next(session_generator) is session
    with pytest.raises(StopIteration):
        next(session_generator)

    assert session.closed is True
