import hashlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.requests import Request

from app.models import Base, Document, DocumentChunk, DocumentPage, PageStatus
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
    assert '<div class="app-shell">' in response.text
    assert 'class="nav-link is-active" href="http://testserver/"' in response.text
    assert 'class="dashboard-hero"' in response.text
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
    ("path", "pending_message", "pending_label", "layout_class"),
    [
        (
            "/upload",
            "Файл загружается и индексируется",
            "Идёт индексация...",
            "upload-layout",
        ),
        (
            "/ask",
            "Ищем информацию в документах",
            "Ищем ответ...",
            "question-layout",
        ),
    ],
)
def test_processing_forms_render_accessible_pending_status(
    path,
    pending_message,
    pending_label,
    layout_class,
):
    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 200
    assert pending_message in response.text
    assert pending_label in response.text
    assert f'class="{layout_class}"' in response.text
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
    assert 'class="empty-state-icon"' in response.text


def test_documents_page_renders_delete_form(web_session_factory):
    document_id = add_web_document(web_session_factory)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.get("/documents")

    assert response.status_code == 200
    assert f"/documents/{document_id}/delete" in response.text
    assert "Удалить документ" in response.text
    assert "data-confirm-message=" in response.text


def test_delete_document_removes_database_records_vectors_and_owned_original(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import dependencies, web_routes

    pdf_bytes = b"%PDF-1.7 source"
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    document_id = add_web_document(
        web_session_factory,
        filename="source.pdf",
        file_hash=file_hash,
    )
    with web_session_factory() as session:
        page = DocumentPage(
            document_id=document_id,
            page_number=1,
            content_hash="b" * 64,
            status=PageStatus.PARSED,
            text="Indexed content",
        )
        session.add(page)
        session.flush()
        session.add(
            DocumentChunk(
                page_id=page.id,
                page_number=1,
                chunk_index=0,
                text="Indexed chunk",
            )
        )
        session.commit()

    storage_directory = tmp_path / "documents"
    storage_directory.mkdir()
    stored_original = storage_directory / f"{file_hash}.pdf"
    stored_original.write_bytes(pdf_bytes)
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    input_original = input_directory / "source.pdf"
    input_original.write_bytes(pdf_bytes)
    monkeypatch.setattr(web_routes, "ORIGINAL_DOCUMENT_DIRECTORY", storage_directory)
    monkeypatch.setattr(web_routes, "INPUT_DOCUMENT_DIRECTORY", input_directory)

    deleted_document_ids = []

    class FakeVectorStore:
        def delete_by_document(self, deleted_document_id):
            deleted_document_ids.append(deleted_document_id)

    monkeypatch.setattr(dependencies, "get_vector_store", FakeVectorStore)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            f"/documents/{document_id}/delete",
            follow_redirects=False,
        )
        listing = client.get(response.headers["location"])

    assert response.status_code == 303
    assert response.headers["location"] == "/documents?deleted=1"
    assert deleted_document_ids == [document_id]
    assert not stored_original.exists()
    assert input_original.exists()
    assert "Документ и его данные удалены." in listing.text
    with web_session_factory() as session:
        assert session.get(Document, document_id) is None
        assert session.query(DocumentPage).count() == 0
        assert session.query(DocumentChunk).count() == 0


def test_delete_document_keeps_database_records_when_vector_delete_fails(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies

    document_id = add_web_document(web_session_factory)

    class FailingVectorStore:
        def delete_by_document(self, deleted_document_id):
            raise RuntimeError("Qdrant unavailable")

    monkeypatch.setattr(dependencies, "get_vector_store", FailingVectorStore)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(f"/documents/{document_id}/delete")

    assert response.status_code == 503
    assert "Запись в базе данных оставлена без изменений." in response.text
    with web_session_factory() as session:
        assert session.get(Document, document_id) is not None


def test_delete_document_returns_not_found_for_unknown_id(web_session_factory):
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post("/documents/missing/delete")

    assert response.status_code == 404


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


def test_documents_page_links_to_and_serves_original_pdf(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import web_routes

    pdf_bytes = b"%PDF-1.7 original"
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    document_id = add_web_document(
        web_session_factory,
        filename="contract.pdf",
        file_hash=file_hash,
    )
    storage_directory = tmp_path / "documents"
    storage_directory.mkdir()
    (storage_directory / f"{file_hash}.pdf").write_bytes(pdf_bytes)
    monkeypatch.setattr(
        web_routes,
        "ORIGINAL_DOCUMENT_DIRECTORY",
        storage_directory,
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        listing = client.get("/documents")
        original = client.get(f"/documents/{document_id}/original")

    assert f'href="http://testserver/documents/{document_id}/original"' in listing.text
    assert "contract.pdf</a>" in listing.text
    assert original.status_code == 200
    assert original.headers["content-type"] == "application/pdf"
    assert original.headers["content-disposition"].startswith("inline;")
    assert original.content == pdf_bytes


def test_documents_page_links_to_matching_pdf_in_input_directory(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import web_routes

    pdf_bytes = b"%PDF-1.7 indexed through CLI"
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    document_id = add_web_document(
        web_session_factory,
        filename="cli-document.pdf",
        file_hash=file_hash,
    )
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "cli-document.pdf").write_bytes(pdf_bytes)
    monkeypatch.setattr(
        web_routes,
        "ORIGINAL_DOCUMENT_DIRECTORY",
        tmp_path / "documents",
    )
    monkeypatch.setattr(web_routes, "INPUT_DOCUMENT_DIRECTORY", input_directory)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        listing = client.get("/documents")
        original = client.get(f"/documents/{document_id}/original")

    assert f"/documents/{document_id}/original" in listing.text
    assert "cli-document.pdf</a>" in listing.text
    assert original.status_code == 200
    assert original.content == pdf_bytes


def test_original_route_rejects_input_pdf_with_mismatched_hash(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import web_routes

    expected_bytes = b"%PDF-1.7 expected"
    document_id = add_web_document(
        web_session_factory,
        filename="cli-document.pdf",
        file_hash=hashlib.sha256(expected_bytes).hexdigest(),
    )
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    (input_directory / "cli-document.pdf").write_bytes(b"%PDF-1.7 changed")
    monkeypatch.setattr(
        web_routes,
        "ORIGINAL_DOCUMENT_DIRECTORY",
        tmp_path / "documents",
    )
    monkeypatch.setattr(web_routes, "INPUT_DOCUMENT_DIRECTORY", input_directory)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        original = client.get(f"/documents/{document_id}/original")

    assert original.status_code == 404


def test_documents_without_retained_original_have_no_link_and_return_404(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import web_routes

    document_id = add_web_document(web_session_factory, filename="cli-document.pdf")
    monkeypatch.setattr(
        web_routes,
        "ORIGINAL_DOCUMENT_DIRECTORY",
        tmp_path / "documents",
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        listing = client.get("/documents")
        original = client.get(f"/documents/{document_id}/original")

    assert f'href="http://testserver/documents/{document_id}/original"' not in (
        listing.text
    )
    assert original.status_code == 404


def test_upload_indexes_valid_pdf_and_cleans_temporary_file(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import dependencies, web_routes

    storage_directory = tmp_path / "documents"
    monkeypatch.setattr(
        web_routes,
        "ORIGINAL_DOCUMENT_DIRECTORY",
        storage_directory,
    )

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
    file_hash = hashlib.sha256(b"%PDF-1.7 test").hexdigest()
    assert (storage_directory / f"{file_hash}.pdf").read_bytes() == b"%PDF-1.7 test"
    (storage_directory / f"{file_hash}.pdf").unlink()


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
    from app.answer import AnswerResult

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
            return AnswerResult(
                "<script>alert(1)</script> Ответ. [стр. 2]",
                llm_model_name="gpt-4o-mini",
            )

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
    assert "Используемая LLM: gpt-4o-mini" in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt; Ответ. [стр. 2]" in response.text
    assert "<script>alert(1)</script>" not in response.text


def test_question_form_initial_state_lists_available_documents(web_session_factory):
    document_id = add_web_document(web_session_factory)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.get("/ask")

    assert response.status_code == 200
    assert "Результат" not in response.text
    assert "Источники для поиска" in response.text
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
        ("/documents/missing/delete", {}),
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


def add_web_chat(
    session_factory,
    document_id="doc-1",
    title="Test chat",
):
    with session_factory() as session:
        document = Document(filename="document.pdf", file_hash="d" * 64)
        session.add(document)
        session.flush()
        from app.models import Chat, Message, MessageRole

        chat = Chat(
            id=document_id,
            documents=[document],
            title=title,
        )
        session.add(chat)
        session.flush()
        message = Message(
            chat_id=chat.id,
            role=MessageRole.USER,
            content="Тестовый вопрос",
        )
        session.add(message)
        session.commit()
        return chat.id


def test_chats_list_renders_empty_state(web_session_factory):
    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.get("/chats")

    assert response.status_code == 200
    assert "Чатов ещё нет" in response.text


def test_chats_list_renders_existing(web_session_factory):
    add_web_chat(web_session_factory)
    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.get("/chats")

    assert response.status_code == 200
    assert "document.pdf" in response.text


def test_shared_layout_displays_recent_chats_in_right_sidebar(web_session_factory):
    chat_id = add_web_chat(
        web_session_factory,
        document_id="sidebar-chat",
        title="Чат по договору",
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        home_response = client.get("/")
        chats_response = client.get("/chats")
        chat_response = client.get(f"/chats/{chat_id}")

    assert home_response.status_code == 200
    assert "chat-sidebar" not in home_response.text
    assert "Чат по договору" not in home_response.text
    assert '<aside class="chat-sidebar" aria-label="Чаты">' in chats_response.text
    assert "Чат по договору" in chats_response.text
    assert f'href="http://testserver/chats/{chat_id}"' in chats_response.text
    assert 'class="chat-sidebar-link is-active"' in chat_response.text
    assert f'href="http://testserver/chats/{chat_id}"' in chat_response.text
    assert "В процессе" in chat_response.text
    assert "Новый чат" not in chats_response.text
    assert 'placeholder="Выберите чат"' in chats_response.text
    assert 'aria-label="Отправить вопрос" disabled' in chats_response.text
    assert 'placeholder="Напишите вопрос..."' in chat_response.text
    assert '<span aria-hidden="true">▶</span>' in chat_response.text


def test_shared_layout_lists_all_chats_without_limit(web_session_factory):
    with web_session_factory() as session:
        document = Document(filename="document.pdf", file_hash="d" * 64)
        session.add(document)
        session.flush()
        from app.models import Chat

        session.add_all(
            [
                Chat(
                    id=f"chat-{index}",
                    title=f"Saved chat {index}",
                    documents=[document],
                )
                for index in range(10)
            ]
        )
        session.commit()

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.get("/chats/chat-0")

    assert response.status_code == 200
    assert "Saved chat 9" in response.text


def test_create_chat_with_multiple_documents(web_session_factory):
    with web_session_factory() as session:
        first = Document(filename="first.pdf", file_hash="1" * 64)
        second = Document(filename="second.pdf", file_hash="2" * 64)
        session.add_all([first, second])
        session.commit()
        document_ids = [first.id, second.id]

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.post(
            "/chats",
            data={"document_ids": document_ids},
            follow_redirects=False,
        )

    assert response.status_code == 303
    with web_session_factory() as session:
        from app.models import Chat

        chat = session.query(Chat).one()
        assert {document.id for document in chat.documents} == set(document_ids)


def test_create_chat_requires_at_least_one_document(web_session_factory):
    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.post("/chats", data={})

    assert response.status_code == 422


def test_chat_view_returns_404_for_missing(web_session_factory):
    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.get("/chats/missing-id")

    assert response.status_code == 404


def test_chat_view_renders_document_and_messages(web_session_factory):
    chat_id = add_web_chat(web_session_factory)
    with web_session_factory() as session:
        from app.models import Message, MessageRole

        first_message = session.query(Message).filter_by(chat_id=chat_id).one()
        first_message.created_at = datetime(2026, 1, 1, 10, 0)
        session.add_all(
            [
                Message(
                    chat_id=chat_id,
                    role=MessageRole.ASSISTANT,
                    content="Первый ответ",
                    created_at=datetime(2026, 1, 1, 10, 1),
                ),
                Message(
                    chat_id=chat_id,
                    role=MessageRole.USER,
                    content="Второй вопрос",
                    created_at=datetime(2026, 1, 1, 10, 2),
                ),
                Message(
                    chat_id=chat_id,
                    role=MessageRole.ASSISTANT,
                    content="Второй ответ",
                    created_at=datetime(2026, 1, 1, 10, 3),
                ),
            ]
        )
        session.commit()

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.get(f"/chats/{chat_id}")

    assert response.status_code == 200
    assert "document.pdf" in response.text
    expected_history = [
        "Тестовый вопрос",
        "Первый ответ",
        "Второй вопрос",
        "Второй ответ",
    ]
    positions = [response.text.index(message) for message in expected_history]
    assert positions == sorted(positions)
    assert response.text.count('class="message-card message-user"') == 2
    assert response.text.count('class="message-card message-assistant"') == 2


def test_send_message_empty_content_returns_422(web_session_factory):
    chat_id = add_web_chat(web_session_factory)
    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "   "},
        )

    assert response.status_code == 422
    assert "Введите сообщение" in response.text


def test_send_message_persists_user_and_assistant(web_session_factory, monkeypatch):
    chat_id = add_web_chat(web_session_factory)

    class FakeRAGPipeline:
        def __init__(self):
            self.answers = [
                "Первый ответ. [стр. 1]",
                "Второй ответ. [стр. 1]",
            ]

        def answer_with_sources(self, question, document_id=None, **kwargs):
            from app.answer import ChatAnswerResult
            from app.context import SourceReference

            return ChatAnswerResult(
                answer=self.answers.pop(0),
                model_name="test-model",
                sources=(
                    SourceReference(
                        document_id="doc-1",
                        page_number=1,
                        chunk_id="chunk-1",
                    ),
                ),
            )

    from app import dependencies

    rag_pipeline = FakeRAGPipeline()
    monkeypatch.setattr(
        dependencies,
        "create_rag_pipeline",
        lambda: rag_pipeline,
    )

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        first_response = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "Каков срок договора?"},
        )
        second_response = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "А сколько длится второй?"},
        )
        history_response = client.get(f"/chats/{chat_id}")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert history_response.status_code == 200
    expected_history = [
        "Каков срок договора?",
        "Первый ответ. [стр. 1]",
        "А сколько длится второй?",
        "Второй ответ. [стр. 1]",
    ]
    positions = [history_response.text.index(message) for message in expected_history]
    assert positions == sorted(positions)


def test_send_message_rejects_messages_above_maximum_length(web_session_factory):
    chat_id = add_web_chat(web_session_factory)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "a" * 2001},
        )

    assert response.status_code == 422
    assert "не должно превышать 2000 символов" in response.text


def test_delete_chat_removes_record(web_session_factory):
    chat_id = add_web_chat(web_session_factory)
    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.post(
            f"/chats/{chat_id}/delete",
            follow_redirects=False,
        )

    assert response.status_code == 303
    with web_session_factory() as session:
        from app.models import Chat

        assert session.get(Chat, chat_id) is None


def test_chat_mutating_routes_reject_cross_origin(web_session_factory):
    chat_id = add_web_chat(web_session_factory)
    with TestClient(app) as client:
        response = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "Вопрос"},
            headers={"Origin": "https://untrusted.example"},
        )
        response_delete = client.post(
            f"/chats/{chat_id}/delete",
            headers={"Origin": "https://untrusted.example"},
        )

    assert response.status_code == 403
    assert response_delete.status_code == 403


def test_chat_list_link_in_navigation(web_session_factory):
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert 'href="http://testserver/chats"' in response.text
    assert "Чаты" in response.text


def test_delete_document_preserves_multi_document_chat_and_messages(
    web_session_factory, monkeypatch
):
    with web_session_factory() as session:
        document = Document(filename="document.pdf", file_hash="e" * 64)
        other_document = Document(filename="other.pdf", file_hash="f" * 64)
        session.add_all([document, other_document])
        session.flush()
        from app.models import Chat, Message, MessageRole

        chat = Chat(
            documents=[document, other_document],
            title="Test",
        )
        session.add(chat)
        session.flush()
        message = Message(
            chat_id=chat.id,
            role=MessageRole.USER,
            content="Тест",
        )
        session.add(message)
        session.commit()
        document_id = document.id
        other_document_id = other_document.id
        chat_id = chat.id
        message_id = message.id

    application = create_app(session_factory=web_session_factory)

    class FakeVectorStore:
        def delete_by_document(self, document_id):
            pass

        def upsert(self, vector_id, vector, payload):
            pass

        def search(
            self,
            vector,
            limit=5,
            score_threshold=None,
            document_id=None,
            document_ids=None,
        ):
            return []

        def delete(self, vector_id):
            pass

        def delete_by_page(self, page_id, exclude_ids=None):
            pass

    from app import dependencies

    monkeypatch_store = FakeVectorStore()
    monkeypatch.setattr(dependencies, "get_vector_store", lambda: monkeypatch_store)

    with TestClient(application) as client:
        response = client.post(
            f"/documents/{document_id}/delete",
            follow_redirects=False,
        )

    assert response.status_code == 303

    with web_session_factory() as session:
        from app.models import Chat, Message

        assert session.get(Document, document_id) is None
        saved_chat = session.get(Chat, chat_id)
        assert saved_chat is not None
        assert [document.id for document in saved_chat.documents] == [other_document_id]
        assert session.get(Message, message_id) is not None


def test_delete_chat_cascades_to_messages(web_session_factory):
    chat_id = add_web_chat(web_session_factory)

    with web_session_factory() as session:
        from app.models import Message, MessageRole

        message = Message(
            chat_id=chat_id,
            role=MessageRole.USER,
            content="Тест",
        )
        session.add(message)
        session.commit()
        message_id = message.id

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.post(
            f"/chats/{chat_id}/delete",
            follow_redirects=False,
        )

    assert response.status_code == 303

    with web_session_factory() as session:
        from app.models import Chat, Message

        assert session.get(Chat, chat_id) is None
        assert session.get(Message, message_id) is None
