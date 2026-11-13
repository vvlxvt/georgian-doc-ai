import asyncio
import hashlib
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.datastructures import UploadFile as StarletteUploadFile
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
    assert "Выбрать документы" in response.text
    assert "Задайте вопрос" not in response.text
    assert "/ask" not in response.text
    assert '<div class="app-shell">' in response.text
    assert 'class="nav-link is-active" href="http://testserver/"' in response.text
    assert "data-vector-upload-status" in response.text
    assert "Загрузка в векторную БД" in response.text
    assert 'class="nav-upload-status"' in response.text
    assert "nav-upload-dots" in response.text
    upload_status_start = response.text.index("data-vector-upload-status")
    upload_status_markup = response.text[
        upload_status_start : upload_status_start + 180
    ]
    assert 'role="status"' in upload_status_markup
    assert 'aria-live="polite"' in upload_status_markup
    assert "hidden" in upload_status_markup
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
    assert 'form.matches("[data-vector-upload-form]")' in response.text
    assert 'document.querySelector("[data-vector-upload-status]")' in response.text
    assert 'form.classList.contains("chat-composer")' in response.text
    assert "messageList.append(messageCard)" in response.text
    assert 'messageInput.value = ""' in response.text
    assert 'event.target.closest("[data-edit-question]")' in response.text
    assert 'editButton.dataset.editQuestion = ""' in response.text
    assert "submittedContent.name = messageInput.name" in response.text
    assert "messageInput.scrollHeight" in response.text
    assert "messageInput.placeholder = pendingMessage" in response.text
    assert "sessionStorage.setItem(chatScrollStorageKey, window.location.pathname)" in (
        response.text
    )
    assert 'lastElementChild?.scrollIntoView({ block: "end" })' in response.text
    assert "form.querySelector('input[name=\"chat_ids\"]:checked')" in response.text
    assert "Array.from(form.elements).filter(" in response.text
    assert 'element.name === "document_ids" && element.type === "checkbox"' in (
        response.text
    )
    assert "submitButton.disabled = !hasSelectedDocument()" in response.text
    assert 'fileInput.addEventListener("change"' in response.text
    assert "form.requestSubmit()" in response.text
    assert 'form.addEventListener("submit", async (event) =>' in response.text
    assert 'response.headers.get("content-type")?.includes("text/event-stream")' in (
        response.text
    )
    assert 'uploadEvent.event === "completed"' in response.text
    assert 'uploadEvent.event === "failed"' in response.text
    assert 'uploadEvent.event === "page_progress"' in response.text
    assert "Осталось обработать страниц:" in response.text
    assert "pagesRemaining.hidden = true" in response.text


def test_upload_form_starts_processing_when_pdf_is_selected():
    with TestClient(app) as client:
        response = client.get("/upload")

    assert response.status_code == 200
    assert 'class="upload-layout"' in response.text
    assert 'id="pdf-file" name="file" type="file"' in response.text
    assert "Выбор файла сразу начнёт загрузку и индексацию." in response.text
    assert '<button class="button button-primary" type="submit">' not in response.text
    assert "<noscript>" in response.text
    assert "form-pending.js" in response.text
    assert "data-vector-upload-form" in response.text
    assert "data-upload-progress" in response.text
    assert 'data-upload-step="extraction"' in response.text
    assert 'data-upload-step="ocr" data-status="pending" hidden' in response.text
    assert 'data-upload-step="embeddings"' in response.text
    assert 'data-upload-step="indexing"' in response.text
    assert 'data-upload-badge data-status="pending"' in response.text
    assert 'data-pages-remaining aria-live="polite" hidden' in response.text


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


def test_documents_page_renders_chat_selection_and_create_form(web_session_factory):
    document_id = add_web_document(web_session_factory)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.get("/documents")

    assert response.status_code == 200
    assert "Выберите документы, чтобы начать по ним чат." in response.text
    assert 'name="document_ids"' in response.text
    assert f'value="{document_id}"' in response.text
    assert 'form="create-chat-form"' in response.text
    assert 'aria-label="Выбрать document.pdf для чата"' in response.text
    assert "Выбрать для чата</span>" not in response.text
    document_name_position = response.text.index('class="document-card-name"')
    assert response.text.index("</label>") < document_name_position
    assert (
        response.text.index("document.pdf", document_name_position)
        > document_name_position
    )
    assert "document-file-extension" not in response.text
    assert 'class="document-page-count">0 стр.</span>' in response.text
    assert 'class="document-status-empty">Нет статусов страниц</span>' in response.text
    assert "document-card-name" in response.text
    assert 'class="document-file-mark"' not in response.text
    assert 'id="create-chat-form"' in response.text
    assert 'action="http://testserver/chats"' in response.text
    assert "Выберите хотя бы один документ, чтобы начать чат." in response.text
    assert 'aria-describedby="document-chat-hint"' in response.text
    assert "Начать чат" in response.text
    assert "Обсудите с AI-ассистентом содержание документа." in response.text


def test_documents_page_chat_form_creates_chat_from_selected_documents(
    web_session_factory,
):
    first_document_id = add_web_document(
        web_session_factory,
        filename="first.pdf",
        file_hash="1" * 64,
    )
    second_document_id = add_web_document(
        web_session_factory,
        filename="second.pdf",
        file_hash="2" * 64,
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            "/chats",
            data={"document_ids": [first_document_id, second_document_id]},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/chats/")
    with web_session_factory() as session:
        from app.models import Chat

        chat = session.query(Chat).one()
        assert {document.id for document in chat.documents} == {
            first_document_id,
            second_document_id,
        }


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
        session.add_all(
            [
                DocumentPage(
                    document_id=document.id,
                    page_number=2,
                    content_hash="c" * 64,
                    status=PageStatus.PARSED,
                    text="Extracted text",
                ),
                DocumentPage(
                    document_id=document.id,
                    page_number=3,
                    content_hash="d" * 64,
                    status=PageStatus.OCR,
                    text="Recognized text",
                ),
            ]
        )
        session.commit()

    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.get("/documents")

    assert response.status_code == 200
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.pdf" in response.text
    assert "<script>alert(1)</script>.pdf" not in response.text
    assert "Ошибка OCR: 1" in response.text
    assert "Распознано OCR: 1" in response.text
    assert "Текст извлечён: 1" in response.text
    assert 'class="status-badge status-badge--error"' in response.text
    assert 'class="status-badge status-badge--ocr"' in response.text
    assert 'class="status-badge status-badge--parsed"' in response.text


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
    assert 'aria-label="contract.pdf"' in listing.text
    assert ">contract.pdf</a>" in listing.text
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
    assert 'aria-label="cli-document.pdf"' in listing.text
    assert ">cli-document.pdf</a>" in listing.text
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
        def __init__(
            self,
            session,
            progress_callback=None,
            page_progress_callback=None,
        ):
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
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"event": "completed"' in response.text
    assert '"document_id": "document-123"' in response.text
    assert '"filename": "valid.pdf"' in response.text
    assert processed_paths
    assert not Path(processed_paths[0]).exists()
    file_hash = hashlib.sha256(b"%PDF-1.7 test").hexdigest()
    assert (storage_directory / f"{file_hash}.pdf").read_bytes() == b"%PDF-1.7 test"
    (storage_directory / f"{file_hash}.pdf").unlink()


def test_upload_without_sse_accept_keeps_html_form_response(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import dependencies, web_routes

    monkeypatch.setattr(
        web_routes,
        "ORIGINAL_DOCUMENT_DIRECTORY",
        tmp_path / "documents",
    )

    class FakeDocumentPipeline:
        def __init__(
            self,
            session,
            progress_callback=None,
            page_progress_callback=None,
        ):
            assert progress_callback is None
            assert page_progress_callback is None

        def process(self, path):
            return "document-html"

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
            headers={"Accept": "text/html"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Документ проиндексирован" in response.text


def test_upload_streams_each_pipeline_stage_and_success(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import dependencies, web_routes

    monkeypatch.setattr(
        web_routes,
        "ORIGINAL_DOCUMENT_DIRECTORY",
        tmp_path / "documents",
    )

    class ProgressPipeline:
        def __init__(
            self,
            session,
            progress_callback=None,
            page_progress_callback=None,
        ):
            self.progress_callback = progress_callback
            self.page_progress_callback = page_progress_callback

        def process(self, path):
            self.page_progress_callback(2, 2)
            for step in ("extraction", "storage", "embeddings", "indexing"):
                self.progress_callback(step, "processing")
                self.progress_callback(step, "completed")
            self.page_progress_callback(1, 2)
            self.page_progress_callback(0, 2)
            return "streamed-document"

    monkeypatch.setattr(
        dependencies,
        "create_document_pipeline",
        ProgressPipeline,
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            "/upload",
            files={"file": ("valid.pdf", b"%PDF-1.7 test", "application/pdf")},
        )

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert response.headers["content-type"].startswith("text/event-stream")
    assert [
        (event.get("step"), event.get("status"))
        for event in events
        if event["event"] == "progress"
    ] == [
        ("upload", "completed"),
        ("extraction", "processing"),
        ("extraction", "completed"),
        ("storage", "processing"),
        ("storage", "completed"),
        ("embeddings", "processing"),
        ("embeddings", "completed"),
        ("indexing", "processing"),
        ("indexing", "completed"),
    ]
    assert events[-1]["event"] == "completed"
    assert [
        (event["remaining_pages"], event["total_pages"])
        for event in events
        if event["event"] == "page_progress"
    ] == [(2, 2), (1, 2), (0, 2)]


def test_upload_processing_continues_after_progress_stream_disconnects(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import dependencies, web_routes

    started = Event()
    release = Event()
    finished = Event()
    processed_paths = []
    storage_directory = tmp_path / "documents"
    monkeypatch.setattr(
        web_routes,
        "ORIGINAL_DOCUMENT_DIRECTORY",
        storage_directory,
    )

    class SlowPipeline:
        def __init__(
            self,
            session,
            progress_callback=None,
            page_progress_callback=None,
        ):
            pass

        def process(self, path):
            processed_paths.append(path)
            started.set()
            assert release.wait(timeout=5)
            finished.set()
            return "disconnected-document"

    monkeypatch.setattr(dependencies, "create_document_pipeline", SlowPipeline)
    application = create_app(session_factory=web_session_factory)
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/upload",
        "headers": [],
        "app": application,
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "query_string": b"",
        "root_path": "",
        "http_version": "1.1",
    }

    async def disconnect_stream():
        response = await web_routes.upload_document(
            Request(scope),
            StarletteUploadFile(
                filename="valid.pdf",
                file=BytesIO(b"%PDF-1.7 test"),
            ),
        )
        iterator = response.body_iterator
        first_event = await iterator.__anext__()
        await iterator.aclose()
        assert await asyncio.to_thread(started.wait, 2)
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        for _ in range(100):
            if not application.state.upload_tasks:
                break
            await asyncio.sleep(0.01)
        return first_event

    first_event = asyncio.run(disconnect_stream())
    assert '"step": "upload"' in first_event

    assert processed_paths
    assert not Path(processed_paths[0]).exists()
    file_hash = hashlib.sha256(b"%PDF-1.7 test").hexdigest()
    assert (storage_directory / f"{file_hash}.pdf").exists()


def test_concurrent_uploads_receive_independent_progress_streams(
    web_session_factory,
    monkeypatch,
    tmp_path,
):
    from app import dependencies, web_routes

    monkeypatch.setattr(
        web_routes,
        "ORIGINAL_DOCUMENT_DIRECTORY",
        tmp_path / "documents",
    )

    class NamedPipeline:
        def __init__(
            self,
            session,
            progress_callback=None,
            page_progress_callback=None,
        ):
            self.progress_callback = progress_callback

        def process(self, path):
            self.progress_callback("extraction", "processing")
            self.progress_callback("extraction", "completed")
            return Path(path).stem

    monkeypatch.setattr(dependencies, "create_document_pipeline", NamedPipeline)
    application = create_app(session_factory=web_session_factory)

    async def upload(filename, content):
        response = await web_routes.upload_document(
            Request(
                {
                    "type": "http",
                    "method": "POST",
                    "path": "/upload",
                    "headers": [],
                    "app": application,
                    "scheme": "http",
                    "server": ("testserver", 80),
                    "client": ("testclient", 50000),
                    "query_string": b"",
                    "root_path": "",
                    "http_version": "1.1",
                }
            ),
            StarletteUploadFile(filename=filename, file=BytesIO(content)),
        )
        return [
            json.loads(line.removeprefix("data: "))
            async for line in response.body_iterator
            if line.startswith("data: ")
        ]

    async def upload_concurrently():
        return await asyncio.gather(
            upload("first.pdf", b"%PDF-1.7 first"),
            upload("second.pdf", b"%PDF-1.7 second"),
        )

    first_events, second_events = asyncio.run(upload_concurrently())
    assert first_events[-1]["event"] == "completed"
    assert first_events[-1]["document_id"] == "first"
    assert first_events[-1]["filename"] == "first.pdf"
    assert second_events[-1]["event"] == "completed"
    assert second_events[-1]["document_id"] == "second"
    assert second_events[-1]["filename"] == "second.pdf"


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
        def __init__(
            self,
            session,
            progress_callback=None,
            page_progress_callback=None,
        ):
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

    assert failed.status_code == 200
    assert '"event": "failed"' in failed.text
    assert "Не удалось обработать документ" in failed.text
    assert "private internal details" not in failed.text
    assert failed_paths
    assert not Path(failed_paths[0]).exists()


@pytest.mark.parametrize("method", ["get", "post"])
def test_removed_ask_routes_return_not_found(method):
    with TestClient(app) as client:
        response = getattr(client, method)(
            "/ask",
            **({"data": {"question": "Каков срок?"}} if method == "post" else {}),
        )

    assert response.status_code == 404


@pytest.mark.parametrize(
    ("path", "data"),
    [
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
    file_hash="d" * 64,
):
    with session_factory() as session:
        document = Document(filename="document.pdf", file_hash=file_hash)
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
    assert "Выберите документы на странице" in response.text
    assert 'href="http://testserver/documents"' in response.text
    assert "Документы для нового чата" not in response.text
    assert "Создать чат" not in response.text
    assert 'name="document_ids"' not in response.text


def test_chats_list_renders_existing(web_session_factory):
    add_web_chat(web_session_factory)
    application = create_app(session_factory=web_session_factory)
    with TestClient(application) as client:
        response = client.get("/chats")

    assert response.status_code == 200
    assert "Test chat" in response.text
    assert '<section class="page-heading chats-page-heading"' in response.text
    assert '<h1 id="page-title">Хранимые чаты</h1>' in response.text
    assert "Создайте чат по выбранным документам на странице" in response.text
    assert "Документы для нового чата" not in response.text
    assert "Создать чат" not in response.text
    assert 'name="document_ids"' not in response.text
    assert "<th>Документы</th>" not in response.text
    assert "<th>Действие</th>" not in response.text
    assert "Удалить чат и все его сообщения?" not in response.text


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
    assert "chat-sidebar" not in chats_response.text
    assert "Чат по договору" in chats_response.text
    assert f'href="http://testserver/chats/{chat_id}"' in chats_response.text
    assert 'class="striped-table"' in chats_response.text
    assert '<aside class="chat-sidebar" aria-label="Чаты">' in chat_response.text
    assert "Чат по договору" in chat_response.text
    assert 'name="chat_ids"' in chat_response.text
    assert "Удалить выбранное" in chat_response.text
    assert 'class="chat-sidebar-link is-active"' in chat_response.text
    assert f'href="http://testserver/chats/{chat_id}"' in chat_response.text
    assert "В процессе" in chat_response.text
    assert "Новый чат" not in chat_response.text
    assert 'placeholder="Выберите чат"' not in chats_response.text
    assert 'aria-label="Отправить вопрос" disabled' not in chats_response.text
    assert 'placeholder="Напишите вопрос..."' in chat_response.text
    assert (
        'data-pending-message="Формируем ответ по выбранным документам..."'
        in chat_response.text
    )
    assert "<textarea" in chat_response.text
    assert "</textarea>" in chat_response.text
    assert '<span aria-hidden="true">▶</span>' in chat_response.text
    assert 'class="chat-aside"' not in chat_response.text
    assert 'class="chat-main"' in chat_response.text


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
        assert chat.title.startswith("first.pdf + second.pdf — ")
        assert chat.title.endswith(chat.created_at.strftime("%Y-%m-%d"))


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


def test_chat_view_renders_document_and_messages(web_session_factory, monkeypatch):
    from app import dependencies

    monkeypatch.setattr(
        dependencies,
        "get_llm_service",
        lambda: SimpleNamespace(model="provider/model-v1"),
    )
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
    assert response.text.count("data-edit-question") == 2
    assert response.text.count("Редактировать вопрос") == 2
    assert response.text.count("Ассистент / provider/model-v1") == 2


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
    composer = second_response.text.split('id="content"', 1)[1].split("</textarea>", 1)[
        0
    ]
    assert composer.endswith(">")
    assert "Использованные страницы" not in second_response.text
    assert 'class="message-sources"' not in history_response.text
    expected_history = [
        "Каков срок договора?",
        "Первый ответ. [стр. 1]",
        "А сколько длится второй?",
        "Второй ответ. [стр. 1]",
    ]
    positions = [history_response.text.index(message) for message in expected_history]
    assert positions == sorted(positions)


def test_send_message_maps_llm_failures_to_safe_guidance(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies, web_routes
    from app.llm import LLMError, LLMErrorCategory
    from app.models import Message, MessageRole

    chat_id = add_web_chat(web_session_factory)
    log_records = []

    class BrokenRAGPipeline:
        def answer_with_sources(self, question, **kwargs):
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
            f"/chats/{chat_id}/messages",
            data={"content": "Каков срок?"},
        )

    assert response.status_code == 200
    assert "ограничил частоту запросов" in response.text
    assert "raw provider body" not in response.text
    assert log_records == [
        (
            "Chat message LLM processing failed",
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
    assert "secret token" not in repr(log_records)
    with web_session_factory() as session:
        messages = (
            session.query(Message)
            .filter_by(chat_id=chat_id, role=MessageRole.USER)
            .all()
        )
        assert all(message.content != "Каков срок?" for message in messages)


def test_send_message_warns_on_unknown_llm_outcome(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies
    from app.llm import LLMError, LLMErrorCategory

    chat_id = add_web_chat(web_session_factory)

    class BrokenRAGPipeline:
        def answer_with_sources(self, question, **kwargs):
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
        response = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "Каков срок?"},
        )

    assert response.status_code == 200
    assert "провайдер мог его обработать" in response.text
    assert "Проверьте лимит расходов" in response.text
    assert "private transport detail" not in response.text


def test_send_message_hides_unexpected_failure_details(
    web_session_factory,
    monkeypatch,
):
    from app import dependencies, web_routes

    chat_id = add_web_chat(web_session_factory)
    log_records = []

    class BrokenRAGPipeline:
        def answer_with_sources(self, question, **kwargs):
            raise RuntimeError("private internal details")

    monkeypatch.setattr(
        dependencies,
        "create_rag_pipeline",
        lambda: BrokenRAGPipeline(),
    )
    monkeypatch.setattr(
        web_routes.logger,
        "error",
        lambda message, *, extra: log_records.append((message, extra)),
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            f"/chats/{chat_id}/messages",
            data={"content": "Вопрос после сбоя"},
        )

    assert response.status_code == 200
    assert "Не удалось получить ответ" in response.text
    assert "private internal details" not in response.text
    assert log_records == [
        (
            "Chat message processing failed",
            {"error_type": "RuntimeError"},
        )
    ]


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
    assert f'>{"a" * 2001}</textarea>' in response.text


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


def test_delete_selected_chats_removes_only_selected_records(web_session_factory):
    selected_chat_ids = [
        add_web_chat(
            web_session_factory,
            document_id=f"selected-{index}",
            file_hash=f"{index + 1:064x}",
        )
        for index in range(2)
    ]
    other_chat_id = add_web_chat(
        web_session_factory,
        document_id="keep-chat",
        file_hash="3" * 64,
    )
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            "/chats/delete-selected",
            data={"chat_ids": selected_chat_ids},
            follow_redirects=False,
        )

    assert response.status_code == 303
    with web_session_factory() as session:
        from app.models import Chat, Message

        assert all(session.get(Chat, chat_id) is None for chat_id in selected_chat_ids)
        assert all(
            session.query(Message).filter_by(chat_id=chat_id).count() == 0
            for chat_id in selected_chat_ids
        )
        assert session.get(Chat, other_chat_id) is not None


def test_delete_selected_chats_without_selection_does_not_delete_anything(
    web_session_factory,
):
    chat_id = add_web_chat(web_session_factory)
    application = create_app(session_factory=web_session_factory)

    with TestClient(application) as client:
        response = client.post(
            "/chats/delete-selected",
            data={},
            follow_redirects=False,
        )

    assert response.status_code == 303
    with web_session_factory() as session:
        from app.models import Chat

        assert session.get(Chat, chat_id) is not None


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
        response_delete_selected = client.post(
            "/chats/delete-selected",
            data={"chat_ids": [chat_id]},
            headers={"Origin": "https://untrusted.example"},
        )

    assert response.status_code == 403
    assert response_delete.status_code == 403
    assert response_delete_selected.status_code == 403


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
