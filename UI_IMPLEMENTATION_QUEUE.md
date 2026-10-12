# UI Implementation Queue

## Scope and current-state findings

This queue tracks the additive, server-rendered UI for the existing Georgian
document RAG application. The CLI (`main.py preflight` and `main.py index`)
remains supported and unchanged.

Verified existing capabilities:

- PDF ingestion is provided by `DocumentPipeline.process(path)` and assembled
  by `create_document_pipeline(session)`.
- Questions can be answered by `RAGPipeline.answer(question, ...)`, assembled
  by `create_rag_pipeline()`. It returns the answer text, including citations;
  there is no existing HTTP endpoint or structured source list.
- `Document`, `DocumentPage`, and `DocumentChunk` are stored in SQLite.
- At the time this queue was created, the project used Jinja2, pytest,
  SQLAlchemy, and `httpx`, but had no FastAPI application, router, templates,
  or static UI. This queue adds the UI as an additive interface.
- There is no document-delete operation, user authentication, question-history
  store, or job-status API. The UI must not imply these capabilities exist.

Proposed first-release workflows:

1. Ask a question against the configured Qdrant collection and display the
   Russian answer returned by the existing RAG pipeline.
2. Upload and index a PDF through the existing ingestion pipeline.
3. Browse document records already stored in SQLite; do not imply this listing
   proves that every corresponding vector currently exists in Qdrant.

Question history, document deletion, per-document RAG selection, and public
multi-user access are out of scope unless separately approved.

## Approved implementation decisions

- **Access scope:** Local-only, bound to `127.0.0.1`, without authentication.
  Do not expose it on a LAN or public network until authentication and
  authorization are designed. Mutating form POSTs reject a mismatched `Origin`.
- **Upload limit:** Maximum PDF upload is 50 MiB, enforced while streaming to a
  temporary file and cleaned up after processing.

The user approved both decisions on 2026-10-12.

## Status

- Total: 6
- Completed: 6
- Current: none
- Blocked: 0
- Remaining: 0
- Latest verification: 124 passed, 3 integration tests deselected; Ruff, Black,
  and `pip check` passed. Local browser smoke check returned HTTP 200 for all
  pages and static CSS. The server was stopped after verification.

## Tasks

### 1. [x] Adding the FastAPI application entry point

**Purpose and scope:** Add an ASGI application as an additive interface while
preserving the existing CLI and its behavior. Keep service construction lazy
and use existing dependency factories rather than creating duplicate RAG or
ingestion wiring.

**Relevant existing files:** `main.py`, `app/dependencies.py`,
`app/database.py`, `app/config.py`, `app/preflight.py`, `requirements.txt`.

**Expected changes:** Add an app factory/entry point (proposed: `app/web.py`),
request-scoped SQLAlchemy session handling, router registration, and static
mounting. Add only the necessary web dependencies: FastAPI, an ASGI server, and
multipart parsing support for PDF uploads; Jinja2 and `httpx` are already
present. Document the development bind as loopback only.

**Dependencies:** None. The local-only access decision is approved.

**Requirements:**

- Keep `main.py preflight` and `main.py index` working unchanged.
- Do not eagerly load the embedding model or require all services to be
  available just to import the ASGI app.
- Surface request failures explicitly; do not turn failed operations into
  successful responses.

**Tests:** Add `tests/test_web.py` covering app creation, a basic route response,
session cleanup, and startup/import without loading a real model or making
network calls. Retain and run `tests/test_main.py`.

**Verification commands:**

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_web.py tests\test_main.py
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m black --check .
```

**Definition of done:** The app is importable and testable without external
services, CLI tests still pass, and the documented local ASGI command starts
the server.

**Outcome:** Added an app factory, request-scoped session dependency, static
mount, liveness route, and FastAPI/ASGI/multipart dependencies. `tests/test_web.py`
and `tests/test_main.py` pass without initializing remote services or changing
the CLI.

### 2. [x] Creating the shared page layout and responsive design system

**Purpose and scope:** Establish consistent, accessible page structure and
visual foundations for the approved workflows.

**Relevant existing files:** `README.md`, `AGENTS.md`, task 1 application setup.

**Expected changes:** Add a Jinja base template, reusable navigation/form
partials where needed, static CSS with design tokens and responsive layout, and
the initial home page. Use semantic HTML, Russian user-facing copy, keyboard
focus states, sufficient contrast, and reduced-motion support. Load Google
Fonts only as an optional enhancement with system fallbacks; do not make page
usability depend on an external font service.

**Dependencies:** Task 1.

**Tests:** Verify templates render and extend the base template, common
navigation and metadata exist, static assets resolve, and dynamic template
values are escaped. No browser/accessibility compliance claim without a
corresponding browser or accessibility run.

**Verification commands:**

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_web.py
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m black --check .
```

**Definition of done:** All UI pages can share one layout, assets are mounted
from the app, and the layout remains usable at narrow and wide viewport sizes
by CSS inspection or available browser checks.

**Outcome:** Added a Russian-language shared layout, home page, responsive CSS
tokens/layout/components, keyboard skip link, focus states, and reduced-motion
rules. Templates and static assets render through the ASGI app.

### 3. [x] Listing locally recorded documents

**Purpose and scope:** Provide a read-only view of documents recorded in SQLite,
including a useful empty state.

**Relevant existing files:** `app/models.py` (`Document`, `DocumentPage`,
`PageStatus`), `app/database.py`, `tests/conftest.py`.

**Expected changes:** Add a page route and template that query real database
records. Display only fields that exist, such as filename, creation time, page
count, and available page-processing statuses. Use a page-scoped SQLAlchemy
session. Do not add delete controls or suggest that local metadata guarantees
remote Qdrant vector health.

**Dependencies:** Tasks 1 and 2.

**Tests:** Cover populated and empty listings, ordering/pagination if
implemented, database failure handling, and HTML escaping of filenames.

**Verification commands:**

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_web.py
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m black --check .
```

**Definition of done:** The page uses persisted database data, reports the
empty state accurately, and does not introduce unsupported document actions.

**Outcome:** Added paginated read-only document listing with actual page
processing-status counts, an empty state, filename escaping, and a notice that
SQLite metadata does not prove vectors are present in Qdrant.

### 4. [x] Uploading and indexing PDF documents

**Purpose and scope:** Expose the existing ingestion workflow through a
server-rendered PDF upload form.

**Relevant existing files:** `app/pipeline.py`, `app/pdf.py`,
`app/dependencies.py`, `app/preflight.py`, `app/models.py`, `main.py`.

**Expected changes:** Add a POST workflow that validates the upload server-side,
streams it to a uniquely named temporary PDF, calls
`create_document_pipeline(session).process(path)`, and removes the temporary
file in all outcomes. Reuse the established preflight/ingestion behavior where
appropriate. Add the approved upload size limit; never trust the browser MIME
type or uploaded filename as a filesystem path. Show success only after
processing returns successfully. Report actionable errors without exposing
stack traces, internal paths, credentials, or service response secrets.

**Dependencies:** Tasks 1 and 2; upload limit and access-scope decisions must
be confirmed.

**Tests:** Use test PDFs and fake pipelines/services to cover valid upload,
invalid/empty/non-PDF input, size boundary, ingestion failure, temporary-file
cleanup, and success only after backend confirmation. Ensure tests do not call
live Qdrant, Ollama, Tesseract, or embedding downloads.

**Verification commands:**

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_web.py tests\test_pipeline.py
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m black --check .
```

**Definition of done:** Valid PDFs use the existing pipeline and are visible in
the document list; rejected uploads do not reach ingestion; temporary files are
cleaned up on success and failure.

**Outcome:** Added PDF upload with a 50 MiB streamed-copy limit, basename-only
temporary paths, extension/signature/empty-file validation, cleanup, explicit
failure responses, and same-origin checks for POST forms. Tests use a fake
pipeline and verify the temporary file is removed.

### 5. [x] Asking questions against indexed documents

**Purpose and scope:** Add a Russian-language question form that uses the
existing RAG pipeline and displays its grounded answer and citations.

**Relevant existing files:** `app/rag.py`, `app/dependencies.py`,
`app/answer.py`, `app/retrieval.py`, `app/context.py`, `tests/test_rag.py`.

**Expected changes:** Add a server-rendered question submission route calling
`create_rag_pipeline().answer(question)`. Validate and bound question input on
the server, preserve the entered question after recoverable errors, and show
the exact returned answer without inventing extra sources or citations. Keep
question history out of scope because none is persisted by the backend.

**Dependencies:** Tasks 1 and 2.

**Tests:** Cover valid submission with a fake RAG pipeline, blank and
overlong questions, backend failure, answer escaping, and rendering of an
insufficient-information answer. No live model/network calls in ordinary tests.

**Verification commands:**

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_web.py tests\test_rag.py tests\test_answer.py
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m black --check .
```

**Definition of done:** Submitted questions use the existing RAG service and
render only its actual answer; validation and service failures are visible and
do not appear as successful answers.

**Outcome:** Added a Russian question form calling the existing RAG pipeline,
with server-side empty/length validation, same-origin POST checking, answer
escaping, insufficient-information rendering, and a non-success response on
backend failure. No question history or fabricated source list is shown.

### 6. [x] Documenting and verifying the complete UI

**Purpose and scope:** Integrate the workflows, update project documentation,
and verify compatibility with the existing CLI and suite.

**Relevant existing files:** `README.md`, `AGENTS.md`, `requirements.txt`,
`main.py`, all UI routes/templates/tests from tasks 1–5.

**Expected changes:** Document UI setup, local run command, routes/workflows,
configuration or upload limit, test commands, and limitations. Update the
layout map and dependency notes in `AGENTS.md`. Keep `.env.example` in sync if
new settings are introduced. Review the final diff for secrets, debug code,
unrelated changes, and inaccurate claims.

**Dependencies:** Tasks 1–5.

**Tests and verification commands:**

```powershell
.\venv\Scripts\python.exe -m pytest -m "not integration"
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m black --check .
.\venv\Scripts\python.exe -m uvicorn app.web:app --host 127.0.0.1 --port 8000
```

The server command is a manual smoke check; verify the home, document, upload,
and question pages where implemented. Stop the server after verification.

**Definition of done:** Documentation matches the delivered UI, all queue items
are verified, the non-integration suite and quality checks pass, the existing
CLI remains compatible, and the final report distinguishes automated checks
from any manual/browser checks.

**Outcome:** Updated `README.md` and `AGENTS.md` with the UI run command,
loopback-only access warning, routes, upload limit, architecture and test
commands. Added pinned FastAPI, Uvicorn, multipart parsing and their resolved
runtime dependencies. Final checks: 124 non-integration tests passed and 3
integration tests were deselected; Ruff, Black and `pip check` passed. Uvicorn
ran on `127.0.0.1`; `/`, `/documents`, `/upload`, `/ask`, `/health`, and
`/static/css/main.css` each returned HTTP 200. Browser snapshots were checked
for the home, document list, upload and question pages. The server was stopped.
