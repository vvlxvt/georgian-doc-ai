# English and Georgian Document RAG

RAG application for extracting English and Georgian PDF documents and answering grounded questions in Russian.

Source documents may be in English, Georgian, or contain both languages. Answers and prompt text are Russian, and all code and documentation is English.

## Architecture

Ingestion and query paths are separate:

- **Ingestion**: `app/pipeline.py` reads each PDF page (`app/pdf.py`). If its native text layer contains at least `MIN_TEXT_LENGTH` characters, that text is used directly regardless of whether it is English or Georgian. Shorter or missing native text falls back to Tesseract OCR (`app/ocr.py`), configured with both `kat` (Georgian) and `eng` (English) by default. Tesseract recognizes both languages together, including mixed-language pages; the application does not detect a single page language or choose one OCR language dynamically. The extracted text is normalized (`app/text.py`), chunked by characters (`app/chunking.py`), stored in SQLite (`app/models.py`, `app/repositories.py`), and indexed in Qdrant (`app/indexing.py`, `app/qdrant_store.py`).
- **Query**: `app/rag.py` embeds the question (`app/embeddings.py`), searches Qdrant (`app/retrieval.py`), validates and wraps hits as untrusted source blocks with request-local IDs (`app/context.py`), and asks the configured LLM provider for a Russian answer with `[стр. N]` citations (`app/answer.py`, `app/llm.py`). Prompt instructions are versioned and separated from evidence where the provider supports separate messages. OpenRouter is selected by default; Ollama can be selected explicitly.

External services sit behind abstract interfaces (`EmbeddingService`, `VectorStore`, `OCRProcessor`, `LLMService`) and are constructed once per process in `app/dependencies.py`.

The project provides both the command-line tools in `main.py` and a
server-rendered web UI using FastAPI and Jinja2. The UI reuses the existing
ingestion and RAG services and preserves the CLI.

## Local setup

1. Install dependencies:

   ```powershell
   .\venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and adjust service URLs or model names. For a
   remote Qdrant instance, set `QDRANT_URL` to its endpoint and set
   `QDRANT_API_KEY` if the instance requires authentication. Keep the API key
   in `.env` or another secret store; do not commit it.
3. Start Qdrant and configure the selected LLM provider. The default OpenRouter provider requires `OPENROUTER_API_KEY`; to use Ollama, set `LLM_PROVIDER=ollama`, start Ollama, and install the configured model. Install Tesseract language data for `kat` and `eng` to OCR Georgian and English scans.
4. Apply migrations:

   ```powershell
   .\venv\Scripts\alembic.exe upgrade head
   ```

5. Check all runtime dependencies, including the database schema, before ingestion:

   ```powershell
   .\venv\Scripts\python.exe main.py preflight
   ```

6. Index a document:

   ```powershell
   .\venv\Scripts\python.exe main.py index data\input\georgian_scan.pdf
   ```

Re-running `index` on the same file is cheap: pages whose content hash is unchanged are skipped, so OCR is not repeated. A PDF with a usable text layer is extracted directly; scanned pages or pages with less than `MIN_TEXT_LENGTH` characters of native text use OCR. A long but poor-quality or garbled native text layer can pass the current length check and therefore avoid OCR.

## Web UI

Install the project dependencies, apply migrations, and make sure Qdrant,
the selected LLM provider, and Tesseract are configured. Start the UI bound
to loopback:

```powershell
.\venv\Scripts\python.exe -m uvicorn app.web:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/> in a browser. Do not change the host to
`0.0.0.0`: this first version has no user authentication and is intended only
for local access. State-changing form requests with a mismatched `Origin` are
rejected.

The UI provides:

- `/` — overview.
- `/documents` — records stored in SQLite. This page does not verify whether
  corresponding vectors are present in Qdrant. The filename links to the
  original PDF when available in `data/documents/` or as a matching file in
  `data/input/`; each document can also be deleted with its pages, chunks, and
  Qdrant vectors. Select documents here to start a chat scoped to them.
- `/upload` — PDF upload and indexing through the existing document pipeline.
  Files must have a `.pdf` extension and PDF signature; maximum size is 50 MiB.
- `/health` — web-process liveness only; it does not replace `main.py preflight`.

Uploaded files are streamed to a temporary file, indexed, and then stored as
`data/documents/<sha256>.pdf`; the submitted filename is never used as a
directory path. These stored originals are served inline from the local-only
`/documents` page. Existing CLI-indexed documents can also link to a matching
PDF in `data/input/`; the file's SHA-256 is verified before it is served.
Documents indexed from elsewhere do not retain their source path and need to
be uploaded through the UI to make the original available. Deleting a document
removes its SQLite records, matching Qdrant vectors, and hash-named uploaded
original; matching files in `data/input/` are kept. Chat pages display answers
with their validated source references.
Answers use ordinary Russian text with `[стр. N]` page citations. The
application checks that cited page numbers occur in the retrieved results and
that each detected sentence has a citation. This membership check does not
prove that a source factually supports the associated claim.

Chat message processing is synchronous: each message request waits for
retrieval, provider completion, and answer validation. User messages and
answers are persisted in SQLite; prompts, retrieved text, and raw provider
responses are not. The selected provider still receives request content and
applies its own retention and data-handling terms. The prompt is versioned in `app/answer.py`
(`answer-v3`). The model is asked for ordinary text rather than a fixed JSON
schema because OpenRouter's automatic model selection may return varying
formats. Empty output, missing page citations, and citations to pages absent
from the retrieved results are rejected. This does not validate factual
support or guarantee that every citation actually substantiates its sentence.

## Configuration

All operational settings are environment-backed; see `.env.example` for the full list with defaults.

| Variable                                | Default                     | Notes                                                                |
| --------------------------------------- | --------------------------- | -------------------------------------------------------------------- |
| `DATABASE_URL`                          | `sqlite:///data/app.db`     | SQLAlchemy URL                                                       |
| `QDRANT_URL`                            | `http://localhost:6333`     | Remote instances can use their HTTPS endpoint                        |
| `QDRANT_API_KEY`                        | _(empty)_                   | Optional; set for authenticated remote Qdrant instances              |
| `QDRANT_COLLECTION`                     | `documents`                 |                                                                      |
| `EMBEDDING_MODEL`                       | `BAAI/bge-m3`               |                                                                      |
| `EMBEDDING_SIZE`                        | `1024`                      | Must match the Qdrant collection                                     |
| `LLM_PROVIDER`                          | `openrouter`                | `openrouter` or `ollama`; there is no provider fallback              |
| `LLM_MAX_ATTEMPTS`                      | `2`                         | Total attempts, including the first; range 1-3                       |
| `LLM_TOTAL_TIMEOUT`                     | `180.0`                     | Overall request budget in seconds; maximum 300                       |
| `OLLAMA_URL`                            | `http://localhost:11434`    |                                                                      |
| `OLLAMA_MODEL`                          | `llama3.2:3b`               |                                                                      |
| `OPENROUTER_URL`                        | `https://api.openrouter.ai` | Base URL; the adapter uses `/api/v1/chat/completions`                |
| `OPENROUTER_API_KEY`                    | _(empty)_                   | Required when `LLM_PROVIDER=openrouter`; keep it secret              |
| `OPENROUTER_MODEL`                      | `gpt-4o-mini`               | Model identifier sent to OpenRouter                                  |
| `OPENROUTER_CONNECT_TIMEOUT`            | `10.0`                      | Seconds                                                              |
| `OPENROUTER_READ_TIMEOUT`               | `120.0`                     | Seconds for read and write; pool timeout follows the connect timeout |
| `OPENROUTER_SUPPORTS_STRUCTURED_OUTPUT` | `false`                     | Set true only when supported by the configured model                 |
| `OPENROUTER_SUPPORTS_JSON_MODE`         | `false`                     | Set true only when supported by the configured model                 |
| `OPENROUTER_SUPPORTS_STREAMING`         | `false`                     | Set true only when supported by the configured model                 |
| `TESSERACT_LANGUAGES`                   | `kat+eng`                   | Tesseract OCR languages joined with `+`; install each trained data file |
| `CHUNK_MAX_LENGTH`                      | `1000`                      | Characters, not tokens                                               |
| `CHUNK_OVERLAP`                         | `100`                       | Words of overlap between chunks                                      |
| `MIN_TEXT_LENGTH`                       | `20`                        | Native text shorter than this many characters falls back to OCR      |

`OPENROUTER_SUPPORTS_STRUCTURED_OUTPUT` and `OPENROUTER_SUPPORTS_JSON_MODE`
describe configured provider capabilities. The current answer pipeline requests
plain text and does not send a response schema.

Changing `EMBEDDING_MODEL` or `EMBEDDING_SIZE` requires a compatible Qdrant collection and re-indexing all documents. `QdrantVectorStore` refuses to start against a collection whose vector size or distance does not match, so the mismatch surfaces at startup rather than at indexing time.

On startup, the application also ensures keyword payload indexes for `document_id` and `page_id`, which are required for filtered retrieval and stale-vector cleanup.


## Chat

The web UI provides persistent multi-document chats. Chats and messages are
stored in SQLite, and each chat can be associated with one or more documents.
Follow-up questions retrieve context only from the selected documents' indexed
chunks. Questions are limited to chats with selected documents; there is no
one-shot search across the entire library.

### Chat workflows

1. **Create a chat:** On the Documents page, select one or more documents and
   start a chat, or create one from the Chats page. Its search scope is limited
   to those documents.
2. **Send a message:** Open the chat, type your question in Russian,
   and submit. The answer includes `[стр. N]` citations backed by the
   retrieved chunks.
3. **View history:** Reopening a chat displays all persisted user and
   assistant messages with their source page references.
4. **Delete a chat:** Deleting a chat removes it and all its
   messages. Its source documents remain in the library.

### Routes

| Route | Method | Description |
|---|---|---|
| `/chats` | GET | List all chats with their documents |
| `/chats` | POST | Create a chat from selected documents (`document_ids` form fields) |
| `/chats/{id}` | GET | View a chat with message history |
| `/chats/{id}/messages` | POST | Send a message in a chat (`content` form field) |
| `/chats/delete-selected` | POST | Delete selected chats and their messages (`chat_ids` form fields) |
| `/chats/{id}/delete` | POST | Delete a chat and all its messages |

### Constraints

- **Local-only:** The UI binds to `127.0.0.1` and must not be exposed to a
  network without authentication and authorization.
- **No authentication:** The initial chat MVP has no login or access control.
- **History limit:** Retrieval uses a bounded window of recent turns; older
  history is not sent to the LLM. Recent turns help resolve follow-up questions
  and form the retrieval query, capped at 6,000 characters. Only
  selected-document chunks are evidence for answers.
- **Citations only:** Stored source references contain document ID, page
  number, and chunk ID — never the retrieved source text itself.
- **No automatic retries:** LLM failures leave the user message persisted
  without a fabricated assistant answer.

### Limitations

- New chat titles use the selected document filename(s) and creation date
  (`YYYY-MM-DD`); multi-document names are joined in selection order.
- Deleting a document removes its association with chats but preserves
  their messages and any associations with other documents. Qdrant vectors are
  removed before the SQL commit.

## Migrations

Schema changes go through Alembic; `Base.metadata.create_all()` is never used for production schema evolution.

```powershell
.\venv\Scripts\alembic.exe upgrade head
.\venv\Scripts\alembic.exe downgrade base
.\venv\Scripts\alembic.exe current
```

The initial revision is idempotent: it inspects existing tables and indexes, so it upgrades an older `data/app.db` that predates `document_chunks` and the `(document_id, page_number)` unique constraint.

## Tests

```powershell
.\venv\Scripts\python.exe -m pytest -m "not integration"
.\venv\Scripts\python.exe -m pytest -m integration
```

The default suite needs no Qdrant, Ollama, Tesseract, or model download: it uses an in-memory SQLite engine, `QdrantClient(":memory:")`, and fakes. Tests marked `integration` require a running Ollama with the configured model and skip themselves cleanly when it is unavailable.

The web routes are covered by `tests/test_web.py`; run them with:

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_web.py
```

The two `test_embeddings.py` tests that load the real `BAAI/bge-m3` model are opt-in via `RUN_REAL_MODEL_TESTS=1`. Loading that model on Windows can abort the interpreter with `Windows fatal exception: access violation` inside `torch`, which is a hard crash rather than a Python exception, so the tests must skip before the model is touched:

```powershell
$env:RUN_REAL_MODEL_TESTS = "1"
.\venv\Scripts\python.exe -m pytest -m integration
```

Lint and format:

```powershell
.\venv\Scripts\python.exe -m ruff check .
.\venv\Scripts\python.exe -m black --check .
```

## Operational notes

- SQLite and Qdrant are not updated atomically. Ingestion writes the new chunks and vectors first and only removes the previous chunk rows once indexing succeeds; on failure it rolls the session back and restores the previous vectors.
- `main.py preflight` validates the schema, Qdrant, the selected LLM provider configuration, and Tesseract language data. For Ollama it checks the configured model; for OpenRouter it checks that the API key is configured without sending a live or potentially billable request.
- OpenRouter requests may incur provider charges. The application never switches providers or models automatically; preflight checks only that a key is configured, not whether it is valid or funded.
- LLM calls retry only safe connection-establishment failures and HTTP 429 responses, within the attempt and total-time limits. `Retry-After` is honored when valid. Read/write timeouts, connection loss after transmission, other HTTP statuses, and answer-validation failures are not retried; an unknown outcome may mean the provider processed the request.
- Chat message failures map typed provider errors to safe Russian guidance and log provider, model, error category, attempts, elapsed time, and the provider request ID when available. The application never logs the API key, question, retrieved text, prompt, or model response.

The real OpenRouter smoke test is intentionally opt-in and may incur charges.
Run it only when an OpenRouter key is configured and you explicitly want a
live provider request:

```powershell
$env:RUN_OPENROUTER_TESTS = "1"
.\venv\Scripts\python.exe -m pytest tests\test_rag_integration.py -m integration -k openrouter
Remove-Item Env:RUN_OPENROUTER_TESTS
```
