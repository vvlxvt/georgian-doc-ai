# Georgian Document RAG

RAG prototype for extracting Georgian PDF documents and answering grounded questions in Russian.

Source documents are Georgian, answers and prompt text are Russian, and all code and documentation is English.

## Architecture

Ingestion and query paths are separate:

- **Ingestion**: `app/pipeline.py` reads a PDF (`app/pdf.py`), uses the native text layer when a page has enough text and falls back to Tesseract OCR (`app/ocr.py`), normalizes it (`app/text.py`), chunks it by characters (`app/chunking.py`), stores pages and chunks in SQLite (`app/models.py`, `app/repositories.py`), and indexes the chunks in Qdrant (`app/indexing.py`, `app/qdrant_store.py`).
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
3. Start Qdrant and configure the selected LLM provider. The default OpenRouter provider requires `OPENROUTER_API_KEY`; to use Ollama, set `LLM_PROVIDER=ollama`, start Ollama, and install the configured model. Install Tesseract language data for `kat` and `eng`.
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

Re-running `index` on the same file is cheap: pages whose content hash is unchanged are skipped, so OCR is not repeated.

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
- `/documents` — read-only records stored in SQLite. This page does not verify
  whether corresponding vectors are present in Qdrant.
- `/upload` — PDF upload and indexing through the existing document pipeline.
  Files must have a `.pdf` extension and PDF signature; maximum size is 50 MiB.
- `/ask` — submit a question to the existing RAG pipeline and display its
  Russian answer. Select one or more documents to limit the search; with no
  documents selected, the search includes all documents.
- `/health` — web-process liveness only; it does not replace `main.py preflight`.

Uploaded files are streamed to a temporary file, removed after processing, and
never use the submitted filename as a directory path. Question history and
document deletion are not provided. The answer page displays the answer string
from RAG and does not independently reconstruct source references.
Answer source IDs are checked against retrieved results and page citations are
rendered from their metadata; this membership check does not prove that a
source factually supports the associated claim.

## Configuration

All operational settings are environment-backed; see `.env.example` for the full list with defaults.

| Variable | Default | Notes |
| --- | --- | --- |
| `DATABASE_URL` | `sqlite:///data/app.db` | SQLAlchemy URL |
| `QDRANT_URL` | `http://localhost:6333` | Remote instances can use their HTTPS endpoint |
| `QDRANT_API_KEY` | *(empty)* | Optional; set for authenticated remote Qdrant instances |
| `QDRANT_COLLECTION` | `documents` | |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | |
| `EMBEDDING_SIZE` | `1024` | Must match the Qdrant collection |
| `LLM_PROVIDER` | `openrouter` | `openrouter` or `ollama`; there is no provider fallback |
| `LLM_MAX_ATTEMPTS` | `2` | Total attempts, including the first; range 1-3 |
| `LLM_TOTAL_TIMEOUT` | `180.0` | Overall request budget in seconds; maximum 300 |
| `OLLAMA_URL` | `http://localhost:11434` | |
| `OLLAMA_MODEL` | `llama3.2:3b` | |
| `OPENROUTER_URL` | `https://api.openrouter.ai` | Base URL; the adapter uses `/api/v1/chat/completions` |
| `OPENROUTER_API_KEY` | *(empty)* | Required when `LLM_PROVIDER=openrouter`; keep it secret |
| `OPENROUTER_MODEL` | `gpt-4o-mini` | Model identifier sent to OpenRouter |
| `OPENROUTER_CONNECT_TIMEOUT` | `10.0` | Seconds |
| `OPENROUTER_READ_TIMEOUT` | `120.0` | Seconds for read and write; pool timeout follows the connect timeout |
| `OPENROUTER_SUPPORTS_STRUCTURED_OUTPUT` | `false` | Set true only when supported by the configured model |
| `OPENROUTER_SUPPORTS_JSON_MODE` | `false` | Set true only when supported by the configured model |
| `OPENROUTER_SUPPORTS_STREAMING` | `false` | Set true only when supported by the configured model |
| `TESSERACT_LANGUAGES` | `kat+eng` | Languages joined with `+` |
| `CHUNK_MAX_LENGTH` | `1000` | Characters, not tokens |
| `CHUNK_OVERLAP` | `100` | Words of overlap between chunks |
| `MIN_TEXT_LENGTH` | `20` | Below this a page goes to OCR |

Changing `EMBEDDING_MODEL` or `EMBEDDING_SIZE` requires a compatible Qdrant collection and re-indexing all documents. `QdrantVectorStore` refuses to start against a collection whose vector size or distance does not match, so the mismatch surfaces at startup rather than at indexing time.

On startup, the application also ensures keyword payload indexes for `document_id` and `page_id`, which are required for filtered retrieval and stale-vector cleanup.

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
- LLM calls retry only safe connection-establishment failures and HTTP 429 responses, within the attempt and total-time limits. `Retry-After` is honored when valid. Read/write timeouts, connection loss after transmission, other HTTP statuses, and answer-validation failures are not retried; an unknown outcome may mean the provider processed the request.
- `/ask` maps typed provider errors to safe Russian guidance and logs provider, model, error category, attempts, elapsed time, and the provider request ID when available. It never logs the API key, question, retrieved text, prompt, or model response.
