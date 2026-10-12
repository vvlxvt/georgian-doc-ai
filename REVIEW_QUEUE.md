# Review Queue

## Status
- Total: 9
- Completed: 9
- Current: none
- Blocked: 0

## Tasks

### 1. [x] Add database migrations and enforce one page per document/page number
**Review:** "No schema migrations"; the current local database lacks `document_chunks`, and the additional review noted that `(document_id, page_number)` needs a uniqueness constraint to prevent duplicate pages during concurrent processing.

**Changes:** Introduce Alembic migrations for the existing SQLite schema, including creation of `document_chunks`, and add a database-level unique constraint for `DocumentPage.document_id` plus `DocumentPage.page_number`. Define the upgrade path for the existing `data/app.db`; do not rely on `Base.metadata.create_all()` for production schema evolution.

**Files:** `alembic.ini`, `alembic/env.py`, `alembic/versions/*`, `app/models.py`, `app/database.py`, dependency configuration, and documentation for applying migrations.

**Tests:** Add a migration test that upgrades a database created by the old schema and verifies `document_chunks` exists; add a model/repository test proving a duplicate page number for the same document is rejected; preserve in-memory test schema setup where appropriate.

**Verification:** Run migrations against a copy of the existing local database, inspect the resulting tables and constraints, and run the database/repository test suite.

### 2. [x] Validate existing Qdrant collection compatibility
**Review:** Additional review comment: Qdrant does not verify vector size or distance for an already existing collection, so changing the embedding model can fail only at indexing time.

**Changes:** When the collection already exists, read its vector configuration and fail with a clear configuration error unless its vector size and distance match the configured embedding service. Decide and document the migration/reindexing procedure when the embedding model changes.

**Files:** `app/qdrant_store.py`, `app/config.py`, Qdrant-store tests, and operational documentation.

**Tests:** Add tests using in-memory Qdrant for a compatible existing collection and for incompatible vector size/distance; retain the collection-creation test.

**Verification:** Execute the Qdrant-store tests and manually start with an intentionally incompatible collection to confirm a readable startup failure.

### 3. [x] Connect document ingestion to vector indexing with failure recovery
**Review:** "Chunks never reach Qdrant" and "Vector indexing disabled"; the `ChunkIndexer` dependency and calls are commented out, so uploaded documents are never searchable.

**Changes:** Inject `ChunkIndexer` into `DocumentPipeline`; after creating and flushing replacement chunks, index them in Qdrant. Define a consistent failure/retry strategy for the non-atomic SQLite/Qdrant boundary, including cleanup or a persisted state that enables retry. Preserve stale-vector removal only after successful replacement indexing.

**Files:** `app/pipeline.py`, `app/indexing.py`, `app/dependencies.py` or a new ingestion factory, repository/model code if an indexing status is persisted, and pipeline/indexing tests.

**Tests:** Add an end-to-end ingestion test with fake embeddings and an in-memory vector store that processes a PDF/page, then retrieves the new chunk. Add tests for re-indexing a changed page, Qdrant/indexing failure, and retry/cleanup behavior.

**Verification:** Process a supplied Georgian PDF into SQLite and Qdrant, query for a known passage, and confirm that retrieved metadata includes the correct document ID, page, and chunk text.

### 4. [x] Make application services process-scoped rather than per request
**Review:** "Embedding model should be process-scoped"; `create_rag_pipeline()` constructs `HuggingFaceEmbeddingService` each time and can repeatedly load `SentenceTransformer`.

**Changes:** Establish an application lifecycle that initializes the embedding model, Qdrant client/store, and LLM service once per process and reuses them. Keep construction testable through dependency injection and provide clear startup errors for unavailable external services.

**Files:** `app/dependencies.py`, application entry point when introduced, `app/config.py`, and dependency tests.

**Tests:** Add tests proving repeated pipeline resolution reuses the embedding service and that test dependencies can substitute fakes without loading a real model.

**Verification:** Start the application/service twice within one process, confirm a single model initialization through a fake or instrumentation, and run dependency tests without network/model downloads.

### 5. [x] Harden answer generation for Russian, citations, and untrusted context
**Review:** "Russian answers and citations are not enforced"; the prompt does not require Russian output or page citations and interpolates arbitrary document text as instructions.

**Changes:** Replace the prompt contract to require Russian-only answers, cite source pages for material statements, state that the model may use only the supplied context, and delimit untrusted retrieved text so that instructions contained in documents are ignored. Return the insufficient-information response in Russian. Ensure the context format retains source metadata needed for citations.

**Files:** `app/answer.py`, `app/context.py`, `app/rag.py` if structured sources are returned, `tests/test_answer.py`, `tests/test_context.py`, and RAG tests.

**Tests:** Assert Russian response instructions, citation format, context delimiters, insufficient-information wording, and that a prompt-injection-like chunk is represented as source data rather than executable instruction.

**Verification:** Use a fake LLM to inspect the generated prompt, then run an integration/evaluation case with a Georgian source and Russian question and verify Russian output with the correct page citation.

### 6. [x] Make the external Ollama integration test reliable and meaningful
**Review:** "External integration test is weak and environment-bound"; it requires a real local Ollama but is unmarked and only asserts that the answer is non-empty.

**Changes:** Mark the test as integration, skip it with a clear reason when Ollama or its configured model is unavailable, and assert grounded expected content plus a page citation rather than only a non-empty string. Keep deterministic RAG behavior under fake LLM tests.

**Files:** `pyproject.toml`, `tests/test_rag_integration.py`, `tests/test_rag.py`, shared test fixtures, and test-run documentation.

**Tests:** Add marker registration and availability checks; strengthen the real-service assertion; add or retain a deterministic equivalent with a fake LLM for normal CI.

**Verification:** Run the default suite on a machine without Ollama and confirm a clean skip; run `pytest -m integration` where Ollama and the model are available and confirm grounded assertions pass.

### 7. [x] Restore coverage for long-sentence chunking
**Review:** "Nested test is never collected"; `test_long_sentence_is_split_by_words` is nested inside `test_overlap` and pytest does not collect it.

**Changes:** Move the test to module scope and add edge cases relevant to Georgian legal text, including very long tokens and a chunk-length boundary. Decide whether chunk length is intentionally measured in characters or should become token-based, then document and test that choice.

**Files:** `tests/test_chunking.py`, potentially `app/chunking.py` and developer documentation if chunking semantics change.

**Tests:** Ensure pytest collects the moved test; add boundary cases for a long sentence, a word longer than `max_length`, overlap limits, and Georgian punctuation/text.

**Verification:** Run `pytest tests/test_chunking.py -vv` and confirm the long-sentence case appears as an independently collected test and all chunk size invariants hold.

### 8. [x] Document and validate the operational RAG configuration
**Review:** The reviewed configuration hardcodes local service endpoints and model settings; together with the migration and external-service findings, this leaves deployments unable to reproduce or diagnose startup behavior.

**Changes:** Move operational settings to environment-backed configuration with documented defaults, add an `.env.example`, and provide startup/preflight validation for Qdrant, Ollama, the configured model, and Georgian Tesseract language data. Document local development, migration, indexing, and integration-test commands.

**Files:** `app/config.py`, `app/ocr.py`, `app/database.py`, application entry point, `.env.example`, `README.md`, and configuration/OCR tests.

**Tests:** Add settings parsing/override tests and preflight tests using mocks for each unavailable dependency; do not make unit tests require running external services.

**Verification:** Start from a clean environment using only the documented setup, run the preflight check, apply migrations, index a sample PDF, and execute the non-integration test suite.

**Outcome:** Operational settings are env-backed with documented defaults in `.env.example`, and ingestion tuning (`CHUNK_MAX_LENGTH`, `CHUNK_OVERLAP`, `MIN_TEXT_LENGTH`) was moved out of hardcoded constants. `validate_runtime()` now also checks the database schema, so an unmigrated database fails at preflight with "run 'alembic upgrade head' first" instead of mid-ingestion. `main.py index <pdf>` provides the documented indexing entry point and assembles its collaborators through `create_document_pipeline()`. `README.md` documents setup, configuration, migrations, indexing, and both test commands.

Verified against a clean database: fresh schema rejected by preflight, `alembic upgrade head` creating all three tables plus the unique `(document_id, page_number)` index, preflight then passing, and indexing `data/input/georgian_scan.pdf` producing 4 pages and 37 retrievable chunks whose payloads carry `document_id`, `page_id`, `page_number`, `chunk_index`, and `text`. Re-running the same file skipped all 4 pages and added no chunks.

Fixed alongside this item: `PDFPage` required a `_doc_path` argument that `PDFReader.pages()` never passed, which made every real ingestion call raise `TypeError`; the vestigial field was removed. Also moved the real-model tests in `tests/test_embeddings.py` behind the `integration` marker and a `RUN_REAL_MODEL_TESTS=1` opt-in (loading `BAAI/bge-m3` aborts the interpreter with a `torch` access violation, a hard crash that cannot be caught, so the skip must precede model construction), removed a module-level live Ollama call from `tests/test_llm.py` in favour of the new `OllamaLLMService._post` seam, pinned pytest's base temp directory into `.tmp/pytest` so an ACL-denied `%TEMP%` cannot fail the suite, deleted three dead scripts from `tests/`, and cleared the remaining ruff findings. `pytest -m "not integration"` is 100 passed with `ruff` and `black` clean, and `pytest -m integration` passes its real-Ollama RAG case.

### 9. [x] Support remote Qdrant endpoints and API keys
**Changes:** Keep the local endpoint as the default while allowing a remote Qdrant URL and optional API key from the environment. Pass the configured credentials through application wiring and preflight, and document remote setup without committing secrets.

**Files:** `app/config.py`, `app/dependencies.py`, `app/preflight.py`, Qdrant configuration/preflight tests, `.env.example`, and `README.md`.

**Verification:** Configuration tests assert the environment overrides; dependency and preflight tests assert that the remote URL and API key reach the Qdrant client.
