# Persistent Document Chat Implementation Queue

## Status

- Total: 8
- Done: 8
- In progress: 1
- Blocked: 0
- Current: none

## Scope and inspected architecture

This queue describes the local, single-user MVP. Each chat can include one or
more documents. Existing `/ask` remains unchanged as a one-shot workflow; chats
add persisted message history and follow-up questions.

Reusable components confirmed in the repository:

- `Document`, `DocumentPage`, `DocumentChunk` SQLAlchemy models and Alembic revision `20261006_01`.
- `DocumentChunkRepository`, the existing `get_session` request dependency, and the in-memory SQLite test fixture.
- `RAGPipeline.answer(..., document_id=...)`, `Retriever`, Qdrant document filters, `ContextBuilder.build_context()` and `BuiltContext.sources`.
- `AnswerGenerator` returns an `AnswerResult` string with model metadata and validates page citations against the retrieved `SourceRecord` pages.
- `LLMService.call(LLMRequest)` is the provider-neutral interface. Its prompt contract supports distinct system and user prompts. Rewriting a query would require another LLM call and is not currently exposed as a dedicated RAG operation.
- FastAPI/Jinja pages in `app/web_routes.py`, shared layout and forms, plus request-scoped SQLAlchemy sessions.
- Document deletion is currently orchestrated in the web route: delete Qdrant vectors, then delete the SQLAlchemy document. ORM relationships cascade from `Document` to pages and from pages to chunks, but the database FK definitions do not currently specify `ON DELETE CASCADE`.

## Architectural constraints and risks

1. `AGENTS.md` currently says not to persist questions, answers, prompts, or retrieval context, and says migrations require explicit approval. This requested feature requires retaining user and assistant message text, plus one migration. No implementation is authorized until the user approves Phase 2 with this persistence scope in view.
2. RAG already validates citations before returning an answer, while source records are only available as `BuiltContext.sources` inside `AnswerGenerator`. Persisting source references needs a small explicit return contract (for example, a structured chat result) or an injected callback/result method; do not reimplement citation parsing or bypass `AnswerGenerator` validation.
3. Follow-up retrieval uses a bounded chronological window of recent user/assistant messages concatenated with the current question; no extra LLM query-rewrite call is made. The original question is passed separately to answer generation, with history marked as untrusted context for resolving references only. Retrieved chunks from the chat's selected documents remain the only evidence for answer claims.
4. Existing document deletion can remove vectors before the database commit. Chat deletion must be covered by ORM cascades and the new migration; test SQLite foreign-key behavior because relying on ORM cascades and relying on database cascades differ.
5. LLM calls are synchronous and may be billable. Persisting the user message before a provider call means failed calls leave a user message without an assistant answer; this is intentional chat history behavior unless product direction chooses a different transaction policy. Never add unbounded retries or log stored message content.
6. UI remains local-only on `127.0.0.1`; the new mutating routes should follow the existing Origin check and server-rendered Jinja patterns.

## Open decisions to resolve before dependent implementation

- Which source fields should be persisted/displayed: page number only, or document ID plus page number and source/chunk IDs? Recommended: document ID, page number, and chunk ID/source ID; never persist retrieved source text as chat metadata.
- On LLM failure, keep the user message and show a retry/new-message affordance, or remove it? Recommended: retain it and show the error without fabricating an assistant message.

## Tasks

### CHAT-01 — Add chat and message persistence schema

- **ID:** CHAT-01
- **Title:** Add chat and message models and migration
- **Goal:** Persist chats scoped to one or more documents and ordered user/assistant messages in SQLite.
- **Status:** DONE
- **Dependencies:** None
- **Scope:** Add `Chat` and `Message` models, role constraints, timestamps, indexes, foreign keys, ORM cascade behavior, and a new defensive Alembic revision. Ensure deleting a document cascades chats/messages and deleting a chat cascades messages. Use UTC timestamps and a `New chat` title default. Confirm SQLite FK enforcement behavior before relying on database-level cascades.
- **Minimal tests:** Extend migration tests for legacy upgrade and table/index/FK shape; extend the in-memory model tests to assert chat/message ordering and deletion cascades for chat and document.
- **Acceptance criteria:** Migration upgrades the current schema; models create a chat tied to selected documents; messages retain role/content/source metadata and stable chronological order; deleting a chat removes its messages while deleting a document removes only its chat association; prior document/page/chunk data remain intact.

### CHAT-02 — Implement chat repository

- **ID:** CHAT-02
- **Title:** Add repository operations for chats and messages
- **Goal:** Keep chat persistence queries outside route and orchestration code.
- **Status:** DONE
- **Dependencies:** CHAT-01
- **Scope:** Add a repository for chat creation/listing/get-with-messages/update/delete, message creation, and bounded retrieval of recent messages. Make message order deterministic using timestamp plus ID or an explicit sequence constraint.
- **Minimal tests:** Repository tests for create/list/get, document association, message ordering, recent-message limit, metadata update, and delete.
- **Acceptance criteria:** All required persistence operations are exposed by the repository; message history is ordered deterministically and recent-history retrieval is bounded; route code need not issue chat-specific SQL.

### CHAT-03 — Return validated RAG sources for chat

- **ID:** CHAT-03
- **Title:** Expose validated answer sources through the existing RAG path
- **Goal:** Let chat persist and display source references without duplicating retrieval, context building, or citation validation.
- **Status:** TODO
- **Dependencies:** None
- **Scope:** Extend the existing RAG/answer result contract to return the already validated answer and corresponding `SourceRecord` references. Preserve `RAGPipeline.answer()` string-compatible behavior for `/ask` and existing callers, or provide a structured method alongside it. Include only source metadata needed for citation references; do not persist source text or prompts.
- **Minimal tests:** Extend answer/RAG unit tests to assert validated page/source metadata is returned, invalid citations still raise the existing validation error, insufficient-context behavior stays unchanged, and existing string behavior remains compatible.
- **Acceptance criteria:** Chat can obtain answer plus validated retrieved source references from the existing RAG orchestration; existing `/ask` behavior and validation remain intact; no second citation parser or retrieval implementation is introduced.

### CHAT-04 — Implement chat orchestration and follow-up retrieval

- **ID:** CHAT-04
- **Title:** Add application service for sending chat messages
- **Goal:** Orchestrate bounded follow-up questions scoped to the chat's selected documents.
- **Status:** DONE
- **Dependencies:** CHAT-02, CHAT-03
- **Scope:** Add a chat service that validates chat/document, stores the user message, loads a configured bounded number of recent turns, derives a retrieval query from those turns plus the current question, calls existing RAG with exactly the associated document IDs, then stores assistant text and validated source references. Keep history separate from evidence and treat it as untrusted. Preserve the original question for answer generation. Keep the user message on LLM failure and return a typed/error result without persisting a fake assistant response. Do not persist prompts, retrieved text, or model raw responses.
- **Minimal tests:** Unit tests with fake repository/RAG/LLM collaborators for success, document scoping, history limit, follow-up query, rewrite fallback if implemented, source persistence, missing/deleted chat, and provider failure behavior.
- **Acceptance criteria:** One service owns chat orchestration; retrieval is restricted to the selected documents; history is bounded and cannot override system instructions; answer sources are exactly those validated by the RAG path; failure semantics match the approved decision.

### CHAT-05 — Add chat web routes

- **ID:** CHAT-05
- **Title:** Add server-rendered chat endpoints
- **Goal:** Expose chat list/create/view/send/delete workflows through the existing FastAPI application.
- **Status:** DONE
- **Dependencies:** CHAT-02, CHAT-04
- **Scope:** Add GET/POST `/chats`, GET `/chats/{id}`, POST `/chats/{id}/messages`, and DELETE `/chats/{id}` using current request session dependency and thin route handlers. Validate selected document IDs and message length, handle not-found cases, preserve safe provider error mapping, and apply same-origin checks to mutating requests. Keep `/ask` routes unchanged.
- **Minimal tests:** Extend `tests/test_web.py` for routes, validation, missing/deleted records, service error handling, Origin rejection, and `/ask` regression.
- **Acceptance criteria:** All listed operations work with the repository/service; invalid or deleted IDs return safe 404/422 behavior; no raw provider detail/message text is logged; existing `/ask` tests continue to pass.

### CHAT-06 — Add basic Jinja chat interface

- **ID:** CHAT-06
- **Title:** Build chat list and chat pages
- **Goal:** Let a local user create and continue chats scoped to selected documents in the existing UI.
- **Status:** DONE
- **Dependencies:** CHAT-05
- **Scope:** Add templates following the current base layout and style conventions for chat list/create, message history, send form, source page references, and chat deletion. Start chats from one or more existing documents; do not add JS framework or build tooling. Ensure answer text is escaped and errors are Russian and actionable.
- **Minimal tests:** Extend web tests to assert chat list, document association, persisted history after a fresh GET, source page display, deletion controls, and HTML escaping.
- **Acceptance criteria:** UI supports all MVP flows; reopening a chat displays persisted messages and citations; source links/labels identify the selected document and page; templates use existing shared layout and remain local-only.

### CHAT-07 — Verify deletion, migration, and existing workflow regressions

- **ID:** CHAT-07
- **Title:** Verify cascade deletion and compatibility end to end
- **Goal:** Protect document deletion, chat persistence, and current CLI/one-shot RAG behavior.
- **Status:** DONE
- **Dependencies:** CHAT-01, CHAT-05, CHAT-06
- **Scope:** Add focused integration-level tests using in-memory SQLite and fake vector/RAG services for deleting a chat, deleting its document, and continuing to list/use other documents. Review ordering of Qdrant deletion and SQL commit in current document deletion workflow after cascades are added. No new infrastructure.
- **Minimal tests:** New or extended web/repository tests; run migration tests, all non-integration tests, CLI tests, and existing `/ask` regression tests.
- **Acceptance criteria:** No orphaned chats/messages after either deletion; Qdrant cleanup remains correct; CLI and `/ask` tests pass; migrations upgrade and downgrade without losing legacy data on upgrade.

### CHAT-08 — Document implemented chat MVP

- **ID:** CHAT-08
- **Title:** Update README with chat workflow and limitations
- **Goal:** Keep user setup and behavior documentation aligned with the shipped MVP.
- **Status:** TODO
- **Dependencies:** CHAT-06, CHAT-07
- **Scope:** Document creating a chat from selected documents, selected-document retrieval, persistence of chat/message text and source references, schema migration command, follow-up/history limit, provider/cost behavior, local-only access, and MVP limitations. Keep config docs synchronized; no new configuration unless the approved history/rewrite design requires it.
- **Minimal tests:** No new application tests; verify documentation commands and configuration names against the implementation.
- **Acceptance criteria:** README accurately describes available chat routes/UI, stored fields, setup/migration, and known limits; no unimplemented capability is presented as available.

## Approval gate

Phase 1 is complete. Implementation tasks remain `TODO` until the user explicitly approves Phase 2. Before implementation, resolve the open decisions above and reconcile the requested persistence/migration scope with the explicit approval gate in `AGENTS.md`.
