# Persistent Document Chat Implementation Queue

## Status

- Total: 8
- Done: 6
- In progress: 1
- Blocked: 0
- Current: CHAT-07

## Scope and inspected architecture

This queue plans a local, single-user MVP with one document per conversation. Existing `/ask` remains unchanged as a one-shot workflow; chat adds stored conversations and follow-up questions.

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
3. Follow-up query rewriting is not currently a RAG feature. The initial implementation should decide whether to form a bounded retrieval query from recent messages or add a provider-neutral rewrite operation. Any rewrite must preserve the original question, keep history out of evidence, and fall back to the original query if rewriting fails. Do not send unbounded conversation history.
4. Existing document deletion can remove vectors before the database commit. Conversation deletion must be covered by ORM cascades and the new migration; test SQLite foreign-key behavior because relying on ORM cascades and relying on database cascades differ.
5. LLM calls are synchronous and may be billable. Persisting the user message before a provider call means failed calls leave a user message without an assistant answer; this is intentional conversation history behavior unless product direction chooses a different transaction policy. Never add unbounded retries or log stored message content.
6. UI remains local-only on `127.0.0.1`; the new mutating routes should follow the existing Origin check and server-rendered Jinja patterns.

## Open decisions to resolve before dependent implementation

- Should history-aware retrieval use direct, bounded concatenation of recent turns or an additional LLM query-rewrite call? Recommended first MVP: bounded recent turns used only to produce a retrieval query via existing provider-neutral LLM capability, with fallback to the original question; verify an extra call's cost/latency is acceptable before implementing. If not, ship original-question-only retrieval and defer rewriting.
- Which source fields should be persisted/displayed: page number only, or document ID plus page number and source/chunk IDs? Recommended: document ID, page number, and chunk ID/source ID; never persist retrieved source text as chat metadata.
- On LLM failure, keep the user message and show a retry/new-message affordance, or remove it? Recommended: retain it and show the error without fabricating an assistant message.

## Tasks

### CHAT-01 — Add conversation and message persistence schema

- **ID:** CHAT-01
- **Title:** Add conversation and message models and migration
- **Goal:** Persist one-document conversations and ordered user/assistant messages in the existing SQLite database.
- **Status:** DONE
- **Dependencies:** None
- **Scope:** Add `Conversation` and `Message` models, role constraints, timestamps, indexes, foreign keys, ORM cascade behavior, and a new defensive Alembic revision. Ensure deleting a document cascades conversations/messages and deleting a conversation cascades messages. Use UTC timestamps and a `New conversation` title default. Confirm SQLite FK enforcement behavior before relying on database-level cascades.
- **Minimal tests:** Extend migration tests for legacy upgrade and table/index/FK shape; extend the in-memory model tests to assert conversation/message ordering and deletion cascades for conversation and document.
- **Acceptance criteria:** Migration upgrades the current schema; models create a conversation tied to one document; messages retain role/content/source metadata and stable chronological order; deleting either parent leaves no associated child rows; prior document/page/chunk data remain intact.

### CHAT-02 — Implement conversation repository

- **ID:** CHAT-02
- **Title:** Add repository operations for conversations and messages
- **Goal:** Keep chat persistence queries outside route and orchestration code.
- **Status:** DONE
- **Dependencies:** CHAT-01
- **Scope:** Add a repository for conversation creation/listing/get-with-messages/update/delete, message creation, and bounded retrieval of recent messages. Make message order deterministic using timestamp plus ID or an explicit sequence constraint.
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
- **Title:** Add application service for sending conversation messages
- **Goal:** Orchestrate bounded follow-up questions scoped to the conversation's document.
- **Status:** DONE
- **Dependencies:** CHAT-02, CHAT-03
- **Scope:** Add a chat service that validates conversation/document, stores the user message, loads a configured bounded number of recent turns, derives a retrieval query according to the approved open-decision choice, calls existing RAG with exactly the associated `document_id`, then stores assistant text and validated source references. Keep history separate from evidence and treat it as untrusted. Preserve original question. If rewriting is selected and fails, use the original question. Keep user message on LLM failure and return a typed/error result without persisting a fake assistant response. Do not persist prompts, retrieved text, or model raw responses.
- **Minimal tests:** Unit tests with fake repository/RAG/LLM collaborators for success, document scoping, history limit, follow-up query, rewrite fallback if implemented, source persistence, missing/deleted conversation, and provider failure behavior.
- **Acceptance criteria:** One service owns chat orchestration; retrieval is restricted to the document; history is bounded and cannot override system instructions; answer sources are exactly those validated by the RAG path; failure semantics match the approved decision.

### CHAT-05 — Add conversation web routes

- **ID:** CHAT-05
- **Title:** Add server-rendered conversation endpoints
- **Goal:** Expose conversation list/create/view/send/delete workflows through the existing FastAPI application.
- **Status:** DONE
- **Dependencies:** CHAT-02, CHAT-04
- **Scope:** Add GET/POST `/conversations`, GET `/conversations/{id}`, POST `/conversations/{id}/messages`, and DELETE `/conversations/{id}` using current request session dependency and thin route handlers. Validate selected document IDs and message length, handle not-found cases, preserve safe provider error mapping, and apply same-origin checks to mutating requests. Keep `/ask` routes unchanged.
- **Minimal tests:** Extend `tests/test_web.py` for routes, validation, missing/deleted records, service error handling, Origin rejection, and `/ask` regression.
- **Acceptance criteria:** All listed operations work with the repository/service; invalid or deleted IDs return safe 404/422 behavior; no raw provider detail/message text is logged; existing `/ask` tests continue to pass.

### CHAT-06 — Add basic Jinja chat interface

- **ID:** CHAT-06
- **Title:** Build conversation list and chat pages
- **Goal:** Let a local user create and continue document-scoped conversations in the existing UI.
- **Status:** DONE
- **Dependencies:** CHAT-05
- **Scope:** Add templates following the current base layout and style conventions for conversation list/create, message history, send form, source page references, and conversation deletion. Start conversations from an existing document selection; do not add JS framework or build tooling. Ensure answer text is escaped and errors are Russian and actionable.
- **Minimal tests:** Extend web tests to assert conversation list, document association, persisted history after a fresh GET, source page display, deletion controls, and HTML escaping.
- **Acceptance criteria:** UI supports all MVP flows; reopening a conversation displays persisted messages and citations; source links/labels identify the selected document and page; templates use existing shared layout and remain local-only.

### CHAT-07 — Verify deletion, migration, and existing workflow regressions

- **ID:** CHAT-07
- **Title:** Verify cascade deletion and compatibility end to end
- **Goal:** Protect document deletion, chat persistence, and current CLI/one-shot RAG behavior.
- **Status:** TODO
- **Dependencies:** CHAT-01, CHAT-05, CHAT-06
- **Scope:** Add focused integration-level tests using in-memory SQLite and fake vector/RAG services for deleting a conversation, deleting its document, and continuing to list/use other documents. Review ordering of Qdrant deletion and SQL commit in current document deletion workflow after cascades are added. No new infrastructure.
- **Minimal tests:** New or extended web/repository tests; run migration tests, all non-integration tests, CLI tests, and existing `/ask` regression tests.
- **Acceptance criteria:** No orphaned conversations/messages after either deletion; Qdrant cleanup remains correct; CLI and `/ask` tests pass; migrations upgrade and downgrade without losing legacy data on upgrade.

### CHAT-08 — Document implemented chat MVP

- **ID:** CHAT-08
- **Title:** Update README with conversation workflow and limitations
- **Goal:** Keep user setup and behavior documentation aligned with the shipped MVP.
- **Status:** TODO
- **Dependencies:** CHAT-06, CHAT-07
- **Scope:** Document creating a conversation from a document, document-scoped retrieval, persistence of conversation/message text and source references, schema migration command, follow-up/history limit, provider/cost behavior, local-only access, and MVP limitations. Keep config docs synchronized; no new configuration unless the approved history/rewrite design requires it.
- **Minimal tests:** No new application tests; verify documentation commands and configuration names against the implementation.
- **Acceptance criteria:** README accurately describes available chat routes/UI, stored fields, setup/migration, and known limits; no unimplemented capability is presented as available.

## Approval gate

Phase 1 is complete. Implementation tasks remain `TODO` until the user explicitly approves Phase 2. Before implementation, resolve the open decisions above and reconcile the requested persistence/migration scope with the explicit approval gate in `AGENTS.md`.


