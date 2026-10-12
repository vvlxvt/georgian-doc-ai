# LLM Implementation Queue

## Project inspection

### Existing architecture

- Ingestion and retrieval are separate. `DocumentPipeline` handles PDF
  extraction/OCR and indexing; `Retriever` embeds a query and searches Qdrant;
  `RAGPipeline` passes results through `ContextBuilder` and `AnswerGenerator`.
- `LLMService` uses typed `LLMRequest`/`LLMResponse` contracts and preserves
  `generate(prompt: str) -> str` for compatibility. Ollama and OpenRouter
  adapters use `httpx` with a test seam at `_post`.
- `AnswerGenerator` owns a Russian-language prompt, page-citation instructions,
  and untrusted-source markers. It returns a fixed insufficient-information
  sentence without calling the LLM when the context is empty. It does not
  parse a response schema or validate citations.
- Qdrant results contain chunk IDs and payload metadata including
  `document_id`, `page_id`, `page_number`, `chunk_index`, and text. The context
  currently renders page numbers and text but not stable application-issued
  source identifiers.
- The local FastAPI/Jinja UI has a synchronous `/ask` form, selected-document
  filtering, pending UI feedback, and generic safe failure text. It has no
  JSON API, question history, job-status endpoint, authentication, or
  background task queue.
- SQLite currently stores documents, pages, and chunks only. Schema changes
  use Alembic; the sole existing revision is defensive/idempotent for legacy
  databases. Request-scoped database sessions and SQLAlchemy repositories are
  established patterns.
- Settings use frozen `pydantic-settings` and `.env.example`. OpenRouter is
  the approved default provider; Ollama is available by explicit selection.
  Model capabilities are conservative and explicit configuration values.
- Tests use pytest, fakes, mocked HTTP seams, in-memory SQLite/Qdrant, and an
  opt-in/availability-guarded real-Ollama integration test. The default suite
  is expected to remain deterministic and network-free.

### Reuse and completed requirements

Do not schedule duplicate work for:

- PDF ingestion, OCR fallback, hash-gated reprocessing, document/page/chunk
  persistence, Qdrant indexing, and rollback ordering.
- Existing embedding, vector-store, retrieval, context-builder, RAG, and
  Ollama abstractions.
- Russian answer instructions, page citation format, untrusted-source
  boundary wording, and the empty-context no-LLM behavior.
- Local UI forms, question input validation, document selection, pending
  indication, generic safe error display, and same-origin POST checks.
- General environment-backed settings, dependency injection, migrations,
  pytest fixtures, and existing lint/test commands.

### Missing capabilities

- OpenRouter adapter and explicit provider selection/configuration.
- A provider-neutral request/response/usage/error contract rich enough to
  retain provider request IDs and optional token/cost metadata.
- Application-level error classification, bounded retry policy, and safe
  user-facing mapping.
- Prompt version metadata, structured output parsing, and validation of
  citations against the exact retrieved chunks.
- Optional durable LLM run/result records, retention policy, and persistence
  failure behavior.
- Actionable web error states and observability beyond the current generic
  route-level exception handling.

### Risks and constraints

- OpenRouter calls may incur charges. A timeout after request transmission
  does not prove the provider did not process the request; never blindly retry
  ambiguous outcomes or silently switch provider/model.
- Structured-output support is model-dependent. OpenRouter documents
  `response_format`, but the selected model's actual support must be checked
  and configured explicitly; local parsing/validation remains mandatory.
- Citation-ID validation can prove that a reference exists in the retrieved
  set, but cannot prove semantic support for every generated claim.
- Writing an LLM result to SQLite happens after the paid provider call. A
  persistence failure must not cause an automatic second provider call.
- The current UI call is synchronous. There is no evidence yet that it needs a
  durable job system; do not add one without measured latency/operational need
  and explicit approval.
- The existing web UI is local-only. Any future network exposure requires a
  separate authentication/authorization decision.
- Preserve ingestion, OCR, hash, vector, and existing `/ask` behavior unless a
  reviewed contract change is approved.

### Repository and verification baseline

- Branch: `master`.
- Latest commit at inspection: `9f54922 feat: add local web UI and document-scoped search`.
- Existing user changes left untouched: `data/app.db` and `prompt.txt`.
- Latest recorded verification for the current code: `131 passed, 3 deselected`
  with `pytest -m "not integration"`; Ruff and Black passed. This was recorded
  before this planning document was created and was not rerun for planning.
- Commit convention: descriptive conventional-style commit subjects are
  already used. Each future implementation task should be committed
  separately, without staging pre-existing user files.

### Relevant references

- Project architecture and operation: `README.md`, `AGENTS.md`.
- Existing LLM, prompt, and retrieval path: `app/llm.py`, `app/answer.py`,
  `app/context.py`, `app/retrieval.py`, `app/rag.py`,
  `app/dependencies.py`.
- Existing schema and migration strategy: `app/models.py`,
  `app/repositories.py`, `alembic/versions/20261006_01_add_chunks_and_page_constraint.py`.
- Existing UI integration: `app/web.py`, `app/web_routes.py`,
  `tests/test_web.py`.
- Provider API reference reviewed during planning (2026-10-12):
  [OpenRouter API reference](https://openrouter.ai/docs/api-reference/overview).
  It documents the chat-completions endpoint and optional structured-output
  response format. Re-check current request, error, usage, model-capability,
  and rate-limit documentation immediately before implementing the adapter.

## Architecture proposal

### Component boundaries

1. Keep `Retriever`, `ContextBuilder`, `AnswerGenerator`, `RAGPipeline`, and
   the `/ask` form as the application flow. Keep
   `RAGPipeline.answer(...) -> str` as the existing user-facing contract.
2. Evolve the existing `LLMService` boundary rather than adding a parallel
   provider registry: define small application-owned request/response/usage
   records and typed provider failures, then have the existing Ollama
   implementation and a focused OpenRouter adapter implement that boundary.
3. Select exactly one configured provider in `app/dependencies.py`. Preserve
   Ollama as the default. OpenRouter is opt-in through environment settings.
   Do not add automatic provider/model fallback.
4. Keep prompt rendering and response validation in application code, not in
   the provider adapter. Version the prompt/schema in source or another
   lightweight repository-local mechanism; do not add a prompt-management
   service or database until a demonstrated need exists.
5. Add source IDs derived from retrieved chunk IDs to the model context.
   Require structured output where explicitly supported; otherwise parse the
   same application schema from ordinary text. Validate every cited source ID
   against the retrieval result set and render page citations from trusted
   application metadata, never model-invented document IDs or URLs.
6. Keep request processing synchronous initially. Persisted run records and
   user-visible history/status depend on the product decisions in task 1.
   Reconsider durable jobs only after measuring real response latency and
   confirming operational requirements.

### Data flow

Question and selected document IDs
-> existing RAG retrieval
-> normalized source records (chunk ID, document ID, page, text)
-> empty/insufficient-evidence short circuit
-> versioned prompt and bounded context
-> configured `LLMService`
-> typed provider response and usage metadata
-> parse/schema validation
-> citation-ID and evidence-reference validation
-> application-rendered answer with trusted `[стр. N]` citations
-> optional durable run/result persistence
-> existing server-rendered answer response.

### Error and retry flow

- Retrieval/database/Qdrant errors remain distinguishable from a valid empty
  result. Empty or insufficient evidence does not invoke a provider.
- Provider errors map to stable internal categories: configuration/auth,
  quota/credits, rate limit, connection/timeout, transient server/overload,
  invalid request/context limit, and malformed/truncated response.
- Parser/schema/citation failures are validation errors, not transport errors.
  Do not retry them automatically by default; a future explicitly bounded
  repair request would be a separately costed provider call.
- Persistence failures are reported distinctly and must never trigger another
  LLM request automatically.
- Use at most one retry-policy layer. Retry only explicitly classified,
  demonstrably safe transient failures within attempt and elapsed-time limits.
  Respect a valid `Retry-After`; use bounded exponential backoff with jitter.
  Treat uncertain post-send timeouts and other ambiguous outcomes as
  potentially billable; do not retry them unless provider idempotency behavior
  is documented and the user approves that policy.

### Persistence proposal (approval required)

If run history/audit is required, introduce one `llm_runs` table through a
defensive Alembic revision. Proposed minimum metadata: generated run ID,
status, timestamps/duration, configured provider/model, prompt/schema version,
retrieved source IDs, provider request ID, usage/cost when returned, and a
safe error category. Do not store full prompts, retrieved document text, API
keys, raw provider bodies, or raw model output by default.

Whether to persist the user's question and validated answer, how long to keep
records, and whether to expose a history/status page remain unresolved. Do not
create migration or storage behavior until the user approves these choices.
If approved, persist the initial run state before the provider call and commit
the validated result before reporting durable success. A failed final write
must be surfaced without reissuing a possibly billable call.

### Provider extension strategy

Use the same provider-neutral contract for Ollama and OpenRouter. Provider
adapters own URL, headers, request/response translation, provider error
classification, and request ID/usage extraction. Application orchestration
owns prompts, retries, validation, persistence, and user messages. Record
capabilities explicitly for the selected model; do not assume universal JSON
mode, context size, streaming, or token/cost metadata.

## Progress

- Total tasks: 8
- Completed: 4 (decisions recorded and tasks 2-4 implemented)
- Current task: task 5
- Blocked: none
- Remaining: 3 (task 7 explicitly declined)
- Latest test result: 163 passed, 3 deselected; Ruff and Black passed
- Latest commit: `5468fdb`

## Tasks

### Task 1: Approving persistence, provider, and execution decisions

* Status: `[x]`
* Goal: Resolve product choices that materially affect cost, privacy, and
  application behavior before changing source code or schema.
* Review/requirement: Requirements 5-11 and 16 of `prompt.txt`.
* Dependencies: None.
* Files to inspect: `prompt.txt`, `README.md`, `app/config.py`,
  `app/web_routes.py`, `app/models.py`, `alembic/versions/`.
* Expected changes: Record approved decisions in this queue: OpenRouter is the
  default provider; processing remains synchronous; do not persist questions,
  answers, prompts, retrieved context, or LLM-run records; do not add history
  or job-status UI; validate internal source IDs and render the existing
  `[стр. N]` citations. Do not automatically fall back to another provider.
* Public contracts affected: None until implementation approval.
* Failure scenarios: Unapproved data retention or paid-provider behavior
  could expose private questions or create unexpected charges.
* Tests to add/update: None for the planning decision.
* Verification commands: Review the approved decisions against this queue;
  no code test is applicable.
* Definition of done: The user-approved provider, data-retention, citation,
  and sync-vs-job decisions are recorded below.
* Commit message: `docs(llm): record approved implementation decisions`
* Commit hash: `a2c4cab`

### Task 2: Defining provider-neutral LLM contracts and error taxonomy

* Status: `[x]`
* Goal: Give both provider adapters and the RAG layer stable typed contracts.
* Review/requirement: Requirements 4, 6, and 7 of `prompt.txt`.
* Dependencies: Task 1.
* Files to inspect: `app/llm.py`, `app/answer.py`, `app/rag.py`,
  `app/dependencies.py`, `tests/test_llm.py`, `tests/test_answer.py`.
* Expected changes: Extend/revise the existing `LLMService` contract with
  minimal request, response, optional usage/cost, provider/model/request-ID
  metadata, explicit capability data, and typed application error categories.
  Adapt Ollama and fakes while keeping `RAGPipeline.answer(...) -> str`
  unchanged. Avoid duplicate provider registries or unnecessary base classes.
* Public contracts affected: Internal LLM-service boundary; preserve the
  existing UI/RAG answer string contract.
* Failure scenarios: Missing response content, malformed provider data,
  omitted usage metadata, or capability not supported by the selected model.
* Tests to add/update: Contract conformance for the Ollama fake/adapter;
  metadata-absent and malformed response cases; compatibility tests for
  `AnswerGenerator` and `RAGPipeline`.
* Verification commands:
  `.\venv\Scripts\python.exe -m pytest tests\test_llm.py tests\test_answer.py tests\test_rag.py`
  `.\venv\Scripts\python.exe -m ruff check .`
  `.\venv\Scripts\python.exe -m black --check .`
* Definition of done: Provider-specific payloads are outside the application
  contract, errors have stable categories, and current Ollama-backed behavior
  remains covered.
* Commit message: `feat(llm): define provider-neutral contracts`
* Commit hash: `53d3e2e`

### Task 3: Adding explicit LLM configuration and OpenRouter adapter

* Status: `[x]`
* Goal: Configure the approved default OpenRouter provider without silently
  switching paid services or assuming model capabilities.
* Review/requirement: Requirement 5 of `prompt.txt`; re-verify official
  OpenRouter documentation before implementation.
* Dependencies: Tasks 1 and 2.
* Files to inspect: `app/config.py`, `.env.example`, `app/dependencies.py`,
  `app/llm.py`, `app/preflight.py`, `requirements.txt`, `tests/test_config.py`,
  `tests/test_llm.py`.
* Expected changes: Add environment-backed provider selection, optional
  `OPENROUTER_API_KEY`, configurable API base URL/model, explicit model
  capabilities, and connect/read timeouts. Implement the documented
  chat-completions request/response mapping, authentication header, request-ID
  and usage/cost extraction when present, and provider error classification.
  Keep OpenRouter selected by default and Ollama available explicitly. Do not log credentials or
  request/response bodies. Add an opt-in live smoke test only; never require
  credentials for ordinary tests. Add no dependency unless an implementation
  gap proves one is necessary.
* Public contracts affected: Settings, provider factory, and internal LLM
  response/error contract from task 2.
* Failure scenarios: Missing key, rejected credentials, rate limit, quota
  exhaustion, timeout, malformed response, unsupported structured output,
  and upstream server failure.
* Tests to add/update: Mocked HTTP serialization and parsing, optional headers
  and metadata, all mapped error classes, missing-key behavior, settings and
  `.env.example` parity, explicit provider selection, no accidental live call.
* Verification commands:
  `.\venv\Scripts\python.exe -m pytest tests\test_config.py tests\test_llm.py tests\test_dependencies.py`
  `.\venv\Scripts\python.exe -m ruff check .`
  `.\venv\Scripts\python.exe -m black --check .`
* Definition of done: OpenRouter remains the approved default, both providers
  work through mocked tests, missing configuration fails clearly, and no
  fallback or secret leakage is possible.
* Commit message: `feat(llm): add configurable OpenRouter provider`
* Commit hash: `5468fdb`

### Task 4: Versioning prompts and preparing source-aware context

* Status: `[x]`
* Goal: Make prompt changes reviewable and provide trustworthy source
  references for response validation.
* Review/requirement: Requirements 8 and 9 of `prompt.txt`.
* Dependencies: Tasks 1 and 2.
* Files to inspect: `app/answer.py`, `app/context.py`, `app/retrieval.py`,
  `app/rag.py`, `tests/test_answer.py`, `tests/test_context.py`.
* Expected changes: Give the current prompt/schema an explicit version;
  separate privileged instructions from untrusted evidence using the provider
  contract's message structure where supported, while preserving safe
  delimiters for Ollama. Normalize each retrieved result into an application
  source record with chunk ID, document ID, page number, and text. Assign
  request-local source IDs and include only validated metadata/text in context.
  Keep prompts in source-controlled files/code initially; do not store raw
  rendered prompts.
* Public contracts affected: Internal context and prompt representation only.
* Failure scenarios: Missing metadata, empty source text, prompt injection
  text, context over selected model budget, and prompt/schema version mismatch.
* Tests to add/update: Prompt rendering by version, message separation,
  preserved injection boundary, missing metadata, deterministic source IDs,
  and empty-context short circuit.
* Verification commands:
  `.\venv\Scripts\python.exe -m pytest tests\test_answer.py tests\test_context.py tests\test_rag.py`
  `.\venv\Scripts\python.exe -m ruff check .`
  `.\venv\Scripts\python.exe -m black --check .`
* Definition of done: Every supplied evidence block has stable request-local
  provenance; prompt version is available to diagnostics; retrieved text
  remains untrusted and outside privileged instructions.
* Commit message: `feat(rag): add versioned source-aware prompts`
* Commit hash: recorded after commit

### Task 5: Parsing structured answers and validating citations

* Status: `[ ]`
* Goal: Reject malformed or unsupported model output before presenting it as a
  grounded answer.
* Review/requirement: Requirements 8 and 9 of `prompt.txt`.
* Dependencies: Tasks 2 and 4; task 3 for provider structured-output
  capability handling.
* Files to inspect: `app/answer.py`, `app/context.py`, `app/rag.py`,
  `app/web_routes.py`, `tests/test_answer.py`, `tests/test_rag.py`,
  `tests/test_web.py`.
* Expected changes: Define a small response schema (proposed: answer segments
  with source IDs), parse and validate required fields/types/non-empty answer,
  reject unknown citation IDs, and render page citations using retrieved
  metadata rather than model-invented values. Request structured output only
  where the configured model explicitly supports it; always validate locally.
  Preserve the existing user-facing Russian answer and `[стр. N]` format.
* Public contracts affected: Internal LLM output contract; existing `/ask`
  form still receives a string.
* Failure scenarios: Invalid JSON, truncated/empty output, schema mismatch,
  missing/unknown/duplicate citations, refusal, and insufficient evidence.
* Tests to add/update: Valid output, malformed and truncated output, invalid
  source IDs, missing evidence, refusal handling, and answer escaping.
* Verification commands:
  `.\venv\Scripts\python.exe -m pytest tests\test_answer.py tests\test_rag.py tests\test_web.py`
  `.\venv\Scripts\python.exe -m ruff check .`
  `.\venv\Scripts\python.exe -m black --check .`
* Definition of done: No model-invented document ID, page, or URL is trusted;
  invalid output yields an explicit failure or safe insufficient-evidence
  result. Documentation states that citation membership is not proof of
  semantic entailment.
* Commit message: `feat(rag): validate structured answers and citations`
* Commit hash: pending

### Task 6: Implementing bounded retries and actionable failure mapping

* Status: `[ ]`
* Goal: Make transient failures explicit and retry only when a duplicate paid
  operation is acceptably unlikely.
* Review/requirement: Requirements 6 and 7 of `prompt.txt`.
* Dependencies: Tasks 1-3.
* Files to inspect: `app/llm.py`, `app/config.py`, `app/dependencies.py`,
  `app/web_routes.py`, `tests/test_llm.py`, `tests/test_web.py`.
* Expected changes: Implement one bounded retry policy with attempt and total
  time limits, deterministic jitter injection for tests, and valid
  `Retry-After` handling. Retry only explicitly safe connection failures or
  rejected transient requests according to verified provider semantics.
  Treat ambiguous post-send timeouts as outcome-unknown and do not retry by
  default. Do not retry auth, quota, invalid request, unsupported capability,
  schema/citation validation, or persistence failures. Map internal categories
  to safe, actionable Russian UI messages and structured logs containing
  provider/model, category, attempts, duration, and request ID when available.
* Public contracts affected: Internal typed errors and user-facing failure
  messages; no raw provider detail in the response.
* Failure scenarios: Retry exhaustion, invalid `Retry-After`, auth/quota
  errors, ambiguous provider outcome, and cancellation/total-time limit.
* Tests to add/update: Deterministic backoff/attempt tests, retry/no-retry
  matrix, `Retry-After` parsing, final error category, safe UI mapping, and
  assertions that secrets/prompts/raw provider bodies are absent from logs.
* Verification commands:
  `.\venv\Scripts\python.exe -m pytest tests\test_llm.py tests\test_web.py`
  `.\venv\Scripts\python.exe -m ruff check .`
  `.\venv\Scripts\python.exe -m black --check .`
* Definition of done: Attempts are bounded and observable; no nested retries
  exist; uncertain billable outcomes are not retried automatically.
* Commit message: `feat(llm): add bounded retry and error policy`
* Commit hash: pending

### Task 7: Persisting approved LLM run results

* Status: `[-]` (explicitly declined in task 1; no LLM-run persistence or
  history)
* Goal: Persist only the approved minimum result metadata with transaction and
  retry behavior that does not duplicate provider charges.
* Review/requirement: Requirement 10 of `prompt.txt`.
* Dependencies: Tasks 1, 2, 4, and 5.
* Files to inspect: `app/models.py`, `app/repositories.py`,
  `app/database.py`, `alembic/env.py`,
  `alembic/versions/20261006_01_add_chunks_and_page_constraint.py`,
  `tests/test_migrations.py`, `tests/test_repositories.py`.
* Expected changes: If approved, add one `LLMRun` record/repository and a
  defensive Alembic revision. Use an explicit lifecycle and stable run ID;
  persist validated results only after successful validation; store metadata
  and only the user content/answer fields explicitly approved in task 1. Do
  not persist raw prompts or source text. Define retention/deletion and
  recovery for interrupted or unknown-outcome runs. A retry of persistence
  must reuse the same run ID and never call the provider again.
* Public contracts affected: Database schema/model and possibly an optional
  history endpoint/page if approved.
* Failure scenarios: Database unavailable before the request, provider
  success followed by write failure, transaction rollback, duplicate
  persistence retry, interrupted run, and stale/incomplete status.
* Tests to add/update: Legacy-schema migration, lifecycle transitions,
  idempotent write, rollback, post-provider persistence failure without a
  second provider call, retention behavior, and safe output.
* Verification commands:
  `.\venv\Scripts\python.exe -m pytest tests\test_migrations.py tests\test_repositories.py tests\test_web.py`
  `.\venv\Scripts\alembic.exe upgrade head`
  `.\venv\Scripts\python.exe -m ruff check .`
  `.\venv\Scripts\python.exe -m black --check .`
* Definition of done: The approved artifacts are migrated and persisted
  atomically with stable run identity; existing document/page/chunk rows and
  hash-gated ingestion are unchanged; failure cannot trigger a duplicate
  provider call.
* Commit message: `feat(llm): persist validated processing runs`
* Commit hash: pending

### Task 8: Integrating the pipeline, documenting operation, and final review

* Status: `[ ]`
* Goal: Wire approved components into the existing UI, prove regressions are
  absent, and document operational behavior.
* Review/requirement: Requirements 11-16 of `prompt.txt`.
* Dependencies: Tasks 1-7, excluding task 7 if persistence is explicitly
  declined.
* Files to inspect: `app/dependencies.py`, `app/web_routes.py`, `app/web.py`,
  `README.md`, `AGENTS.md`, `.env.example`, `tests/test_web.py`,
  `tests/test_rag_integration.py`.
* Expected changes: Wire the selected service without changing the current
  form path or CLI; render actionable safe statuses/errors and optional usage
  metadata approved for display. Document provider configuration, cost risk,
  retries, data retention, prompt/schema versioning, capability limits,
  synchronous behavior, and local-only security. Add an opt-in OpenRouter
  smoke test gated by an explicit setting and valid credentials. Do not add a
  job system unless task 1 explicitly approves it.
* Public contracts affected: Existing `POST /ask` server-rendered form only
  unless a separate API contract is approved. No response-format break.
* Failure scenarios: No matching retrieval, each provider error class,
  invalid output, persistence failure, and missing/misconfigured settings.
* Tests to add/update: Full mocked RAG-to-provider flow; web request/error
  contract; existing ingestion, OCR, indexing and CLI regressions; opt-in
  smoke-test skip behavior.
* Verification commands:
  `.\venv\Scripts\python.exe -m pytest -m "not integration"`
  `.\venv\Scripts\python.exe -m ruff check .`
  `.\venv\Scripts\python.exe -m black --check .`
  `.\venv\Scripts\python.exe -m pip check`
  Optional, only when explicitly enabled and credentials are configured:
  `.\venv\Scripts\python.exe -m pytest -m integration`
* Definition of done: All approved contracts and docs agree; full offline
  suite and quality checks pass; live-provider tests remain opt-in; final diff
  contains no secrets or unrelated user changes.
* Commit message: `feat(llm): integrate validated provider pipeline`
* Commit hash: pending

## Important decisions and unresolved questions

Approved decisions:

1. OpenRouter remains the default provider. Do not silently fall back to
   Ollama or another provider.
2. Keep `/ask` synchronous. Do not add background jobs or a job-status API.
3. Do not persist questions, answers, prompts, retrieved document text,
   provider bodies, credentials, or LLM-run records. Do not add result history.
4. Keep the existing Russian answer string and `[стр. N]` citations. Use
   request-local source IDs internally and validate them before rendering
   page-number citations.
5. Treat provider request IDs, token use, and cost as optional metadata;
   missing usage is not a successful-shaped fabricated value.
6. Citation membership does not prove factual entailment.

No product decisions remain open for the implementation queue.

## Failure and recovery matrix

| Failure | Retry default | User outcome | Recovery / data handling |
| --- | --- | --- | --- |
| Qdrant/search unavailable | No LLM retry | Search unavailable | Log safe category; leave document ingestion/status untouched |
| No useful retrieved context | No | Fixed insufficient-information answer | Do not call provider |
| Missing provider configuration / authentication | No | Configuration/authentication action | Do not expose key or response body |
| Quota/credits exhausted | No | Provider account action | No model/provider fallback |
| Rate limit with valid `Retry-After` | Bounded only if response is known rejected and within total budget | Busy/retry-later after exhaustion | Record attempt count and delay |
| Connect failure before request transmission | Bounded, per policy | Temporary service error after exhaustion | Safe retry with jitter |
| Read timeout or connection lost after send | No by default | Outcome uncertain; retry manually only after warning | Avoid duplicate paid requests; retain safe run metadata if approved |
| Provider 5xx/overload | Only when documented rejection is safe | Temporary service error | Bounded attempt budget; never nested |
| Invalid request/context too large | No | Request/context cannot be processed | Correct input or apply reviewed context trimming; no blind retry |
| Malformed/truncated model output | No by default | Answer could not be validated | Do not display as grounded success; preserve evidence IDs if approved |
| Unknown/malformed citation | No | Answer could not be validated | Reject invented references; do not render model-supplied URLs |
| SQLite result persistence failure after provider success | Never re-call provider | Result could not be saved reliably | Reuse run ID for storage recovery; record outcome-unknown if possible |

## Proposed migration and dependencies

- Migration: none for the already-approved queue itself. If task 1 approves
  durable LLM run records, propose a single new defensive Alembic revision
  after reviewing retention and status requirements. Do not change existing
  document, page, or chunk schema.
- Dependencies: none currently proposed. `httpx`, Pydantic via
  `pydantic-settings`, SQLAlchemy, Alembic, and pytest already support the
  planned work. Reassess only if validation or retry requirements cannot be
  met reliably with the existing stack.
- Background infrastructure: none proposed. No Celery, Redis, or other task
  queue without measured need and explicit approval.

## Test and verification strategy

- Keep ordinary tests offline, deterministic, and credential-free.
- Mock the OpenRouter HTTP boundary; test exact request construction, response
  parsing, usage/request-ID extraction, and status/error classification.
- Test prompt versioning, context/provenance generation, empty evidence,
  output parsing, citation validation, and limits of grounding checks.
- Inject clocks, sleep, and randomness for retry-policy tests; verify retry
  count, total elapsed budget, `Retry-After`, no retry for auth/quota/invalid
  output, and no retry for ambiguous post-send outcomes.
- If persistence is approved, test upgrade of legacy databases, successful
  and failed lifecycle transitions, transaction rollback, stable run IDs, and
  no second provider call after a write failure.
- Test existing `/ask` behavior and errors with the fake provider; preserve
  the full ingestion/OCR/indexing/CLI regression suite.
- Run the explicit opt-in provider smoke test only with user-provided
  configuration and credentials. Never print credentials or prompt bodies.

## Proposed commit sequence

1. `docs(llm): record approved implementation decisions`
2. `feat(llm): define provider-neutral contracts`
3. `feat(llm): add configurable OpenRouter provider`
4. `feat(rag): add versioned source-aware prompts`
5. `feat(rag): validate structured answers and citations`
6. `feat(llm): add bounded retry and error policy`
7. `feat(llm): persist validated processing runs` (only if persistence is
   approved; otherwise omit)
8. `feat(llm): integrate validated provider pipeline`

For each implementation task, update this queue before work, add/update tests,
run targeted tests and quality checks, review `git diff --check`, stage only
task files, commit separately, then record the verified commit hash and mark
the task complete. Do not commit, push, or deploy during this planning step.
