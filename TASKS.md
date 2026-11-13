# Document Processing Progress Tasks

Implementation plan for transient upload progress. Processing states are held
only in request-local memory and are never persisted.

| ID | Name | Goal | Changes | Acceptance criteria | Tests | Status |
|---|---|---|---|---|---|---|
| DP-01 | Inspect ingestion pipeline | Map upload and processing boundaries before changing behavior. | Review project analysis, upload route, PDF reader, ingestion/indexing services, and related tests. | Sync/thread behavior and real stage boundaries are documented in this plan. | Existing upload and pipeline tests reviewed. | DONE |
| DP-02 | Design event transport | Choose a request-scoped transport without persistent jobs or infrastructure. | Stream SSE events from each upload response; run processing independently in a thread and retain no state after completion. | Concurrent uploads have isolated progress; disconnecting a reader does not cancel processing. | Backend stream/disconnect tests. | DONE |
| DP-03 | Report real pipeline stages | Emit stage changes from PDF extraction, OCR, SQLite writes, embedding, and Qdrant indexing. | Add optional callbacks to existing pipeline components; preserve CLI defaults and persistence order. | Events originate at actual operations; OCR emits only when used. | Focused pipeline/indexing callback tests. | DONE |
| DP-04 | Stream upload progress | Expose validated upload and terminal result events over the upload response. | Move request-owned upload temp data and DB session into an in-memory worker; close resources on success/failure. | SSE ends on completion/failure; errors do not expose internal exception details. | Upload event sequence, failure, isolation, and disconnect tests. | DONE |
| DP-05 | Add upload Stepper UI | Replace the informational upload aside with live per-upload stages. | Render pending/processing/completed/failed UI and consume streamed events in the browser. | Current stage is explicit, OCR is hidden unless used, terminal states stop updates. | Template/static-script assertions and frontend behavior tests. | DONE |
| DP-06 | Document and verify | Update user documentation and run targeted/full checks. | Describe live-only progress and its limitations; run relevant tests, Ruff, and Black. | Existing upload/storage behavior remains intact; required test suite passes. | Upload, pipeline, and non-integration suite; lint/format checks. | DONE |
