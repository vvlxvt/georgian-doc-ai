# UI Implementation Prompt

## Objective

Design and, after approval, implement a maintainable, responsive server-rendered
UI for the application's real document workflows using FastAPI page routes,
Jinja2 templates, semantic HTML, separate CSS, and small amounts of vanilla
JavaScript where useful.

## Project-specific constraint

The application provides both the CLI and a server-rendered FastAPI/Jinja2 UI.
The current web workflows are overview, SQLite-backed document listing,
PDF upload/indexing, and RAG question/answer. Preserve the existing CLI and
reuse the existing ingestion and RAG services. Inspect the actual routes and
templates before extending them; do not describe unsupported workflows as
available or add other frameworks without a demonstrated need and approval.

## Planning gate

Before implementation:

1. Inspect the current application, services, models, configuration, tests,
   and any existing templates or static assets.
2. Determine which user workflows can be supported by existing functionality.
   Record missing backend capabilities rather than inventing them.
3. Propose a concise `UI_IMPLEMENTATION_QUEUE.md` with scoped tasks,
   dependencies, expected files, tests, verification commands, and definitions
   of done.
4. Stop and wait for approval of the implementation plan before modifying
   application source.

Do not create speculative tasks or architecture. Work through an approved queue
one task at a time, update its status after verification, and stop to report
blockers that need a product or architecture decision.

## Implementation requirements

- Reuse existing services and domain logic; page handlers should orchestrate
  backend operations, not duplicate them.
- Keep templates, styles, and browser scripts separate. Use a shared Jinja base
  template and reusable partials where useful; adapt paths to existing
  conventions.
- Render only real backend data. Do not add demo records, fake progress, or
  success responses that the backend has not confirmed.
- Use server-side validation as the security boundary and preserve existing API
  contracts. Report missing backend capabilities for approval before expanding
  backend scope.
- Use Jinja auto-escaping and safe DOM APIs. Do not expose credentials or
  private configuration in rendered pages or browser scripts. Follow existing
  authentication, authorization, and CSRF protections where applicable.
- Keep the interface responsive and keyboard-usable. Use semantic HTML, visible
  focus, associated form labels, adequate contrast, and reduced-motion support.
- Provide relevant populated, empty, validation-error, backend-error, loading,
  and success states. Display success only after backend confirmation.
- Add dependencies only when existing project capabilities cannot meet a
  demonstrated need.

## Verification and documentation

Use the project's existing test, formatting, and linting conventions. Add tests
for implemented routes, rendering, escaping, forms, errors, and compatibility
with existing backend contracts as applicable. Mock external services in
ordinary tests.

Update the existing documentation with how to run and test the UI, its routes
and templates, and any limitations or missing backend capabilities. Do not
claim browser testing, accessibility compliance, or production readiness unless
those were actually verified.

When the approved work is complete, verify every queue item, run the applicable
full test and quality checks, and report:

- Implemented pages and workflows
- Existing backend operations reused
- Files and dependencies changed
- Commands run and their actual results
- Known limitations and unimplemented work
- Final queue status
