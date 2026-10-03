# Repository Guidelines

## Project Structure

`app/` contains the FastAPI service, domain models, selection and answering services, and SQLite persistence. `tests/` contains the regression suite and the explicit offline acceptance audit. `design/` contains the numbered architecture and handoff records; `.runtime/` contains local audit evidence and isolated test artifacts.

Keep new design documents under `design/` and preserve the numeric prefix. Keep secrets in `.env`; never print or commit `.env` contents. Existing `.runtime/jev_context.sqlite3` is user data and must be preserved.

## Build and test

Use the repository Python environment when available:

```powershell
python -m pytest tests -q
python -X utf8 tests/audit_acceptance.py
python -X utf8 tests/audit_frontend_followup.py
python -X utf8 tests/audit_lifecycle.py
python -X utf8 tests/audit_proof_boundaries.py
python -m compileall -q app
```

The acceptance audit uses temporary databases and blocks external calls. Real smoke scripts may be run only when configured credentials are present; report connectivity separately from answer quality.

## Code style

Use UTF-8 Python and Markdown, type annotations, short functions, and explicit status/error values. SQLite schema changes must be additive migrations that work with existing databases. Close connections on success and failure. Do not weaken or skip an existing failing test.

## Scope

This repository implements a local experimental bench. Do not add real RAG, QASPER, production deployment, exact tokenizers, live pricing, or unrelated UI redesign without a new scoped task.
