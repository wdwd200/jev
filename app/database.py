from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from .config import settings
from .models import AnswerRun, Chunk, Comparison, Document, Question, Selection, Workspace

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS workspaces (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, total_tokens INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT, chunk_set_id TEXT, chunk_rule_snapshot TEXT NOT NULL DEFAULT '{}', chunk_rule_fingerprint TEXT, token_count_method TEXT NOT NULL DEFAULT 'legacy');
CREATE TABLE IF NOT EXISTS chunk_sets (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id), created_at TEXT NOT NULL, status TEXT NOT NULL, rule_snapshot TEXT NOT NULL DEFAULT '{}', rule_fingerprint TEXT, token_count_method TEXT NOT NULL DEFAULT 'legacy');
CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id), filename TEXT NOT NULL, content BLOB NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, extraction_ok INTEGER NOT NULL, error TEXT);
CREATE TABLE IF NOT EXISTS chunks (id TEXT PRIMARY KEY, chunk_set_id TEXT NOT NULL, document_id TEXT NOT NULL REFERENCES documents(id), ordinal INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, version TEXT NOT NULL, kind TEXT NOT NULL DEFAULT 'text', source TEXT NOT NULL DEFAULT '{}', image BLOB);
CREATE TABLE IF NOT EXISTS questions (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL REFERENCES workspaces(id), text TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, rewritten TEXT NOT NULL DEFAULT '', rewrite_status TEXT NOT NULL DEFAULT 'original', rewrite_usage TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS selections (id TEXT PRIMARY KEY, question_id TEXT NOT NULL REFERENCES questions(id), workspace_id TEXT NOT NULL, strategy TEXT NOT NULL, strategy_version TEXT NOT NULL, status TEXT NOT NULL, selected_chunks TEXT NOT NULL, total_tokens INTEGER NOT NULL, usage TEXT NOT NULL, reason TEXT, threshold REAL, budget INTEGER, model_name TEXT, elapsed_ms INTEGER, chunk_set_id TEXT, rule_params TEXT NOT NULL DEFAULT '{}', prompt_version TEXT, input_rule_snapshot TEXT NOT NULL DEFAULT '{}', input_rule_fingerprint TEXT, input_token_count_method TEXT NOT NULL DEFAULT 'legacy', input_validity TEXT NOT NULL DEFAULT 'unknown');
CREATE TABLE IF NOT EXISTS answers (id TEXT PRIMARY KEY, selection_id TEXT NOT NULL REFERENCES selections(id), status TEXT NOT NULL, answer TEXT NOT NULL, model_name TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost REAL, elapsed_ms INTEGER, error TEXT, usage TEXT NOT NULL DEFAULT '{}', cost_status TEXT NOT NULL DEFAULT 'not_calculated', prompt_version TEXT);
CREATE TABLE IF NOT EXISTS comparisons (id TEXT PRIMARY KEY, question_id TEXT NOT NULL REFERENCES questions(id), workspace_id TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, reason TEXT, records TEXT NOT NULL, answer_f1 TEXT NOT NULL DEFAULT '{"status":"not_run","value":null}', evidence_f1 TEXT NOT NULL DEFAULT '{"status":"not_run","value":null}');
CREATE INDEX IF NOT EXISTS idx_questions_workspace ON questions(workspace_id);
CREATE INDEX IF NOT EXISTS idx_selections_question ON selections(question_id);
CREATE INDEX IF NOT EXISTS idx_answers_selection ON answers(selection_id);
CREATE INDEX IF NOT EXISTS idx_comparisons_question ON comparisons(question_id);
CREATE TABLE IF NOT EXISTS review_batches (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, contract_type TEXT NOT NULL, status TEXT NOT NULL, token_count INTEGER NOT NULL DEFAULT 0, token_limit INTEGER NOT NULL, reason TEXT, active_chunk_set_id TEXT);
CREATE TABLE IF NOT EXISTS review_files (id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES review_batches(id), filename TEXT NOT NULL, content BLOB NOT NULL, page_count INTEGER NOT NULL, status TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_pages (id TEXT PRIMARY KEY, file_id TEXT NOT NULL REFERENCES review_files(id), batch_id TEXT NOT NULL, page_number INTEGER NOT NULL, status TEXT NOT NULL, content TEXT NOT NULL DEFAULT '[]', error TEXT, attempts INTEGER NOT NULL DEFAULT 0, UNIQUE(file_id, page_number));
CREATE TABLE IF NOT EXISTS review_chunks (id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, chunk_set_id TEXT NOT NULL, file_id TEXT NOT NULL, page_id TEXT NOT NULL, ordinal INTEGER NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, source TEXT NOT NULL, image BLOB, version TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS review_risks (id TEXT PRIMARY KEY, batch_id TEXT NOT NULL REFERENCES review_batches(id), name TEXT NOT NULL, original_query TEXT NOT NULL, retrieval_query TEXT NOT NULL, status TEXT NOT NULL, position INTEGER NOT NULL, custom INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS review_retrievals (id TEXT PRIMARY KEY, risk_id TEXT NOT NULL REFERENCES review_risks(id), round_number INTEGER NOT NULL, query TEXT NOT NULL, method TEXT NOT NULL, status TEXT NOT NULL, candidates TEXT NOT NULL DEFAULT '[]', selected TEXT NOT NULL DEFAULT '[]', error TEXT, calls INTEGER NOT NULL DEFAULT 0, elapsed_ms INTEGER, UNIQUE(risk_id, round_number));
CREATE TABLE IF NOT EXISTS review_decisions (id TEXT PRIMARY KEY, risk_id TEXT NOT NULL REFERENCES review_risks(id), round_number INTEGER NOT NULL, action TEXT NOT NULL, next_query TEXT NOT NULL DEFAULT '', conclusion TEXT, reason TEXT NOT NULL DEFAULT '', pages TEXT NOT NULL DEFAULT '[]', status TEXT NOT NULL, UNIQUE(risk_id, round_number));
CREATE TABLE IF NOT EXISTS review_answers (id TEXT PRIMARY KEY, risk_id TEXT NOT NULL REFERENCES review_risks(id), status TEXT NOT NULL, text TEXT NOT NULL DEFAULT '', model_name TEXT NOT NULL, prompt_version TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0, usage TEXT NOT NULL DEFAULT '{}', cost_status TEXT NOT NULL DEFAULT 'not_calculated', elapsed_ms INTEGER, error TEXT, images_sent INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS review_proofs (id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, risk_id TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, label_conclusion TEXT, label_pages TEXT NOT NULL DEFAULT '[]', verdict INTEGER, evidence_f1 REAL, evidence_status TEXT NOT NULL, rounds INTEGER NOT NULL DEFAULT 0, calls TEXT NOT NULL DEFAULT '{}', tokens TEXT NOT NULL DEFAULT '{}', cost_status TEXT NOT NULL DEFAULT 'not_calculated', review_ms INTEGER, answer_ms INTEGER, valid INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS review_experience (id TEXT PRIMARY KEY, risk_name TEXT NOT NULL, query TEXT NOT NULL, method TEXT NOT NULL, outcome TEXT NOT NULL, proof_id TEXT NOT NULL UNIQUE, batch_id TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS idx_review_pages_batch ON review_pages(batch_id, status);
CREATE INDEX IF NOT EXISTS idx_review_risks_batch ON review_risks(batch_id, position);
"""

class Database:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path or settings.database_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def _session(self):
        conn = self.connect()
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def init(self) -> None:
        with self._session() as conn:
            conn.executescript(SCHEMA)
            migrations = {
                "workspaces": {"chunk_rule_snapshot": "TEXT NOT NULL DEFAULT '{}'", "chunk_rule_fingerprint": "TEXT", "token_count_method": "TEXT NOT NULL DEFAULT 'legacy'"},
                "selections": {"chunk_set_id": "TEXT", "rule_params": "TEXT NOT NULL DEFAULT '{}'", "prompt_version": "TEXT", "input_rule_snapshot": "TEXT NOT NULL DEFAULT '{}'", "input_rule_fingerprint": "TEXT", "input_token_count_method": "TEXT NOT NULL DEFAULT 'legacy'", "input_validity": "TEXT NOT NULL DEFAULT 'unknown'"},
                "answers": {"usage": "TEXT NOT NULL DEFAULT '{}'", "cost_status": "TEXT NOT NULL DEFAULT 'not_calculated'", "prompt_version": "TEXT"},
                "chunks": {"kind": "TEXT NOT NULL DEFAULT 'text'", "source": "TEXT NOT NULL DEFAULT '{}'", "image": "BLOB"},
                "questions": {"rewritten": "TEXT NOT NULL DEFAULT ''", "rewrite_status": "TEXT NOT NULL DEFAULT 'original'", "rewrite_usage": "TEXT NOT NULL DEFAULT '{}'"},
            }
            for table, columns in migrations.items():
                existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                for name, definition in columns.items():
                    if name not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
            # Non-empty old databases are migrated conservatively. Their missing
            # provenance remains explicitly legacy/unknown.
            rows = conn.execute("SELECT id, id AS workspace_id, created_at, chunk_set_id FROM workspaces WHERE chunk_set_id IS NOT NULL").fetchall()
            for row in rows:
                conn.execute("INSERT OR IGNORE INTO chunk_sets (id, workspace_id, created_at, status, rule_snapshot, rule_fingerprint, token_count_method) VALUES (?, ?, ?, 'legacy', '{}', NULL, 'legacy')", (row["chunk_set_id"], row["workspace_id"], row["created_at"]))
            legacy_selections = conn.execute("SELECT id, chunk_set_id, rule_params FROM selections WHERE input_validity = 'unknown' AND (input_rule_snapshot IS NULL OR input_rule_snapshot = '{}')").fetchall()
            for selection in legacy_selections:
                try:
                    params = json.loads(selection["rule_params"] or "{}")
                except (TypeError, ValueError):
                    params = {}
                fingerprint = params.get("chunk_rule_fingerprint")
                raw_snapshot = params.get("chunk_rule_snapshot")
                snapshot = raw_snapshot if isinstance(raw_snapshot, dict) and raw_snapshot else {"status": "unknown"}
                method = params.get("token_count_method", "legacy")
                recorded_chunk_set = selection["chunk_set_id"]
                has_snapshot = raw_snapshot is not None and snapshot.get("status") != "unknown"
                has_chunk_set = bool(recorded_chunk_set) and conn.execute("SELECT 1 FROM chunk_sets WHERE id = ?", (recorded_chunk_set,)).fetchone() is not None
                validity = "valid" if has_snapshot and fingerprint and fingerprint != "legacy" and has_chunk_set else "legacy"
                # Only newly added provenance columns are populated here. A
                # missing historical chunk set must remain missing; the
                # current workspace is not evidence for an old selection.
                conn.execute("UPDATE selections SET input_rule_snapshot = ?, input_rule_fingerprint = ?, input_token_count_method = ?, input_validity = ? WHERE id = ?", (json.dumps(snapshot, ensure_ascii=False), fingerprint, method, validity, selection["id"]))

    def save_workspace(self, workspace: Workspace) -> None:
        with self._session() as conn:
            conn.execute("INSERT INTO workspaces (id, created_at, total_tokens, status, reason, chunk_set_id, chunk_rule_snapshot, chunk_rule_fingerprint, token_count_method) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (workspace.id, workspace.created_at, workspace.total_tokens, workspace.status, workspace.reason, workspace.chunk_set_id, json.dumps(workspace.chunk_rule_snapshot, ensure_ascii=False), workspace.chunk_rule_fingerprint, workspace.token_count_method))
            conn.executemany("INSERT INTO documents (id, workspace_id, filename, content, text, token_count, extraction_ok, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [(d.id, d.workspace_id, d.filename, d.content, d.text, d.token_count, int(d.extraction_ok), d.error) for d in workspace.documents])
            conn.executemany("INSERT INTO chunks (id, chunk_set_id, document_id, ordinal, text, token_count, version, kind, source, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [_chunk_row(c) for c in workspace.chunks])
            if workspace.chunk_set_id:
                conn.execute("INSERT OR IGNORE INTO chunk_sets (id, workspace_id, created_at, status, rule_snapshot, rule_fingerprint, token_count_method) VALUES (?, ?, ?, 'active', ?, ?, ?)", (workspace.chunk_set_id, workspace.id, workspace.created_at, json.dumps(workspace.chunk_rule_snapshot, ensure_ascii=False), workspace.chunk_rule_fingerprint, workspace.token_count_method))

    def save_rechunk(self, old_workspace: Workspace, new_workspace: Workspace) -> Workspace:
        if old_workspace.status != "ready":
            raise ValueError(old_workspace.reason or "Workspace is not ready")
        with self._session() as conn:
            conn.execute("INSERT INTO chunk_sets (id, workspace_id, created_at, status, rule_snapshot, rule_fingerprint, token_count_method) VALUES (?, ?, ?, 'active', ?, ?, ?)", (new_workspace.chunk_set_id, new_workspace.id, new_workspace.created_at, json.dumps(new_workspace.chunk_rule_snapshot, ensure_ascii=False), new_workspace.chunk_rule_fingerprint, new_workspace.token_count_method))
            conn.executemany("INSERT INTO chunks (id, chunk_set_id, document_id, ordinal, text, token_count, version, kind, source, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [_chunk_row(c) for c in new_workspace.chunks])
            conn.execute("UPDATE workspaces SET chunk_set_id = ?, chunk_rule_snapshot = ?, chunk_rule_fingerprint = ?, token_count_method = ? WHERE id = ?", (new_workspace.chunk_set_id, json.dumps(new_workspace.chunk_rule_snapshot, ensure_ascii=False), new_workspace.chunk_rule_fingerprint, new_workspace.token_count_method, new_workspace.id))
            if old_workspace.chunk_set_id:
                conn.execute("UPDATE chunk_sets SET status = 'historical' WHERE id = ?", (old_workspace.chunk_set_id,))
                conn.execute("UPDATE selections SET input_validity = 'obsolete' WHERE workspace_id = ? AND chunk_set_id = ? AND input_validity NOT IN ('obsolete', 'legacy')", (old_workspace.id, old_workspace.chunk_set_id))
        return new_workspace

    def get_workspace(self, workspace_id: str) -> Workspace | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM workspaces WHERE id = ?", (workspace_id,)).fetchone()
            if not row:
                return None
            docs = [Document(r["id"], r["workspace_id"], r["filename"], r["text"], r["token_count"], bool(r["extraction_ok"]), r["error"], bytes(r["content"])) for r in conn.execute("SELECT * FROM documents WHERE workspace_id = ? ORDER BY rowid", (workspace_id,))]
            chunks = [_chunk_from_row(r) for r in conn.execute("SELECT chunks.* FROM chunks JOIN documents ON documents.id = chunks.document_id WHERE chunks.chunk_set_id = ? ORDER BY documents.rowid, chunks.ordinal, chunks.rowid", (row["chunk_set_id"],))]
            try:
                snapshot = json.loads(row["chunk_rule_snapshot"] or "{}")
            except (TypeError, ValueError):
                snapshot = {}
            return Workspace(row["id"], row["created_at"], row["total_tokens"], row["status"], row["reason"], row["chunk_set_id"], docs, chunks, snapshot, row["chunk_rule_fingerprint"], row["token_count_method"])

    def save_question(self, question: Question) -> None:
        with self._session() as conn:
            conn.execute("INSERT INTO questions (id, workspace_id, text, created_at, status, rewritten, rewrite_status, rewrite_usage) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (question.id, question.workspace_id, question.text, question.created_at, question.status, question.rewritten, question.rewrite_status, json.dumps(question.rewrite_usage, ensure_ascii=False)))

    def get_question(self, question_id: str) -> Question | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM questions WHERE id = ?", (question_id,)).fetchone()
            return _question_from_row(row) if row else None

    def list_questions(self, workspace_id: str) -> list[Question]:
        with self._session() as conn:
            return [_question_from_row(r) for r in conn.execute("SELECT * FROM questions WHERE workspace_id = ? ORDER BY created_at, rowid", (workspace_id,))]

    def save_selection(self, selection: Selection) -> None:
        params = selection.rule_params or {}
        snapshot = params.get("chunk_rule_snapshot", "unknown")
        fingerprint = params.get("chunk_rule_fingerprint")
        token_method = params.get("token_count_method", "legacy")
        validity = selection.input_validity
        with self._session() as conn:
            conn.execute("INSERT INTO selections (id, question_id, workspace_id, strategy, strategy_version, status, selected_chunks, total_tokens, usage, reason, threshold, budget, model_name, elapsed_ms, chunk_set_id, rule_params, prompt_version, input_rule_snapshot, input_rule_fingerprint, input_token_count_method, input_validity) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (selection.id, selection.question_id, selection.workspace_id, selection.strategy, selection.strategy_version, selection.status, json.dumps(selection.selected_chunks, ensure_ascii=False), selection.total_tokens, json.dumps(selection.usage, ensure_ascii=False), selection.reason, selection.threshold, selection.budget, selection.model_name, selection.elapsed_ms, selection.chunk_set_id or selection.usage.get("input_chunk_set_id"), json.dumps(selection.rule_params, ensure_ascii=False), selection.prompt_version, json.dumps(snapshot, ensure_ascii=False), fingerprint, token_method, validity))

    def get_selection(self, selection_id: str) -> Selection | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM selections WHERE id = ?", (selection_id,)).fetchone()
            if not row:
                return None
            params = json.loads(row["rule_params"] or "{}")
            return Selection(row["id"], row["question_id"], row["workspace_id"], row["strategy"], row["strategy_version"], row["status"], json.loads(row["selected_chunks"]), row["total_tokens"], json.loads(row["usage"]), row["reason"], row["threshold"], row["budget"], row["model_name"], row["elapsed_ms"], row["chunk_set_id"], params, row["prompt_version"], row["input_validity"])

    def list_selections(self, question_id: str) -> list[Selection]:
        with self._session() as conn:
            ids = [r["id"] for r in conn.execute("SELECT id FROM selections WHERE question_id = ? ORDER BY rowid", (question_id,))]
        return [self.get_selection(i) for i in ids]

    def mark_selection_obsolete(self, selection_id: str, reason: str = "Marked obsolete by caller") -> bool:
        with self._session() as conn:
            result = conn.execute("UPDATE selections SET status = 'obsolete', input_validity = 'obsolete', reason = ? WHERE id = ?", (reason, selection_id))
            return result.rowcount == 1

    def save_documents_changed(self, workspace: Workspace, added: list[Chunk] | None = None,
                               removed_document_id: str | None = None) -> None:
        with self._session() as conn:
            conn.execute("UPDATE workspaces SET total_tokens = ?, status = ?, reason = ? WHERE id = ?",
                         (workspace.total_tokens, workspace.status, workspace.reason, workspace.id))
            if removed_document_id:
                conn.execute("DELETE FROM chunks WHERE document_id = ? AND chunk_set_id = ?",
                             (removed_document_id, workspace.chunk_set_id))
                conn.execute("DELETE FROM documents WHERE id = ?", (removed_document_id,))
            if added:
                document = workspace.documents[-1]
                conn.execute("INSERT INTO documents (id, workspace_id, filename, content, text, token_count, extraction_ok, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (document.id, document.workspace_id, document.filename, document.content, document.text,
                              document.token_count, int(document.extraction_ok), document.error))
                conn.executemany("INSERT INTO chunks (id, chunk_set_id, document_id, ordinal, text, token_count, version, kind, source, image) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                 [_chunk_row(chunk) for chunk in added])

    def save_answer(self, answer: AnswerRun) -> None:
        with self._session() as conn:
            conn.execute("INSERT INTO answers (id, selection_id, status, answer, model_name, input_tokens, output_tokens, cost, elapsed_ms, error, usage, cost_status, prompt_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (answer.id, answer.selection_id, answer.status, answer.answer, answer.model_name, answer.input_tokens, answer.output_tokens, answer.cost, answer.elapsed_ms, answer.error, json.dumps(answer.usage, ensure_ascii=False), answer.cost_status, answer.prompt_version))

    def list_answers(self, selection_ids: Iterable[str]) -> list[AnswerRun]:
        ids = list(selection_ids)
        if not ids:
            return []
        marks = ",".join("?" for _ in ids)
        with self._session() as conn:
            rows = conn.execute(f"SELECT * FROM answers WHERE selection_id IN ({marks}) ORDER BY rowid", ids).fetchall()
            return [AnswerRun(r["id"], r["selection_id"], r["status"], r["answer"], r["model_name"], r["input_tokens"], r["output_tokens"], r["cost"], r["elapsed_ms"], r["error"], json.loads(r["usage"] or "{}"), r["cost_status"], r["prompt_version"]) for r in rows]

    def save_comparison(self, comparison: Comparison) -> None:
        with self._session() as conn:
            conn.execute("INSERT INTO comparisons (id, question_id, workspace_id, created_at, status, reason, records, answer_f1, evidence_f1) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (comparison.id, comparison.question_id, comparison.workspace_id, comparison.created_at, comparison.status, comparison.reason, json.dumps(comparison.records, ensure_ascii=False), json.dumps(comparison.answer_f1, ensure_ascii=False), json.dumps(comparison.evidence_f1, ensure_ascii=False)))

    def get_comparison(self, comparison_id: str) -> Comparison | None:
        with self._session() as conn:
            row = conn.execute("SELECT * FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
            if not row:
                return None
            return Comparison(row["id"], row["question_id"], row["workspace_id"], row["created_at"], row["status"], row["reason"], json.loads(row["records"]), json.loads(row["answer_f1"]), json.loads(row["evidence_f1"]))

    def list_comparisons(self, question_id: str) -> list[Comparison]:
        with self._session() as conn:
            ids = [r["id"] for r in conn.execute("SELECT id FROM comparisons WHERE question_id = ? ORDER BY rowid", (question_id,))]
        return [self.get_comparison(i) for i in ids]

    def create_review_batch(self, batch_id: str, created_at: str, contract_type: str, token_limit: int,
                            files: list[tuple[str, str, bytes, int, list[tuple[str, int]]]]) -> None:
        with self._session() as conn:
            conn.execute("INSERT INTO review_batches (id, created_at, contract_type, status, token_count, token_limit) VALUES (?, ?, ?, 'waiting', 0, ?)",
                         (batch_id, created_at, contract_type, token_limit))
            for file_id, filename, content, page_count, pages in files:
                conn.execute("INSERT INTO review_files (id, batch_id, filename, content, page_count, status) VALUES (?, ?, ?, ?, ?, 'waiting')",
                             (file_id, batch_id, filename, content, page_count))
                conn.executemany("INSERT INTO review_pages (id, file_id, batch_id, page_number, status) VALUES (?, ?, ?, ?, 'waiting')",
                                 [(page_id, file_id, batch_id, number) for page_id, number in pages])

    def review_batch(self, batch_id: str) -> sqlite3.Row | None:
        with self._session() as conn:
            return conn.execute("SELECT * FROM review_batches WHERE id = ?", (batch_id,)).fetchone()

    def review_progress(self, batch_id: str) -> dict[str, Any]:
        with self._session() as conn:
            batch = conn.execute("SELECT * FROM review_batches WHERE id = ?", (batch_id,)).fetchone()
            if not batch:
                return {}
            pages = conn.execute("SELECT status, COUNT(*) AS n FROM review_pages WHERE batch_id = ? GROUP BY status", (batch_id,)).fetchall()
            counts = {row["status"]: row["n"] for row in pages}
            risks = [dict(row) for row in conn.execute("SELECT id, name, original_query, status, position FROM review_risks WHERE batch_id = ? ORDER BY position", (batch_id,))]
            failed = [dict(row) for row in conn.execute("SELECT id, file_id, page_number, error FROM review_pages WHERE batch_id = ? AND status = 'failed' ORDER BY page_number", (batch_id,))]
            return {"id": batch["id"], "status": batch["status"], "contract_type": batch["contract_type"],
                    "created_at": batch["created_at"], "token_count": batch["token_count"], "token_limit": batch["token_limit"],
                    "reason": batch["reason"], "pages": counts, "failed_pages": failed, "risks": risks}

    def claim_review_page(self) -> sqlite3.Row | None:
        with self._session() as conn:
            conn.execute("UPDATE review_pages SET status = 'waiting' WHERE status = 'processing'")
            row = conn.execute("SELECT review_pages.id, review_pages.file_id, review_pages.batch_id, review_pages.page_number, review_files.content, review_files.filename FROM review_pages JOIN review_files ON review_files.id = review_pages.file_id JOIN review_batches ON review_batches.id = review_pages.batch_id WHERE review_pages.status = 'waiting' AND review_batches.status != 'rejected' ORDER BY review_pages.rowid LIMIT 1").fetchone()
            if not row:
                return None
            conn.execute("UPDATE review_pages SET status = 'processing', attempts = attempts + 1 WHERE id = ?", (row["id"],))
            conn.execute("UPDATE review_batches SET status = 'processing' WHERE id = ? AND status = 'waiting'", (row["batch_id"],))
            return row

    def finish_review_page(self, page_id: str, blocks: list[dict[str, Any]]) -> None:
        stored = []
        for block in blocks:
            item = {key: value for key, value in block.items() if key != "image"}
            item["has_image"] = bool(block.get("image"))
            stored.append(item)
        with self._session() as conn:
            conn.execute("UPDATE review_pages SET status = 'complete', content = ?, error = NULL WHERE id = ?",
                         (json.dumps(stored, ensure_ascii=False), page_id))

    def fail_review_page(self, page_id: str, error: str) -> None:
        with self._session() as conn:
            conn.execute("UPDATE review_pages SET status = 'failed', error = ? WHERE id = ?", (error, page_id))

    def review_page_content(self, page_id: str) -> list[dict[str, Any]]:
        with self._session() as conn:
            row = conn.execute("SELECT content FROM review_pages WHERE id = ?", (page_id,)).fetchone()
        return json.loads(row["content"]) if row else []

    def claim_chunk_page(self, chunk_set_id: str) -> sqlite3.Row | None:
        with self._session() as conn:
            row = conn.execute("""SELECT review_pages.* FROM review_pages
                JOIN review_batches ON review_batches.id = review_pages.batch_id
                WHERE review_pages.status = 'complete' AND review_batches.status != 'rejected'
                AND NOT EXISTS (SELECT 1 FROM review_pages failed WHERE failed.batch_id = review_pages.batch_id AND failed.status = 'failed')
                AND NOT EXISTS (SELECT 1 FROM review_pages waiting WHERE waiting.batch_id = review_pages.batch_id AND waiting.status IN ('waiting', 'processing'))
                AND NOT EXISTS (SELECT 1 FROM review_chunks WHERE review_chunks.page_id = review_pages.id AND review_chunks.chunk_set_id = ?)
                ORDER BY review_pages.rowid LIMIT 1""", (chunk_set_id,)).fetchone()
            return row

    def replace_page_chunks(self, page_id: str, chunk_set_id: str, rows: list[tuple]) -> None:
        with self._session() as conn:
            existing = conn.execute("SELECT COUNT(*) AS n FROM review_chunks WHERE page_id = ? AND chunk_set_id = ?", (page_id, chunk_set_id)).fetchone()["n"]
            if existing:
                return
            conn.executemany("INSERT INTO review_chunks (id, batch_id, chunk_set_id, file_id, page_id, ordinal, kind, text, token_count, source, image, version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def refresh_batch_tokens(self, batch_id: str, chunk_set_id: str, token_limit: int) -> str:
        with self._session() as conn:
            total = conn.execute("SELECT COALESCE(SUM(token_count), 0) AS n FROM review_chunks WHERE batch_id = ? AND chunk_set_id = ?", (batch_id, chunk_set_id)).fetchone()["n"]
            pending = conn.execute("SELECT COUNT(*) AS n FROM review_pages WHERE batch_id = ? AND status != 'complete'", (batch_id,)).fetchone()["n"]
            status = "rejected" if total > token_limit else ("ready" if pending == 0 else "processing")
            reason = "正文超过 80 万 token" if total > token_limit else None
            conn.execute("UPDATE review_batches SET token_count = ?, status = ?, reason = ?, active_chunk_set_id = ? WHERE id = ?",
                         (total, status, reason, chunk_set_id, batch_id))
            return status

    def batch_needs_risks(self, batch_id: str) -> bool:
        with self._session() as conn:
            batch = conn.execute("SELECT status FROM review_batches WHERE id = ?", (batch_id,)).fetchone()
            count = conn.execute("SELECT COUNT(*) AS n FROM review_risks WHERE batch_id = ?", (batch_id,)).fetchone()["n"]
            return bool(batch and batch["status"] == "ready" and count == 0)

    def save_risks(self, rows: list[tuple]) -> None:
        with self._session() as conn:
            conn.executemany("INSERT OR IGNORE INTO review_risks (id, batch_id, name, original_query, retrieval_query, status, position, custom) VALUES (?, ?, ?, ?, ?, 'waiting', ?, ?)", rows)

    def claim_risk(self) -> sqlite3.Row | None:
        with self._session() as conn:
            row = conn.execute("SELECT review_risks.*, review_batches.active_chunk_set_id, review_batches.contract_type FROM review_risks JOIN review_batches ON review_batches.id = review_risks.batch_id WHERE review_risks.status = 'waiting' AND review_batches.status = 'ready' ORDER BY review_risks.position LIMIT 1").fetchone()
            if row:
                conn.execute("UPDATE review_risks SET status = 'processing' WHERE id = ?", (row["id"],))
            return row

    def batch_chunks(self, batch_id: str, chunk_set_id: str) -> list[sqlite3.Row]:
        with self._session() as conn:
            return list(conn.execute("SELECT * FROM review_chunks WHERE batch_id = ? AND chunk_set_id = ? ORDER BY ordinal, rowid", (batch_id, chunk_set_id)))

    def save_retrieval(self, row: tuple) -> None:
        with self._session() as conn:
            conn.execute("INSERT OR REPLACE INTO review_retrievals (id, risk_id, round_number, query, method, status, candidates, selected, error, calls, elapsed_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)

    def latest_retrieval(self, risk_id: str) -> sqlite3.Row | None:
        with self._session() as conn:
            return conn.execute("SELECT * FROM review_retrievals WHERE risk_id = ? ORDER BY round_number DESC LIMIT 1", (risk_id,)).fetchone()

    def prior_queries(self, risk_id: str) -> list[str]:
        with self._session() as conn:
            return [row["query"] for row in conn.execute("SELECT query FROM review_retrievals WHERE risk_id = ? ORDER BY round_number", (risk_id,))]

    def save_decision(self, row: tuple) -> None:
        with self._session() as conn:
            conn.execute("INSERT OR REPLACE INTO review_decisions (id, risk_id, round_number, action, next_query, conclusion, reason, pages, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", row)

    def latest_decision(self, risk_id: str) -> sqlite3.Row | None:
        with self._session() as conn:
            return conn.execute("SELECT * FROM review_decisions WHERE risk_id = ? ORDER BY round_number DESC LIMIT 1", (risk_id,)).fetchone()

    def finish_risk(self, risk_id: str, status: str, next_query: str | None = None) -> None:
        with self._session() as conn:
            if next_query is None:
                conn.execute("UPDATE review_risks SET status = ? WHERE id = ?", (status, risk_id))
            else:
                conn.execute("UPDATE review_risks SET status = ?, retrieval_query = ? WHERE id = ?", (status, next_query, risk_id))

    def save_review_answer(self, row: tuple) -> None:
        with self._session() as conn:
            conn.execute("INSERT INTO review_answers (id, risk_id, status, text, model_name, prompt_version, input_tokens, output_tokens, usage, cost_status, elapsed_ms, error, images_sent) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", row)

    def review_results(self, batch_id: str) -> list[dict[str, Any]]:
        with self._session() as conn:
            risks = conn.execute("SELECT * FROM review_risks WHERE batch_id = ? ORDER BY position", (batch_id,)).fetchall()
            results = []
            for risk in risks:
                decision = conn.execute("SELECT * FROM review_decisions WHERE risk_id = ? AND action = 'stop' ORDER BY round_number DESC LIMIT 1", (risk["id"],)).fetchone()
                answer = conn.execute("SELECT * FROM review_answers WHERE risk_id = ? ORDER BY rowid DESC LIMIT 1", (risk["id"],)).fetchone()
                results.append({"id": risk["id"], "name": risk["name"], "status": risk["status"],
                                "conclusion": decision["conclusion"] if decision else None,
                                "pages": json.loads(decision["pages"]) if decision else [],
                                "reason": decision["reason"] if decision else "",
                                "answer": answer["text"] if answer else "",
                                "answer_status": answer["status"] if answer else None,
                                "answer_error": answer["error"] if answer else None})
            return results

def _chunk_row(chunk: Chunk) -> tuple:
    return (chunk.id, chunk.chunk_set_id, chunk.document_id, chunk.ordinal, chunk.text, chunk.token_count,
            chunk.version, chunk.kind, json.dumps(chunk.source, ensure_ascii=False), chunk.image or None)


def _chunk_from_row(row: sqlite3.Row) -> Chunk:
    keys = set(row.keys())
    source = {}
    if "source" in keys and row["source"]:
        try:
            source = json.loads(row["source"])
        except (TypeError, ValueError):
            source = {}
    return Chunk(row["id"], row["chunk_set_id"], row["document_id"], row["ordinal"], row["text"],
                 row["token_count"], row["version"], row["kind"] if "kind" in keys and row["kind"] else "text",
                 source, bytes(row["image"]) if "image" in keys and row["image"] else b"")


def _question_from_row(row: sqlite3.Row) -> Question:
    keys = set(row.keys())
    usage = {}
    if "rewrite_usage" in keys and row["rewrite_usage"]:
        try:
            usage = json.loads(row["rewrite_usage"])
        except (TypeError, ValueError):
            usage = {}
    return Question(row["id"], row["workspace_id"], row["text"], row["created_at"], row["status"],
                    row["rewritten"] if "rewritten" in keys else "",
                    row["rewrite_status"] if "rewrite_status" in keys else "original", usage)


db = Database()
