"""Independent end-to-end lifecycle acceptance, offline and isolated.

Run explicitly: python -X utf8 tests/audit_lifecycle.py
"""
from dataclasses import asdict, replace
from html import escape
import json
from pathlib import Path
import sqlite3
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from audit_acceptance import h  # noqa: E402,F401


def uploaded_run(h):
    response = h.client.post("/workspaces", files={
        "files": ("audit.pdf", h.pdf_bytes("alpha beta gamma delta"), "application/pdf")
    })
    assert response.status_code == 200
    ws = response.json()
    response = h.client.post(f"/workspaces/{ws['id']}/questions",
                             json={"question": "What?", "mode": "full"})
    assert response.status_code == 200
    return ws, response.json()


def test_answer_only_retry_preserves_selection_and_exposes_latest(h):
    ws, result = uploaded_run(h)
    previous = result["runs"][0]
    selection_before = asdict(h.db.get_selection(previous["id"]))
    call_count = len(h.calls)
    response = h.client.post(f"/selections/{previous['id']}/answers/retry")
    assert response.status_code in (200, 201), response.text
    answers = h.db.list_answers([previous["id"]])
    assert len(answers) == 2
    assert len(h.calls) == call_count + 1 and "messages" in h.calls[-1]
    assert asdict(h.db.get_selection(previous["id"])) == selection_before
    history = h.client.get(f"/questions/{result['question']['id']}/runs").json()
    assert len(history["runs"]) == 1
    assert history["runs"][0]["answer"]["id"] == answers[-1].id
    assert answers[0].id == previous["answer"]["id"]


def test_rechunk_preserves_old_data_and_prevents_stale_answer_retry(h, monkeypatch):
    ws, result = uploaded_run(h)
    previous = result["runs"][0]
    old_set = ws["chunk_set_id"]
    monkeypatch.setattr(h.main, "settings", replace(h.config, chunk_size=3,
                                                   chunk_overlap=1, chunk_version="audit-words-v2"))
    response = h.client.post(f"/workspaces/{ws['id']}/rechunk")
    assert response.status_code in (200, 201), response.text
    loaded = h.client.get(f"/workspaces/{ws['id']}").json()
    assert loaded["chunk_set_id"] != old_set
    connection = h.db.connect()
    try:
        assert connection.execute("SELECT count(*) FROM chunks WHERE chunk_set_id=?", (old_set,)).fetchone()[0] > 0
    finally:
        connection.close()
    old_answers = h.db.list_answers([previous["id"]])
    assert old_answers[0].id == previous["answer"]["id"]
    calls_before = len(h.calls)
    denied = h.client.post(f"/selections/{previous['id']}/answers/retry")
    assert denied.status_code in (409, 422)
    assert len(h.calls) == calls_before


def test_saved_chunk_configuration_is_not_replaced_by_current_settings(h, monkeypatch):
    ws, result = uploaded_run(h)
    reopened = h.Database(h.db.path)
    original = reopened.get_workspace(ws["id"])
    assert original.chunk_rule_snapshot["chunk_size"] == 2
    assert original.chunk_rule_snapshot["chunk_overlap"] == 0
    assert original.chunk_rule_fingerprint
    monkeypatch.setattr(h.main, "db", reopened)
    monkeypatch.setattr(h.main, "settings", replace(h.config, chunk_size=3,
                                                   chunk_overlap=1, chunk_version="future-v2"))
    second = h.client.post(f"/workspaces/{ws['id']}/questions",
                           json={"question": "Again?", "mode": "jev"})
    assert second.status_code == 200
    selection = reopened.get_selection(second.json()["runs"][0]["id"])
    assert "audit-words-v1" in json.dumps(asdict(selection))
    still_original = reopened.get_workspace(ws["id"])
    assert still_original.chunk_rule_snapshot == original.chunk_rule_snapshot
    assert still_original.chunk_rule_fingerprint == original.chunk_rule_fingerprint


def test_comparison_is_persisted_without_model_calls_and_reports_incomplete(h, monkeypatch):
    ws, result = uploaded_run(h)
    count = len(h.calls)
    response = h.client.post(f"/questions/{result['question']['id']}/comparisons")
    assert response.status_code in (200, 201), response.text
    comparison = response.json()
    comparison = comparison.get("comparison", comparison)
    assert comparison["status"] in ("incomplete", "invalid")
    serialized = json.dumps(comparison)
    assert result["runs"][0]["id"] in serialized
    assert result["runs"][0]["answer"]["id"] in serialized
    assert "not_run" in serialized
    assert "not_calculated" in serialized
    assert len(h.calls) == count
    monkeypatch.setattr(h.main, "db", h.Database(h.db.path))
    fetched = h.client.get(f"/comparisons/{comparison['id']}")
    assert fetched.status_code == 200
    persisted = fetched.json()
    persisted = persisted.get("comparison", persisted)
    assert persisted == comparison
    listing = h.client.get(f"/questions/{result['question']['id']}/comparisons")
    assert listing.status_code == 200 and comparison["id"] in listing.text
    assert len(h.calls) == count


LEGACY_SCHEMA = """
CREATE TABLE workspaces (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, total_tokens INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT, chunk_set_id TEXT);
CREATE TABLE documents (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, filename TEXT NOT NULL, content BLOB NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, extraction_ok INTEGER NOT NULL, error TEXT);
CREATE TABLE chunks (id TEXT PRIMARY KEY, chunk_set_id TEXT NOT NULL, document_id TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, version TEXT NOT NULL);
CREATE TABLE questions (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL);
CREATE TABLE selections (id TEXT PRIMARY KEY, question_id TEXT NOT NULL, workspace_id TEXT NOT NULL, strategy TEXT NOT NULL, strategy_version TEXT NOT NULL, status TEXT NOT NULL, selected_chunks TEXT NOT NULL, total_tokens INTEGER NOT NULL, usage TEXT NOT NULL, reason TEXT, threshold REAL, budget INTEGER, model_name TEXT, elapsed_ms INTEGER);
CREATE TABLE answers (id TEXT PRIMARY KEY, selection_id TEXT NOT NULL, status TEXT NOT NULL, answer TEXT NOT NULL, model_name TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost REAL, elapsed_ms INTEGER, error TEXT);
"""


def test_nonempty_old_schema_preserves_original_values_and_unknown_provenance(h, tmp_path, monkeypatch):
    path = tmp_path / "nonempty-legacy.sqlite3"
    original = {
        "workspaces": ("legacy-ws", "old-time", 2, "ready", None, "legacy-set"),
        "documents": ("legacy-doc", "legacy-ws", "old.pdf", b"legacy-pdf", "old text", 2, 1, None),
        "chunks": ("legacy-chunk", "legacy-set", "legacy-doc", 0, "old text", 2, "old-rule"),
        "questions": ("legacy-question", "legacy-ws", "What?", "old-time", "created"),
        "selections": ("legacy-selection", "legacy-question", "legacy-ws", "full", "full-v1",
                       "complete", json.dumps([{"chunk_id": "legacy-chunk", "document_id": "legacy-doc",
                                                "text": "old text", "token_count": 2, "score": None}]),
                       2, json.dumps({"source_tokens": 2}), None, None, None, "none", 0),
        "answers": ("legacy-answer", "legacy-selection", "complete", "old answer", "old-model", 5, 2, None, 5, None),
    }
    connection = sqlite3.connect(path)
    try:
        connection.executescript(LEGACY_SCHEMA)
        for table, row in original.items():
            connection.execute("INSERT INTO " + table + " VALUES (" + ",".join("?" for _ in row) + ")", row)
        connection.commit()
    finally:
        connection.close()
    migrated = h.Database(path)
    connection = migrated.connect()
    try:
        for table, row in original.items():
            actual = tuple(connection.execute("SELECT * FROM " + table).fetchone())
            assert actual[:len(row)] == row, table
    finally:
        connection.close()
    legacy = migrated.get_workspace("legacy-ws")
    assert legacy.chunk_rule_snapshot.get("chunk_size") is None
    assert legacy.chunk_rule_snapshot.get("chunk_overlap") is None
    assert legacy.chunk_rule_fingerprint is None
    assert legacy.token_count_method in ("legacy", "unknown")
    monkeypatch.setattr(h.main, "db", migrated)
    count = len(h.calls)
    denied = h.client.post("/selections/legacy-selection/answers/retry")
    assert denied.status_code in (409, 422)
    assert len(h.calls) == count


def test_history_page_preserves_run_details_and_failure_reasons(h, monkeypatch):
    ws, result = uploaded_run(h)
    qid = result["question"]["id"]
    run = result["runs"][0]
    page = h.client.get(f"/questions/{qid}/history")
    assert page.status_code == 200
    assert "alpha" in page.text and "complete" in page.text
    assert escape(ws["chunk_set_id"]) not in page.text
    rag = h.client.post(f"/workspaces/{ws['id']}/questions",
                        json={"question": "RAG?", "mode": "rag"}).json()
    rag_page = h.client.get(f"/questions/{rag['question']['id']}/history")
    assert escape(rag["runs"][0]["reason"]) in rag_page.text
    def fail_generation(*args):
        raise h.services.requests.RequestException("simulated generation failure")
    monkeypatch.setattr(h.services, "post_json", fail_generation)
    failed = h.client.post(f"/workspaces/{ws['id']}/questions",
                           json={"question": "Failed?", "mode": "full"}).json()
    failed_run = failed["runs"][0]
    failed_page = h.client.get(f"/questions/{failed['question']['id']}/history")
    assert failed_run["status"] == "complete" and failed_run["answer"]["status"] == "failed"
    assert escape(failed_run["answer"]["error"]) in failed_page.text


if __name__ == "__main__":
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="lifecycle-review-", dir=runtime) as directory:
        raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "--basetemp", str(Path(directory) / "cases")]))
