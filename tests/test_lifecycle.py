from __future__ import annotations

import sqlite3
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from test_local import pdf_bytes  # noqa: E402

from app.config import settings  # noqa: E402
from app.database import Database  # noqa: E402
from app import services  # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    import app.main as main
    database = Database(tmp_path / "lifecycle.sqlite3")
    config = replace(settings, jev_api_key="jev", deepseek_api_key="deep", chunk_size=2, chunk_overlap=0,
                     standard_rag_configured=False)

    def fake_call(url, key, payload, timeout):
        if "typesafe" in url:
            return {"answers": {"relevance": {"noul": 0.9}}}
        return {"choices": [{"message": {"content": "answer"}}], "usage": {"prompt_tokens": 11, "completion_tokens": 2}}

    monkeypatch.setattr(main, "db", database)
    monkeypatch.setattr(main, "settings", config)
    monkeypatch.setattr(services, "post_json", fake_call)
    return main, database, config


def test_nonempty_legacy_schema_migration_preserves_records(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE workspaces (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, total_tokens INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT, chunk_set_id TEXT);
        CREATE TABLE documents (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, filename TEXT NOT NULL, content BLOB NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, extraction_ok INTEGER NOT NULL, error TEXT);
        CREATE TABLE chunks (id TEXT PRIMARY KEY, chunk_set_id TEXT NOT NULL, document_id TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, version TEXT NOT NULL);
        CREATE TABLE questions (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL);
        CREATE TABLE selections (id TEXT PRIMARY KEY, question_id TEXT NOT NULL, workspace_id TEXT NOT NULL, strategy TEXT NOT NULL, strategy_version TEXT NOT NULL, status TEXT NOT NULL, selected_chunks TEXT NOT NULL, total_tokens INTEGER NOT NULL, usage TEXT NOT NULL, reason TEXT, threshold REAL, budget INTEGER, model_name TEXT, elapsed_ms INTEGER);
        CREATE TABLE answers (id TEXT PRIMARY KEY, selection_id TEXT NOT NULL, status TEXT NOT NULL, answer TEXT NOT NULL, model_name TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost REAL, elapsed_ms INTEGER, error TEXT);
        INSERT INTO workspaces VALUES ('ws_old','2026-01-01',2,'ready',NULL,'chunks_old');
        INSERT INTO documents VALUES ('doc_old','ws_old','old.pdf',X'25504446','old text',2,1,NULL);
        INSERT INTO chunks VALUES ('chunk_old','chunks_old','doc_old',0,'old text',2,'old-v1');
        INSERT INTO questions VALUES ('q_old','ws_old','old question','2026-01-01','created');
        INSERT INTO selections VALUES ('sel_old','q_old','ws_old','full','full-v1','complete','[]',2,'{}',NULL,NULL,NULL,'none',1);
        INSERT INTO answers VALUES ('ans_old','sel_old','complete','old answer','old-model',2,1,NULL,1,NULL);
        """)
    database = Database(path)
    workspace = database.get_workspace("ws_old")
    selection = database.get_selection("sel_old")
    answers = database.list_answers(["sel_old"])
    assert workspace and workspace.chunks[0].text == "old text"
    assert selection and selection.input_validity == "legacy" and selection.rule_params == {}
    assert answers[0].answer == "old answer" and answers[0].usage == {}


def test_partial_old_schema_fingerprint_does_not_infer_current_provenance(tmp_path):
    path = tmp_path / "partial-legacy.sqlite3"
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE workspaces (id TEXT PRIMARY KEY, created_at TEXT NOT NULL, total_tokens INTEGER NOT NULL, status TEXT NOT NULL, reason TEXT, chunk_set_id TEXT);
        CREATE TABLE documents (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, filename TEXT NOT NULL, content BLOB NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, extraction_ok INTEGER NOT NULL, error TEXT);
        CREATE TABLE chunks (id TEXT PRIMARY KEY, chunk_set_id TEXT NOT NULL, document_id TEXT NOT NULL, ordinal INTEGER NOT NULL, text TEXT NOT NULL, token_count INTEGER NOT NULL, version TEXT NOT NULL);
        CREATE TABLE questions (id TEXT PRIMARY KEY, workspace_id TEXT NOT NULL, text TEXT NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL);
        CREATE TABLE selections (id TEXT PRIMARY KEY, question_id TEXT NOT NULL, workspace_id TEXT NOT NULL, strategy TEXT NOT NULL, strategy_version TEXT NOT NULL, status TEXT NOT NULL, selected_chunks TEXT NOT NULL, total_tokens INTEGER NOT NULL, usage TEXT NOT NULL, reason TEXT, threshold REAL, budget INTEGER, model_name TEXT, elapsed_ms INTEGER, rule_params TEXT NOT NULL DEFAULT '{}');
        CREATE TABLE answers (id TEXT PRIMARY KEY, selection_id TEXT NOT NULL, status TEXT NOT NULL, answer TEXT NOT NULL, model_name TEXT NOT NULL, input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL, cost REAL, elapsed_ms INTEGER, error TEXT);
        INSERT INTO workspaces VALUES ('ws_partial','2026-01-02',2,'ready',NULL,'current-set');
        INSERT INTO documents VALUES ('doc_partial','ws_partial','old.pdf',X'25504446','old text',2,1,NULL);
        INSERT INTO chunks VALUES ('chunk_partial','current-set','doc_partial',0,'old text',2,'old-rule');
        INSERT INTO questions VALUES ('q_partial','ws_partial','old question','2026-01-02','created');
        INSERT INTO selections VALUES ('sel_partial','q_partial','ws_partial','full','full-v1','complete','[]',2,'{}',NULL,NULL,NULL,'none',1,'{"chunk_rule_fingerprint":"fingerprint-only"}');
        """)

    database = Database(path)
    selection = database.get_selection("sel_partial")
    assert selection and selection.input_validity != "valid"
    assert selection.chunk_set_id is None
    assert selection.rule_params == {"chunk_rule_fingerprint": "fingerprint-only"}
    workspace = database.get_workspace("ws_partial")
    assert workspace and workspace.chunk_set_id == "current-set"
    with database.connect() as conn:
        row = conn.execute("SELECT strategy, selected_chunks, rule_params FROM selections WHERE id = ?", ("sel_partial",)).fetchone()
    assert tuple(row) == ("full", "[]", '{"chunk_rule_fingerprint":"fingerprint-only"}')


def test_rechunk_snapshots_new_rules_and_invalidates_old_selection(isolated):
    main, database, config = isolated
    client = TestClient(main.app)
    workspace = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("one two three four"), "application/pdf")}).json()
    run = client.post(f"/workspaces/{workspace['id']}/questions", json={"question": "q", "mode": "full"}).json()
    selection_id = run["runs"][0]["id"]
    main.settings = replace(config, chunk_size=3, chunk_version="new-rules")
    changed = client.post(f"/workspaces/{workspace['id']}/rechunk")
    assert changed.status_code == 200
    assert changed.json()["chunk_set_id"] != workspace["chunk_set_id"]
    assert changed.json()["chunk_rule_fingerprint"] != workspace["chunk_rule_fingerprint"]
    assert database.get_selection(selection_id).input_validity == "obsolete"


def test_answer_only_retry_keeps_all_answers_and_latest(isolated):
    main, database, config = isolated
    client = TestClient(main.app)
    workspace = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("one two"), "application/pdf")}).json()
    run = client.post(f"/workspaces/{workspace['id']}/questions", json={"question": "q", "mode": "full"}).json()
    selection_id = run["runs"][0]["id"]
    retried = client.post(f"/selections/{selection_id}/answers/retry")
    assert retried.status_code == 200
    assert len(retried.json()["answers"]) == 2
    history = client.get(f"/questions/{run['question']['id']}/runs").json()["runs"][0]
    assert history["answer"]["id"] == history["answers"][-1]["id"]
    assert database.get_selection(selection_id).status == "complete"


def test_comparison_is_independent_persistent_and_marks_incomplete(isolated):
    main, database, config = isolated
    client = TestClient(main.app)
    workspace = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("one two"), "application/pdf")}).json()
    run = client.post(f"/workspaces/{workspace['id']}/questions", json={"question": "q", "mode": "full"}).json()
    comparison = client.post(f"/questions/{run['question']['id']}/comparisons")
    assert comparison.status_code == 200
    body = comparison.json()
    assert body["status"] == "incomplete"
    assert body["answer_f1"]["status"] == "not_run"
    record = body["records"][0]
    assert record["cost_status"] == "not_calculated" or record["cost_status"] == "calculated"
    loaded = client.get(f"/comparisons/{body['id']}").json()
    assert loaded["records"][0]["selection_id"] == record["selection_id"]
    assert database.get_selection(record["selection_id"]).status == "complete"


def test_history_page_displays_answer_details_and_provenance(isolated):
    main, database, config = isolated
    client = TestClient(main.app)
    workspace = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("one two"), "application/pdf")}).json()
    run = client.post(f"/workspaces/{workspace['id']}/questions", json={"question": "q", "mode": "full"}).json()
    page = client.get(f"/questions/{run['question']['id']}/history")
    assert page.status_code == 200
    assert "answer" in page.text and "complete" in page.text
    assert "full-v1" not in page.text and workspace["chunk_set_id"] not in page.text
    assert "one two" in page.text


def test_selection_retry_preserves_successful_original_and_input_version(isolated):
    main, database, config = isolated
    client = TestClient(main.app)
    workspace = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("one two"), "application/pdf")}).json()
    first = client.post(f"/workspaces/{workspace['id']}/questions", json={"question": "q", "mode": "jev"}).json()
    qid = first["question"]["id"]
    original = first["runs"][0]
    retried = client.post(f"/questions/{qid}/runs/{original['id']}/retry")
    assert retried.status_code == 200
    second = retried.json()["runs"][0]
    assert second["id"] != original["id"]
    assert second["chunk_set_id"] == original["chunk_set_id"]
    assert second["rule_params"] == original["rule_params"]
    assert database.get_selection(original["id"]).status == "complete"
    assert len(client.get(f"/questions/{qid}/runs").json()["runs"]) == 2


def test_rechunk_rejects_rejected_workspace_and_old_retry(isolated):
    main, database, config = isolated
    client = TestClient(main.app)
    rejected = client.post("/workspaces", files={"files": ("bad.pdf", b"invalid pdf", "application/pdf")}).json()
    assert rejected["status"] == "rejected"
    assert client.post(f"/workspaces/{rejected['id']}/rechunk").status_code == 409
    workspace = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("one two"), "application/pdf")}).json()
    first = client.post(f"/workspaces/{workspace['id']}/questions", json={"question": "q", "mode": "full"}).json()
    sid = first["runs"][0]["id"]
    assert client.post(f"/workspaces/{workspace['id']}/rechunk").status_code == 200
    assert client.post(f"/selections/{sid}/answers/retry").status_code == 409
    assert client.post(f"/questions/{first['question']['id']}/runs/{sid}/retry").status_code == 409


def test_comparison_immutable_snapshot_and_unknown_total_cost(isolated):
    main, database, config = isolated
    client = TestClient(main.app)
    workspace = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("one two"), "application/pdf")}).json()
    run = client.post(f"/workspaces/{workspace['id']}/questions", json={"question": "q", "mode": "full"}).json()
    created = client.post(f"/questions/{run['question']['id']}/comparisons").json()
    record = created["records"][0]
    assert record["selection_cost"] == 0
    assert record["generation_cost"] is None and record["total_cost"] is None
    assert record["total_cost_status"] == "not_calculated"
    assert record["measured_stage_total_ms"] is not None
    assert "end_to_end_ms" not in record
    assert client.post(f"/workspaces/{workspace['id']}/rechunk").status_code == 200
    loaded = client.get(f"/comparisons/{created['id']}").json()
    assert loaded["records"] == created["records"]
    assert "stale" in loaded["current_validity"].values()
    assert database.get_comparison(created["id"]).records == created["records"]


def test_invalid_modes_do_not_persist_question(isolated):
    main, database, config = isolated
    client = TestClient(main.app)
    workspace = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("one two"), "application/pdf")}).json()
    assert client.post(f"/workspaces/{workspace['id']}/questions", json={"question": "q", "mode": "typo"}).status_code == 422
    assert client.post(f"/workspaces/{workspace['id']}/ask", data={"question": "q", "mode": "typo"}).status_code == 422
    assert database.list_questions(workspace["id"]) == []
