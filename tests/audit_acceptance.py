"""Offline acceptance audit, explicitly invoked; failures expose delivery gaps.

Run: python tests/audit_acceptance.py
This is separate from the original regression suite. All external calls are
mocked, databases are temporary, and no existing workspace is read or changed.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


@pytest.fixture
def h(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "bootstrap.sqlite3"))
    from app import main, services
    from app.config import settings
    from app.database import Database
    from app.models import Question
    from fastapi.testclient import TestClient
    from test_local import pdf_bytes

    config = replace(settings, jev_api_key="offline-jev", deepseek_api_key="offline-generator",
                     chunk_size=2, chunk_overlap=0, chunk_version="audit-words-v1",
                     workspace_token_limit=1000, jev_threshold=0.5, evidence_token_budget=100,
                     standard_rag_configured=False)
    database = Database(tmp_path / "audit.sqlite3")
    calls = []

    def block_network(*args, **kwargs):
        raise AssertionError("Audit must not call external services")

    def good_call(url, key, payload, timeout):
        calls.append(payload)
        if "questions" in payload:
            return {"answers": {"relevance": {"noul": 0.9}}}
        return {"choices": [{"message": {"content": "alpha"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 1}}

    monkeypatch.setattr(services.requests, "post", block_network)
    monkeypatch.setattr(services, "post_json", good_call)
    monkeypatch.setattr(main, "db", database)
    monkeypatch.setattr(main, "settings", config)

    def workspace(text="alpha beta gamma delta", cfg=None):
        return services.create_workspace([("audit.pdf", pdf_bytes(text))], cfg or config)

    def question(ws):
        return Question("audit-question", ws.id, "What?", "audit-time")

    return SimpleNamespace(main=main, services=services, config=config, db=database,
                           Database=Database, client=TestClient(main.app), calls=calls,
                           workspace=workspace, question=question, pdf_bytes=pdf_bytes)


def test_pass_continuous_questions_survive_database_reopen(h, monkeypatch):
    ws = h.workspace()
    h.db.save_workspace(ws)
    for prompt in ("First?", "Second?"):
        result = h.client.post(f"/workspaces/{ws.id}/questions",
                               json={"question": prompt, "mode": "full"})
        assert result.status_code == 200
        assert result.json()["runs"][0]["answer"]["status"] == "complete"
    reopened = h.Database(h.db.path)
    assert len(reopened.list_questions(ws.id)) == 2
    assert [c.id for c in reopened.get_workspace(ws.id).chunks] == [c.id for c in ws.chunks]
    monkeypatch.setattr(h.main, "db", reopened)
    q = reopened.list_questions(ws.id)[0]
    assert h.client.get(f"/questions/{q.id}/runs").json()["runs"][0]["answer"]["text"] == "alpha"


def test_pass_partial_scoring_does_not_call_generator(h, monkeypatch):
    ws = h.workspace()
    h.db.save_workspace(ws)
    count = 0

    def partial(url, key, payload, timeout):
        nonlocal count
        assert "questions" in payload, "Generator must not run after partial scoring"
        count += 1
        if count == 2:
            raise h.services.requests.RequestException("offline simulated failure")
        return {"answers": {"relevance": {"noul": 0.9}}}

    monkeypatch.setattr(h.services, "post_json", partial)
    result = h.client.post(f"/workspaces/{ws.id}/questions", json={"question": "What?", "mode": "jev"})
    run = result.json()["runs"][0]
    assert run["status"] == "partial" and run["answer"] is None
    assert count == len(ws.chunks)


def test_pass_empty_scoring_does_not_call_generator(h, monkeypatch):
    ws = h.workspace()
    h.db.save_workspace(ws)

    def low(url, key, payload, timeout):
        assert "questions" in payload
        return {"answers": {"relevance": {"noul": 0.1}}}

    monkeypatch.setattr(h.services, "post_json", low)
    run = h.client.post(f"/workspaces/{ws.id}/questions", json={"question": "What?", "mode": "jev"}).json()["runs"][0]
    assert run["status"] == "empty" and run["answer"] is None


def test_pass_generator_transport_failure_keeps_selection(h, monkeypatch):
    ws = h.workspace()
    h.db.save_workspace(ws)

    def offline(*args):
        raise h.services.requests.RequestException("offline simulated failure")

    monkeypatch.setattr(h.services, "post_json", offline)
    run = h.client.post(f"/workspaces/{ws.id}/questions", json={"question": "What?", "mode": "full"}).json()["runs"][0]
    assert run["status"] == "complete" and run["answer"]["status"] == "failed"
    assert h.db.get_selection(run["id"]).status == "complete"


def test_pass_rejected_workspace_creates_no_question(h):
    ws = h.workspace(cfg=replace(h.config, workspace_token_limit=1))
    h.db.save_workspace(ws)
    result = h.client.post(f"/workspaces/{ws.id}/questions", json={"question": "What?", "mode": "all"})
    assert result.status_code == 409
    assert not h.db.list_questions(ws.id) and not h.calls


def test_gap_unknown_mode_is_rejected_before_running(h):
    ws = h.workspace()
    h.db.save_workspace(ws)
    response = h.client.post(f"/workspaces/{ws.id}/questions", json={"question": "What?", "mode": "typo"})
    assert response.status_code in (400, 422), f"Unknown mode ran {[r['strategy'] for r in response.json()['runs']]}"
    assert not h.db.list_questions(ws.id) and not h.calls


def test_gap_full_savings_uses_matching_context_basis(h, monkeypatch):
    config = replace(h.config, chunk_size=3, chunk_overlap=1)
    monkeypatch.setattr(h.main, "settings", config)
    ws = h.workspace("one two three four five", config)
    h.db.save_workspace(ws)
    run = h.client.post(f"/workspaces/{ws.id}/questions", json={"question": "What?", "mode": "full"}).json()["runs"][0]
    assert run["savings_ratio"] == 0, {k: run[k] for k in ("source_tokens", "selected_tokens", "savings_ratio")}


@pytest.mark.parametrize("response", [{}, {"choices": [{"message": {"content": ""}}]}])
def test_gap_empty_provider_response_is_not_success(h, response):
    ws = h.workspace()
    question = h.question(ws)
    selection = h.services.select_full(ws, question, h.config)
    answer = h.services.answer_with_deepseek(selection, question, h.config, lambda *args: response)
    assert answer.status == "failed", f"status={answer.status}, answer={answer.answer!r}"


@pytest.mark.parametrize("score", [2.0, float("nan")])
def test_gap_invalid_probability_is_service_failure(h, score):
    ws = h.workspace("alpha beta")
    result = h.services.select_with_jev(ws, h.question(ws), h.config,
                                       lambda *args: {"answers": {"relevance": {"noul": score}}})
    assert result.status == "failed", f"Invalid probability was treated as {result.status}"


def test_gap_all_candidate_scores_are_retained(h):
    ws = h.workspace()
    scores = iter([0.9, 0.1])
    q = h.question(ws)
    result = h.services.select_with_jev(ws, q, h.config,
                                       lambda *args: {"answers": {"relevance": {"noul": next(scores)}}})
    h.db.save_workspace(ws)
    h.db.save_question(q)
    h.db.save_selection(result)
    found = set()

    def inspect(value):
        if isinstance(value, dict):
            if "chunk_id" in value and "score" in value:
                found.add(value["chunk_id"])
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    inspect(asdict(h.db.get_selection(result.id)))
    assert len(found) == len(ws.chunks), f"Retained scores: {len(found)} / {len(ws.chunks)}"


def test_gap_empty_selection_records_input_chunk_set(h):
    ws = h.workspace()
    q = h.question(ws)
    result = h.services.select_with_jev(ws, q, h.config,
                                       lambda *args: {"answers": {"relevance": {"noul": 0.1}}})
    h.db.save_workspace(ws)
    h.db.save_question(q)
    h.db.save_selection(result)
    assert ws.chunk_set_id in json.dumps(asdict(h.db.get_selection(result.id))), "No input chunk-set snapshot on empty selection"


def test_gap_cross_workspace_question_is_rejected(h):
    ws = h.workspace()
    wrong_question = replace(h.question(ws), workspace_id="another-workspace")
    result = h.services.select_full(ws, wrong_question, h.config)
    assert result.status == "failed", "Selection accepted a question belonging to another workspace"


def test_gap_mixed_chunk_sets_are_rejected(h):
    ws = h.workspace()
    ws.chunks[0].chunk_set_id = "another-chunk-set"
    result = h.services.select_full(ws, h.question(ws), h.config)
    assert result.status == "failed", "Selection accepted chunks outside the active chunk set"


def test_gap_equal_scores_use_stable_document_order(h):
    ws = h.workspace()
    ws.chunks[0].id = "chunk_z"
    ws.chunks[1].id = "chunk_a"
    result = h.services.select_with_jev(ws, h.question(ws), replace(h.config, evidence_token_budget=2),
                                       lambda *args: {"answers": {"relevance": {"noul": 0.9}}})
    assert result.selected_chunks[0]["text"] == ws.chunks[0].text, "Equal-score truncation follows random identifier instead of source order"


def test_gap_persistence_preserves_document_context_order(h):
    ws = h.services.create_workspace([("first.pdf", h.pdf_bytes("first document")),
                                      ("second.pdf", h.pdf_bytes("second document"))], h.config)
    for doc, stable_id in zip(ws.documents, ("doc_z", "doc_a")):
        old_id = doc.id
        doc.id = stable_id
        for chunk in ws.chunks:
            if chunk.document_id == old_id:
                chunk.document_id = stable_id
    h.db.save_workspace(ws)
    loaded = h.db.get_workspace(ws.id)
    assert [c.text for c in loaded.chunks] == [c.text for c in ws.chunks], "Reload reordered documents by random identifier"


def test_gap_database_closes_connections_after_operation(h, monkeypatch):
    import sqlite3

    connections = []
    original_connect = h.db.connect

    def track():
        connection = original_connect()
        connections.append(connection)
        return connection

    monkeypatch.setattr(h.db, "connect", track)
    h.db.list_questions("nonexistent-workspace")
    open_connections = 0
    try:
        for connection in connections:
            try:
                connection.execute("SELECT 1")
                open_connections += 1
            except sqlite3.ProgrammingError:
                pass
        assert open_connections == 0, f"{open_connections} connection(s) still open after operation"
    finally:
        for connection in connections:
            connection.close()


if __name__ == "__main__":
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="acceptance-", dir=runtime) as directory:
        raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "--basetemp", str(Path(directory) / "cases")]))
