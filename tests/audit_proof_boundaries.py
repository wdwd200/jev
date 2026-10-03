"""Independent offline checks for the remaining comparison/lifecycle boundaries."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from audit_acceptance import h  # noqa: E402,F401
from audit_lifecycle import uploaded_run  # noqa: E402


def synthetic_complete_modes(h):
    from app.models import AnswerRun
    ws = h.workspace()
    question = h.question(ws)
    base = h.services.select_full(ws, question, h.config)
    selections = [replace(base, id="proof-" + mode, strategy=mode,
                          input_validity="valid") for mode in ("full", "jev", "rag")]
    answers = [AnswerRun("answer-" + s.id, s.id, "complete", "alpha", "same-model",
                         10, 1, None, 3, usage={"input_tokens_source": "reported",
                                              "output_tokens_source": "reported"},
                         prompt_version="same-prompt-v1") for s in selections]
    return ws, question, selections, answers


def test_missing_answer_prevents_complete_comparison(h):
    from app.comparison import build_comparison
    ws, question, selections, answers = synthetic_complete_modes(h)
    comparison = build_comparison(question, ws, selections, answers[1:])
    assert comparison.status != "complete", "A completed selection without an answer is incomplete"
    assert "answer" in (comparison.reason or "").lower()
    assert comparison.records[0].get("cost") is None, "Missing Full answer does not mean free generation"
    assert comparison.records[0].get("cost_status") != "calculated"


def test_unknown_input_validity_is_not_current(h):
    from app.comparison import build_comparison
    ws, question, selections, answers = synthetic_complete_modes(h)
    selections[0].input_validity = "unknown"
    comparison = build_comparison(question, ws, selections, answers)
    assert comparison.status != "complete"
    assert comparison.records[0]["validity"] != "current"


def test_unknown_historical_context_is_not_filled_from_active_chunks(h):
    from app.comparison import build_comparison
    ws, question, selections, answers = synthetic_complete_modes(h)
    selections[0].usage = {}
    selections[0].chunk_set_id = "old-unknown-set"
    selections[0].input_validity = "legacy"
    comparison = build_comparison(question, ws, selections, answers)
    record = comparison.records[0]
    assert record["full_context_tokens_approx"] is None
    assert record["savings_ratio"] is None
    # Conversely, a genuinely recorded old count remains a known historical fact.
    selections[0].usage = {"source_tokens": 8}
    known_history = build_comparison(question, ws, selections, answers).records[0]
    assert known_history["full_context_tokens_approx"] == 8
    assert known_history["savings_ratio"] == round(1 - selections[0].total_tokens / 8, 4)


def test_generation_basis_difference_is_visible(h):
    from app.comparison import build_comparison
    ws, question, selections, answers = synthetic_complete_modes(h)
    answers[-1].model_name = "different-generation-model"
    answers[-1].prompt_version = "different-prompt"
    comparison = build_comparison(question, ws, selections, answers)
    assert comparison.status != "complete", "Different model/prompt runs are not a same-basis comparison"
    assert "different-generation-model" in json.dumps(asdict(comparison))


def test_only_one_chunk_set_remains_active_after_rechunk(h):
    ws, result = uploaded_run(h)
    response = h.client.post(f"/workspaces/{ws['id']}/rechunk")
    assert response.status_code == 200
    connection = h.db.connect()
    try:
        rows = connection.execute("SELECT id, status FROM chunk_sets WHERE workspace_id=?", (ws["id"],)).fetchall()
    finally:
        connection.close()
    assert len(rows) == 2
    active = [row["id"] for row in rows if row["status"] == "active"]
    assert active == [response.json()["chunk_set_id"]]


def test_unknown_validity_answer_retry_does_not_call_provider(h, monkeypatch):
    ws, result = uploaded_run(h)
    sid = result["runs"][0]["id"]
    uncertain = replace(h.db.get_selection(sid), id="unknown-copy", input_validity="unknown")
    h.db.save_selection(uncertain)
    assert h.db.get_selection(uncertain.id).input_validity != "valid", \
        "Persistence cannot silently promote explicit unknown validity"
    incomplete = replace(uncertain, id="incomplete-legacy-copy", chunk_set_id=None,
                         rule_params={"chunk_rule_fingerprint": "known-but-insufficient"},
                         usage={})
    h.db.save_selection(incomplete)
    connection = h.db.connect()
    try:
        connection.execute("UPDATE selections SET input_validity='unknown' WHERE id=?", (sid,))
        # A prior additive schema can have partial rule_params but no copied
        # snapshot or input-set reference. Current workspace state is not proof.
        connection.execute("UPDATE selections SET input_rule_snapshot='{}' WHERE id=?", (incomplete.id,))
        connection.commit()
    finally:
        connection.close()
    # Reopening/migrating the database must not promote unknown to valid.
    reopened = h.Database(h.db.path)
    migrated = reopened.get_selection(incomplete.id)
    assert migrated.input_validity != "valid"
    assert migrated.chunk_set_id is None
    monkeypatch.setattr(h.main, "db", reopened)
    before = len(h.calls)
    response = h.client.post(f"/selections/{sid}/answers/retry")
    assert response.status_code in (409, 422)
    assert len(h.calls) == before
    selection_retry = h.client.post(f"/questions/{result['question']['id']}/runs/{sid}/retry")
    assert selection_retry.status_code in (409, 422)
    assert len(h.calls) == before


def test_new_run_validity_matches_its_persisted_history(h):
    ws, result = uploaded_run(h)
    immediate = result["runs"]
    assert immediate[0]["input_validity"] == "valid"
    loaded = h.client.get(f"/questions/{result['question']['id']}/runs").json()
    assert loaded["runs"] == immediate


def test_direct_answer_service_rejects_obsolete_input(h):
    ws, result = uploaded_run(h)
    selection = h.db.get_selection(result["runs"][0]["id"])
    selection.input_validity = "obsolete"
    question = h.db.get_question(selection.question_id)
    before = len(h.calls)
    answer = h.services.answer_with_deepseek(selection, question, h.config)
    assert answer.status == "failed"
    assert len(h.calls) == before


def test_comparison_separates_selection_generation_and_total_costs(h):
    ws, result = uploaded_run(h)
    response = h.client.post(f"/questions/{result['question']['id']}/comparisons")
    assert response.status_code == 200
    record = response.json()["records"][0]
    assert record.get("selection_cost") == 0
    assert "generation_cost" in record and record["generation_cost"] is None
    assert "total_cost" in record and record["total_cost"] is None
    assert record.get("total_cost_status") in ("not_calculated", "unknown", "unavailable")


def test_comparison_snapshot_stays_immutable_with_live_reference_validity(h):
    ws, result = uploaded_run(h)
    response = h.client.post(f"/questions/{result['question']['id']}/comparisons")
    assert response.status_code == 200
    cid = response.json()["id"]
    snapshot = asdict(h.db.get_comparison(cid))
    assert h.client.post(f"/workspaces/{ws['id']}/rechunk").status_code == 200
    reread = h.client.get(f"/comparisons/{cid}")
    assert reread.status_code == 200
    assert asdict(h.db.get_comparison(cid)) == snapshot
    payload = reread.json()
    assert payload["records"] == snapshot["records"]
    live = json.dumps({k: v for k, v in payload.items()
                       if k not in snapshot or v != snapshot[k]}).lower()
    assert any(marker in live for marker in ("obsolete", "historical", "stale")), \
        "An immutable snapshot also needs visible current reference validity"


@pytest.mark.parametrize("stage", ["selection", "answer"])
def test_provider_error_does_not_persist_configured_secret(h, stage):
    ws = h.workspace()
    question = h.question(ws)
    def failed_call(*args):
        # Dummy fixture keys only; no real credential is read or printed.
        raise h.services.requests.RequestException(
            "provider echoed " + h.config.jev_api_key + " " + h.config.deepseek_api_key)
    if stage == "selection":
        record = h.services.select_with_jev(ws, question, h.config, failed_call)
    else:
        selection = h.services.select_full(ws, question, h.config)
        record = h.services.answer_with_deepseek(selection, question, h.config, failed_call)
    assert record.status == "failed"
    serialized = json.dumps(asdict(record))
    assert h.config.jev_api_key not in serialized
    assert h.config.deepseek_api_key not in serialized


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="proof-review-", dir=ROOT / ".runtime") as directory:
        raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "--basetemp", str(Path(directory) / "cases")]))
