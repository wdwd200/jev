"""Independent frontend checks for boundaries found while reviewing repairs.

Run explicitly: python -X utf8 tests/audit_frontend_followup.py
Uses the isolated offline fixture from the initial acceptance audit.
"""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from audit_acceptance import h  # noqa: E402,F401 -- shared offline fixture


def test_tied_scores_across_documents_follow_upload_order(h):
    ws = h.services.create_workspace([("first.pdf", h.pdf_bytes("first source")),
                                      ("second.pdf", h.pdf_bytes("second source"))], h.config)
    ws.chunks[0].id = "chunk_z"
    ws.chunks[1].id = "chunk_a"
    selection = h.services.select_with_jev(ws, h.question(ws), replace(h.config, evidence_token_budget=2),
                                           lambda *args: {"answers": {"relevance": {"noul": 0.9}}})
    assert selection.selected_chunks[0]["text"] == "first source"


@pytest.mark.parametrize("response", [None, [], {"choices": [None]}, {"choices": [{"message": None}]}])
def test_invalid_provider_structure_records_failure(h, response):
    ws = h.workspace()
    question = h.question(ws)
    selection = h.services.select_full(ws, question, h.config)
    answer = h.services.answer_with_deepseek(selection, question, h.config, lambda *args: response)
    assert answer.status == "failed"
    assert selection.status == "complete"


def test_generator_rejects_selection_from_another_question(h):
    ws = h.workspace()
    question = h.question(ws)
    selection = h.services.select_full(ws, question, h.config)

    def forbidden(*args):
        raise AssertionError("Mismatched input must not reach the generator")

    answer = h.services.answer_with_deepseek(selection, replace(question, id="another-question"), h.config, forbidden)
    assert answer.status == "failed"


def test_fallback_input_estimate_includes_prompt_and_question(h):
    ws = h.workspace()
    question = h.question(ws)
    selection = h.services.select_full(ws, question, h.config)
    answer = h.services.answer_with_deepseek(selection, question, h.config,
                                            lambda *args: {"choices": [{"message": {"content": "alpha"}}]})
    assert answer.status == "complete"
    assert answer.input_tokens > selection.total_tokens, "Evidence-only length omits the actual request prompt and question"


def test_failed_answer_retains_reported_usage(h):
    ws = h.workspace()
    question = h.question(ws)
    selection = h.services.select_full(ws, question, h.config)
    answer = h.services.answer_with_deepseek(selection, question, h.config,
                                            lambda *args: {"choices": [{"message": {"content": ""}}],
                                                           "usage": {"prompt_tokens": 42, "completion_tokens": 1}})
    assert answer.status == "failed"
    assert answer.input_tokens == 42 and answer.output_tokens == 1


if __name__ == "__main__":
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frontend-review-", dir=runtime) as directory:
        raise SystemExit(pytest.main([__file__, "-q", "--tb=short", "--basetemp", str(Path(directory) / "cases")]))
