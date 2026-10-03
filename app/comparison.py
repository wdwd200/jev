from __future__ import annotations

from typing import Any

from .models import AnswerRun, Comparison, Question, Selection, Workspace, new_id, now
from .scoring import best_evidence_position_f1, best_token_f1, chunk_location, latency_summary, percentile


def build_comparison(question: Question, workspace: Workspace, selections: list[Selection], answers: list[AnswerRun]) -> Comparison:
    answer_by_selection: dict[str, list[AnswerRun]] = {s.id: [] for s in selections}
    for answer in answers:
        answer_by_selection.setdefault(answer.selection_id, []).append(answer)
    records: list[dict[str, Any]] = []
    reasons: list[str] = []
    strategies = {s.strategy for s in selections}
    for required in ("full", "jev", "rag"):
        if required not in strategies:
            reasons.append(f"missing mode: {required}")
    full_context = sum(c.token_count for c in workspace.chunks)
    generation_bases: set[tuple[str, str | None]] = set()
    for selection in selections:
        runs = answer_by_selection.get(selection.id, [])
        latest = runs[-1] if runs else None
        current = selection.chunk_set_id == workspace.chunk_set_id and selection.input_validity == "valid"
        if not current:
            reasons.append(f"{selection.strategy} input is not current/known")
        if selection.status != "complete":
            reasons.append(f"{selection.strategy} selection status={selection.status}")
        if not latest:
            reasons.append(f"{selection.strategy} answer is missing")
        elif latest.status != "complete":
            reasons.append(f"{selection.strategy} answer status={latest.status}")
        if latest and latest.status == "complete":
            generation_bases.add((latest.model_name, latest.prompt_version))
        source_tokens = selection.usage.get("source_tokens")
        answer_stage = latest.elapsed_ms if latest else None
        records.append({
            "strategy": selection.strategy,
            "selection_id": selection.id,
            "selection_status": selection.status,
            "selection_strategy_version": selection.strategy_version,
            "chunk_set_id": selection.chunk_set_id,
            "input_rule_snapshot": selection.rule_params.get("chunk_rule_snapshot", "unknown"),
            "input_rule_fingerprint": selection.rule_params.get("chunk_rule_fingerprint", "legacy"),
            "input_validity": selection.input_validity,
            "body_tokens_approx": workspace.total_tokens,
            "full_context_tokens_approx": source_tokens,
            "selected_context_tokens_approx": selection.total_tokens,
            "savings_ratio": (round(1 - selection.total_tokens / source_tokens, 4) if source_tokens else None),
            "selection_elapsed_ms": selection.elapsed_ms,
            "answers": [{"id": a.id, "status": a.status, "prompt_version": a.prompt_version,
                         "input_tokens": a.input_tokens, "output_tokens": a.output_tokens,
                         "usage": a.usage, "cost": a.cost, "cost_status": a.cost_status,
                         "elapsed_ms": a.elapsed_ms} for a in runs],
            "latest_answer_id": latest.id if latest else None,
            "answer_status": latest.status if latest else None,
            "generation_input_tokens": latest.input_tokens if latest else None,
            "generation_output_tokens": latest.output_tokens if latest else None,
            "generation_model": latest.model_name if latest else None,
            "generation_prompt_version": latest.prompt_version if latest else None,
            "generation_usage_sources": latest.usage if latest else {"input_tokens_source": "unavailable", "output_tokens_source": "unavailable"},
            "answer_elapsed_ms": answer_stage,
            "measured_stage_total_ms": (selection.elapsed_ms + answer_stage) if selection.elapsed_ms is not None and answer_stage is not None else None,
            "cost": latest.cost if latest else None,
            "cost_status": latest.cost_status if latest else "not_calculated",
            "selection_cost": selection.usage.get("selection_cost", 0.0 if selection.strategy == "full" else None),
            "selection_cost_status": selection.usage.get("cost_status", "calculated" if selection.strategy == "full" else "not_calculated"),
            "generation_cost": latest.cost if latest else None,
            "generation_cost_status": latest.cost_status if latest else "not_calculated",
            "validity": "current" if current else "historical",
            "answer_f1": _answer_score(question, latest),
            "evidence_f1": _evidence_score(question, selection, latest),
        })
    if len(generation_bases) > 1:
        reasons.append("generation model or prompt versions differ")
    for record in records:
        selection_cost = record["selection_cost"]
        generation_cost = record["generation_cost"]
        record["total_cost"] = (selection_cost + generation_cost if isinstance(selection_cost, (int, float)) and isinstance(generation_cost, (int, float)) else None)
        record["total_cost_status"] = "calculated" if record["total_cost"] is not None else "not_calculated"
    status = "complete" if not reasons else "incomplete"
    return Comparison(new_id("cmp"), question.id, workspace.id, now(), status,
                      "; ".join(dict.fromkeys(reasons)) if reasons else None, records,
                      _rollup(records, "answer_f1"), _rollup(records, "evidence_f1"))


def score_run(question: Question, selection: Selection, latest: AnswerRun | None) -> dict[str, dict[str, Any]]:
    """Score one finished run. Unanswered runs stay unanswered, not zero."""
    return {"answer_f1": _answer_score(question, latest),
            "evidence_f1": _evidence_score(question, selection, latest)}


def _answer_score(question: Question, latest: AnswerRun | None) -> dict[str, Any]:
    references = _references(question)
    if not latest or latest.status != "complete":
        return {"status": "unanswered", "value": None}
    if not references:
        return {"status": "not_run", "value": None}
    return {"status": "scored", "value": best_token_f1(latest.answer, references)}


def _evidence_score(question: Question, selection: Selection, latest: AnswerRun | None) -> dict[str, Any]:
    evidence_sets = _evidence_sets(question)
    if not latest or latest.status != "complete":
        return {"status": "unanswered", "value": None}
    if not evidence_sets:
        return {"status": "not_run", "value": None}
    return {"status": "scored", "value": best_evidence_position_f1(_locations(selection), evidence_sets)}


def _references(question: Question) -> list[str]:
    raw = question.rewrite_usage.get("references")
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _evidence_sets(question: Question) -> list[list[dict]]:
    raw = question.rewrite_usage.get("evidence_positions")
    if not isinstance(raw, list) or not raw:
        return []
    if isinstance(raw[0], dict):
        raw = [raw]
    sets: list[list[dict]] = []
    for group in raw:
        if not isinstance(group, list):
            continue
        items = [item for item in group if isinstance(item, dict)]
        if items:
            sets.append(items)
    return sets


def _locations(selection: Selection) -> list[dict]:
    return [chunk_location(row) for row in selection.selected_chunks]


def latency_percentiles(records: list[dict[str, Any]]) -> dict[str, Any]:
    """p50 and p95 across the runs in one comparison. Means are not included."""
    rows = [{"strategy": record["strategy"],
             "selection_ms": record.get("selection_elapsed_ms"),
             "total_ms": record.get("measured_stage_total_ms")} for record in records]
    return {"by_strategy": latency_summary(rows),
            "selection_p50": percentile([row["selection_ms"] for row in rows], 50),
            "selection_p95": percentile([row["selection_ms"] for row in rows], 95),
            "total_p50": percentile([row["total_ms"] for row in rows if row["total_ms"] is not None], 50),
            "total_p95": percentile([row["total_ms"] for row in rows if row["total_ms"] is not None], 95)}


def _rollup(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    scored = [record[key]["value"] for record in records
              if record.get(key, {}).get("status") == "scored"]
    if not scored:
        pending = any(record.get(key, {}).get("status") == "unanswered" for record in records)
        return {"status": "unanswered" if pending else "not_run", "value": None, "by_strategy": {}}
    by_strategy = {record["strategy"]: record[key] for record in records if key in record}
    return {"status": "scored", "value": round(sum(scored) / len(scored), 4), "by_strategy": by_strategy}


def comparison_payload(comparison: Comparison) -> dict[str, Any]:
    return {"id": comparison.id, "question_id": comparison.question_id,
            "workspace_id": comparison.workspace_id, "created_at": comparison.created_at,
            "status": comparison.status, "reason": comparison.reason,
            "records": comparison.records, "answer_f1": comparison.answer_f1,
            "evidence_f1": comparison.evidence_f1,
            "latency": latency_percentiles(comparison.records)}
