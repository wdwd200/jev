from __future__ import annotations

import math
import re
import string
from collections import Counter


def token_f1(prediction: str, reference: str) -> float:
    """SQuAD-style word overlap, returned on a 0-100 scale.

    A question with several acceptable references uses the highest score.
    Empty prediction and empty reference score 100; one side empty scores 0.
    """
    predicted = _normalize(prediction)
    gold = _normalize(reference)
    if not predicted and not gold:
        return 100.0
    if not predicted or not gold:
        return 0.0
    common = Counter(predicted) & Counter(gold)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 100.0 * (2 * precision * recall) / (precision + recall)


def best_token_f1(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0
    return max(token_f1(prediction, reference) for reference in references)


def best_evidence_position_f1(selected: list[dict], evidence_sets: list[list[dict]]) -> float:
    """Highest set F1 across several acceptable evidence annotations."""
    if not evidence_sets:
        return 0.0
    return max(evidence_position_f1(selected, evidence) for evidence in evidence_sets)


def evidence_f1(selected_texts: list[str], evidence_texts: list[str]) -> float:
    """Legacy word overlap. Proof scores use evidence_position_f1 instead."""
    return token_f1("\n".join(selected_texts), "\n".join(evidence_texts))


def evidence_position_f1(selected: list[dict], evidence: list[dict]) -> float:
    """QASPER-style set F1 over page, kind, and position.

    A selected block matches one gold item when those three fields agree.
    Multiple gold sets are not handled here; the caller keeps the best score.
    """
    if not selected and not evidence:
        return 100.0
    if not selected or not evidence:
        return 0.0
    chosen = {_key(item) for item in selected}
    gold = {_key(item) for item in evidence}
    overlap = len(chosen & gold)
    if overlap == 0:
        return 0.0
    precision = overlap / len(chosen)
    recall = overlap / len(gold)
    return 100.0 * (2 * precision * recall) / (precision + recall)


def evidence_locations(chunks: list[dict], evidence_texts: list[str]) -> list[dict]:
    """Map evidence text onto chunk locations.

    Downloaded benchmarks store evidence as passages, not page numbers.
    A chunk matches when the shorter side shares at least half its words.
    The returned locations are page, kind, and in-page position.
    """
    found: list[dict] = []
    seen: set[tuple] = set()
    for evidence in evidence_texts:
        gold = _normalize(evidence)
        if not gold:
            continue
        for chunk in chunks:
            words = _normalize(str(chunk.get("text") or ""))
            if not words or not _mostly_shared(words, gold):
                continue
            location = chunk_location(chunk)
            if _key(location) not in seen:
                seen.add(_key(location))
                found.append(location)
    return found


def chunk_location(chunk: dict) -> dict:
    source = chunk.get("source") if isinstance(chunk.get("source"), dict) else {}
    return {"page": source.get("page_start", source.get("page")),
            "kind": source.get("kind", chunk.get("kind")),
            "position": source.get("position")}


def percentile(values: list[float], percent: float) -> float | None:
    """Nearest-rank percentile. An empty list has no percentile."""
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return None
    rank = min(len(ordered), max(1, math.ceil((percent / 100) * len(ordered) - 1e-9)))
    return float(ordered[rank - 1])


def latency_summary(rows: list[dict]) -> dict[str, dict[str, float | int | None]]:
    """p50 and p95 for selection time and question-to-answer time.

    Rows belong to one library. Each row names a strategy and may omit a time
    when that stage did not finish. The mean is intentionally not reported.
    """
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        bucket = grouped.setdefault(row["strategy"], {"selection_ms": [], "total_ms": []})
        if row.get("selection_ms") is not None:
            bucket["selection_ms"].append(row["selection_ms"])
        if row.get("total_ms") is not None:
            bucket["total_ms"].append(row["total_ms"])
    return {strategy: {"selection_p50": percentile(bucket["selection_ms"], 50),
                       "selection_p95": percentile(bucket["selection_ms"], 95),
                       "total_p50": percentile(bucket["total_ms"], 50),
                       "total_p95": percentile(bucket["total_ms"], 95),
                       "count": len(bucket["total_ms"])}
            for strategy, bucket in grouped.items()}


def _mostly_shared(left: list[str], right: list[str]) -> bool:
    overlap = sum((Counter(left) & Counter(right)).values())
    shorter = min(len(left), len(right))
    return shorter > 0 and overlap / shorter >= 0.5


def _key(item: dict) -> tuple:
    return (item.get("page"), item.get("kind"), item.get("position"))


def _normalize(text: str) -> list[str]:
    text = text.lower().replace("-", " ")
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return text.split()
