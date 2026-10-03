"""Run the fixed 15-item slice through Full, standard retrieval, and Jev.

The slice is five text-answerable items from each of QASPER, FinanceBench, and
MultiHop-RAG. Results are written under .runtime and include answers, scores,
token counts, and failures. This is a pipeline check, not a pass/fail claim.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.benchmarks import select_financebench, select_multihop, select_qasper
from app.config import settings
from app.scoring import best_token_f1, chunk_location, evidence_locations, evidence_position_f1, latency_summary
from app.services import (answer_with_deepseek, create_question, create_workspace_from_text,
                         select_full, select_rag, select_with_jev)


def main() -> None:
    config = replace(settings, standard_rag_configured=True, chunk_size=256, chunk_overlap=32,
                     evidence_token_budget=1200, timeout_seconds=180)
    items = [*select_qasper(5), *select_multihop(5), *_finance_from_disk(5)]
    if len(items) != 15:
        raise SystemExit(f"Expected 15 benchmark items, found {len(items)}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / ".runtime" / f"benchmark-slice-{stamp}"
    out.mkdir(parents=True)
    report = []
    latency_rows = []
    for index, item in enumerate(items, start=1):
        print(f"[{index}/15] {item.source} {item.item_id}", flush=True)
        workspace = create_workspace_from_text(item.document_name, item.document_text, config)
        if workspace.status != "ready":
            report.append({"source": item.source, "id": item.item_id, "status": "document_rejected",
                           "reason": workspace.reason})
            _write(out, report)
            continue
        question = create_question(workspace, item.question, benchmark=True)
        catalog = [{"text": chunk.text, "kind": chunk.kind, "source": chunk.source}
                   for chunk in workspace.chunks]
        gold_locations = evidence_locations(catalog, item.evidence)
        row = {"source": item.source, "id": item.item_id, "question": item.question,
               "references": item.answers, "evidence_locations": gold_locations,
               "chunk_count": len(workspace.chunks),
               "document_tokens_approx": workspace.total_tokens, "modes": {}}
        for mode, selector in (("full", select_full), ("rag", select_rag), ("jev", select_with_jev)):
            selection = selector(workspace, question, config)
            mode_row = {"selection_status": selection.status, "reason": selection.reason,
                        "selected_tokens": selection.total_tokens, "selection_ms": selection.elapsed_ms}
            latency = {"source": item.source, "strategy": mode, "selection_ms": selection.elapsed_ms,
                       "total_ms": None}
            if selection.status == "complete":
                answer = answer_with_deepseek(selection, question, config)
                total_ms = None
                if selection.elapsed_ms is not None and answer.elapsed_ms is not None:
                    total_ms = selection.elapsed_ms + answer.elapsed_ms
                latency["total_ms"] = total_ms
                selected_locations = [chunk_location(chunk) for chunk in selection.selected_chunks]
                if answer.status != "complete":
                    evidence_score, evidence_status = None, "unanswered"
                elif not gold_locations:
                    evidence_score, evidence_status = None, "not_run"
                else:
                    evidence_score = evidence_position_f1(selected_locations, gold_locations)
                    evidence_status = "scored"
                mode_row.update({"answer_status": answer.status, "answer": answer.answer,
                                 "error": answer.error, "input_tokens": answer.input_tokens,
                                 "output_tokens": answer.output_tokens, "answer_ms": answer.elapsed_ms,
                                 "total_ms": total_ms,
                                 "answer_f1": best_token_f1(answer.answer, item.answers) if answer.status == "complete" else None,
                                 "evidence_f1": evidence_score,
                                 "evidence_status": evidence_status})
            latency_rows.append(latency)
            row["modes"][mode] = mode_row
            print(f"  {mode}: {selection.status}", flush=True)
        report.append(row)
        _write(out, report, latency_rows)
    print(out / "results.json")


def _finance_from_disk(limit: int):
    from app.benchmarks import BenchmarkItem
    folder = ROOT / ".runtime" / "finance_candidates"
    items = []
    for meta_path in sorted(folder.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        text_path = meta_path.with_suffix(".txt")
        pdf_path = meta_path.with_suffix(".pdf")
        if not text_path.exists():
            continue
        items.append(BenchmarkItem("financebench", meta["id"], meta["question"], [meta["answer"]],
                                   meta["evidence"], pdf_path.name, text_path.read_text(encoding="utf-8"),
                                   pdf_path.read_bytes() if pdf_path.exists() else b"",
                                   {"company": meta.get("company", "")}))
        if len(items) >= limit:
            break
    return items


def _write(folder: Path, report: list[dict], latency_rows: list[dict] | None = None) -> None:
    (folder / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if latency_rows is None:
        return
    grouped: dict[str, list[dict]] = {}
    for row in latency_rows:
        grouped.setdefault(row["source"], []).append(row)
    summary = {source: latency_summary(rows) for source, rows in grouped.items()}
    (folder / "latency.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    sys.path.insert(0, str(ROOT))
    main()
