from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from pypdf import PdfReader
import io


@dataclass
class BenchmarkItem:
    source: str
    item_id: str
    question: str
    answers: list[str]
    evidence: list[str]
    document_name: str
    document_text: str
    document_bytes: bytes = b""
    notes: dict[str, Any] = field(default_factory=dict)


def select_qasper(limit: int = 5, split: str = "train") -> list[BenchmarkItem]:
    from datasets import load_dataset
    rows = load_dataset("NomaDamas/qasper", split=split)
    chosen: list[BenchmarkItem] = []
    for row in rows:
        if _uses_figure(row.get("retrieval_gt") or []):
            continue
        evidence = _qasper_evidence(row)
        answers = [text for text in row.get("answer_gt") or [] if str(text).strip()]
        if not evidence or not answers:
            continue
        text = _qasper_text(row)
        if len(text.split()) < 200:
            continue
        chosen.append(BenchmarkItem("qasper", f"{row['id']}#{len(chosen)}", row["question"][0]
                                    if isinstance(row["question"], list) else row["question"],
                                    answers, evidence, f"{row['id']}.txt", text,
                                    notes={"title": row.get("title", "")}))
        if len(chosen) >= limit:
            break
    return chosen


def select_multihop(limit: int = 5) -> list[BenchmarkItem]:
    questions = json.loads(Path(_hub_file("yixuantt/MultiHopRAG", "MultiHopRAG.json")).read_text(encoding="utf-8"))
    corpus = json.loads(Path(_hub_file("yixuantt/MultiHopRAG", "corpus.json")).read_text(encoding="utf-8"))
    by_url = {row.get("url"): row for row in corpus}
    chosen: list[BenchmarkItem] = []
    for row in questions:
        evidence_rows = []
        for item in row.get("evidence_list") or []:
            article = by_url.get(item.get("url"))
            if article and article.get("body"):
                evidence_rows.append(article)
        if len(evidence_rows) < 2 or not str(row.get("answer") or "").strip():
            continue
        parts = [f"{article['title']}\n{article['body']}" for article in evidence_rows]
        chosen.append(BenchmarkItem("multihop-rag", row.get("query", "")[:80], row["query"],
                                    [row["answer"]], [article["body"] for article in evidence_rows],
                                    "news.txt", "\n\n".join(parts),
                                    notes={"question_type": row.get("question_type", "")}))
        if len(chosen) >= limit:
            break
    return chosen


def select_financebench(limit: int = 5, download: bool = True) -> list[BenchmarkItem]:
    path = Path(_hub_file("PatronusAI/financebench", "financebench_merged.jsonl"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen: list[BenchmarkItem] = []
    for row in rows:
        evidence = [item.get("evidence_text", "") for item in row.get("evidence") or [] if isinstance(item, dict)]
        evidence = [text.strip() for text in evidence if text and text.strip()]
        if not evidence or not row.get("doc_link") or not str(row.get("answer") or "").strip():
            continue
        content = b""
        text = ""
        if download:
            try:
                response = requests.get(row["doc_link"], timeout=30, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
                if response.status_code != 200 or response.content[:4] != b"%PDF":
                    continue
                content = response.content
                reader = PdfReader(io.BytesIO(content))
                text = "\n\n".join((page.extract_text() or "").strip() for page in reader.pages).strip()
            except Exception:
                continue
            if len(text.split()) < 200:
                continue
        chosen.append(BenchmarkItem("financebench", row["financebench_id"], row["question"],
                                    [str(row["answer"])], evidence, f"{row['doc_name']}.pdf", text, content,
                                    notes={"company": row.get("company", ""), "doc_link": row.get("doc_link", "")}))
        if len(chosen) >= limit:
            break
    return chosen


def _qasper_text(row: dict[str, Any]) -> str:
    full_text = row.get("full_text") or {}
    paragraphs = full_text.get("paragraphs") or []
    sections = []
    for paragraph_group in paragraphs:
        if isinstance(paragraph_group, list):
            sections.extend(str(item) for item in paragraph_group)
        elif paragraph_group:
            sections.append(str(paragraph_group))
    return "\n\n".join(part for part in [row.get("title", ""), row.get("abstract", ""), *sections] if part)


def _qasper_evidence(row: dict[str, Any]) -> list[str]:
    found: list[str] = []
    qas = row.get("qas") or {}
    for answer_group in qas.get("answers") or []:
        for answer in answer_group.get("answer") or []:
            for evidence in answer.get("evidence") or []:
                if str(evidence).strip():
                    found.append(str(evidence).strip())
    return found


def _uses_figure(targets: list[Any]) -> bool:
    text = json.dumps(targets, ensure_ascii=False).lower()
    return any(marker in text for marker in (".png", "figure", "table"))


def _hub_file(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset")
