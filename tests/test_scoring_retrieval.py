import numpy as np

from app.scoring import best_token_f1, evidence_f1, token_f1
from app.retrieval import retrieve_standard
from app.models import Chunk
from app.config import settings
from dataclasses import replace


def test_token_f1_matches_known_overlap():
    assert token_f1("the capital of France", "capital of France") == 100.0
    assert token_f1("", "") == 100.0
    assert token_f1("Paris", "") == 0.0
    assert best_token_f1("Sam Bankman Fried", ["Jane Doe", "Sam Bankman-Fried"]) == 100.0


def test_evidence_f1_uses_selected_context():
    score = evidence_f1(["alpha beta gamma"], ["beta gamma delta"])
    assert 50 < score < 100


def test_slice_evidence_uses_position_and_latency_reports_percentiles():
    from app.scoring import evidence_locations, evidence_position_f1, latency_summary, percentile
    chunks = [
        {"text": "Revenue grew in 2018 to 1577.", "kind": "table", "source": {"page_start": 4, "kind": "table", "position": 2}},
        {"text": "A note about the weather.", "kind": "text", "source": {"page_start": 1, "kind": "text", "position": 1}},
    ]
    gold = evidence_locations(chunks, ["Revenue grew in 2018 to 1577."])
    assert gold == [{"page": 4, "kind": "table", "position": 2}]
    assert evidence_position_f1(gold, gold) == 100.0
    summary = latency_summary([
        {"strategy": "full", "selection_ms": 10, "total_ms": 100},
        {"strategy": "full", "selection_ms": 30, "total_ms": 300},
        {"strategy": "full", "selection_ms": 20, "total_ms": None},
        {"strategy": "jev", "selection_ms": 40, "total_ms": 400},
    ])
    assert summary["full"]["selection_p50"] == percentile([10, 30, 20], 50) == 20
    assert summary["full"]["selection_p95"] == 30
    assert summary["full"]["total_p50"] == 100
    assert summary["full"]["total_p95"] == 300
    assert summary["full"]["count"] == 2
    assert "mean" not in summary["full"]


class _Embedder:
    def encode(self, texts, normalize_embeddings=True):
        rows = []
        for text in texts:
            rows.append([1.0 if "revenue" in text.lower() else 0.0, 1.0 if "risk" in text.lower() else 0.0])
        matrix = np.array(rows, dtype=float)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-9, None)


class _Reranker:
    def predict(self, pairs):
        return [1.0 if "revenue" in text.lower() else 0.1 for _, text in pairs]


def test_standard_retrieval_keeps_fixed_top_k_from_same_chunks():
    chunks = [Chunk(f"c{i}", "set", "doc", i, text, 3, "words-v1")
              for i, text in enumerate(["revenue grew", "risk factor", "unrelated note", "revenue risk"])]
    config = replace(settings, rag_recall_k=3, rag_top_k=2)
    result = retrieve_standard(chunks, "What is revenue?", config, embedder=_Embedder(), reranker=_Reranker())
    assert [row["text"] for row in result["selected"]] == ["revenue grew", "revenue risk"]
    assert len(result["candidates"]) == 3
