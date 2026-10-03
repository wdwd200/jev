from __future__ import annotations

from typing import Any

import requests

from .config import Settings
from .models import Chunk


def retrieve_standard(chunks: list[Chunk], question: str, config: Settings,
                      embedder: Any | None = None, reranker: Any | None = None) -> dict[str, list[dict[str, Any]]]:
    """Recall chunks with a fixed embedding model, then keep a fixed reranked count.

    Both models are used as published. This function does not train them and does not
    change the chunk text. A retrieval failure is raised to the caller; it is not
    replaced with full context or Jev.
    """
    if not chunks:
        return {"selected": [], "candidates": []}
    embedder = embedder or _SiliconEmbedder(config)
    reranker = reranker or _SiliconReranker(config)
    query_vector = embedder.encode([question], normalize_embeddings=True)
    chunk_vectors = embedder.encode([chunk.text for chunk in chunks], normalize_embeddings=True)
    recall_scores = (chunk_vectors @ query_vector.T).reshape(-1)
    order = sorted(range(len(chunks)), key=lambda index: (-float(recall_scores[index]), chunks[index].ordinal, chunks[index].id))
    recalled = order[:max(1, config.rag_recall_k)]
    pairs = [(question, chunks[index].text) for index in recalled]
    rerank_scores = [float(score) for score in reranker.predict(pairs)]
    ranked = sorted(zip(recalled, rerank_scores), key=lambda item: (-item[1], chunks[item[0]].ordinal, chunks[item[0]].id))
    candidates = [_row(chunks[index], float(recall_scores[index]), rerank, "recalled")
                  for index, rerank in ranked]
    selected = []
    for row in candidates[:max(1, config.rag_top_k)]:
        chosen = dict(row)
        chosen["decision"] = "selected"
        selected.append(chosen)
    for row in candidates[len(selected):]:
        row["decision"] = "rerank_excluded"
    return {"selected": selected, "candidates": candidates}


class _SiliconEmbedder:
    """Call SiliconFlow instead of loading bge-m3 on this machine."""

    def __init__(self, config: Settings) -> None:
        self.config = config

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> Any:
        import numpy as np
        vectors = []
        for text in texts:
            body = _post(self.config, "/embeddings",
                         {"model": self.config.rag_embedding_model, "input": text})
            vectors.append(body["data"][0]["embedding"])
        matrix = np.array(vectors, dtype=float)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.clip(norms, 1e-9, None)
        return matrix


class _SiliconReranker:
    def __init__(self, config: Settings) -> None:
        self.config = config

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        query = pairs[0][0]
        documents = [text for _, text in pairs]
        body = _post(self.config, "/rerank",
                     {"model": self.config.rag_reranker_model, "query": query, "documents": documents})
        scores = [0.0] * len(documents)
        for item in body["results"]:
            scores[int(item["index"])] = float(item["relevance_score"])
        return scores


def _post(config: Settings, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not config.siliconflow_api_key:
        raise RuntimeError("SILICONFLOW_API_KEY is not configured")
    response = requests.post(config.siliconflow_base_url.rstrip("/") + path,
                             headers={"Authorization": f"Bearer {config.siliconflow_api_key}",
                                      "Content-Type": "application/json"},
                             json=payload, timeout=config.timeout_seconds)
    response.raise_for_status()
    return response.json()


def _row(chunk: Chunk, recall_score: float, rerank_score: float, decision: str) -> dict[str, Any]:
    return {"chunk_id": chunk.id, "score": rerank_score, "recall_score": recall_score,
            "text": chunk.text, "token_count": chunk.token_count, "document_id": chunk.document_id,
            "ordinal": chunk.ordinal, "kind": chunk.kind, "source": chunk.source,
            "has_image": bool(chunk.image), "decision": decision}
