from __future__ import annotations

import math
import time
from typing import Any, Callable

import requests

from .config import Settings
from .models import Chunk, Question, Selection, Workspace, new_id, now


def select_full(workspace: Workspace, question: Question, config: Settings) -> Selection:
    started = time.perf_counter()
    selection = Selection(new_id("sel"), question.id, workspace.id, "full", "full-v1", "processing",
                          model_name="none", chunk_set_id=workspace.chunk_set_id,
                          rule_params={"mode": "full", **_workspace_rule_params(workspace)}, prompt_version="none-v1",
                          input_validity="valid" if workspace.chunk_rule_fingerprint else "legacy")
    valid, reason = _validate_selection_inputs(workspace, question)
    if not valid or not workspace.chunks:
        selection.status, selection.reason = "failed", reason or workspace.reason or "No chunks available"
    else:
        selection.selected_chunks = [_selected_row(c, None) for c in workspace.chunks]
        selection.total_tokens = sum(c.token_count for c in workspace.chunks)
        selection.status = "complete"
    selection.elapsed_ms = round((time.perf_counter() - started) * 1000)
    source_tokens = sum(c.token_count for c in workspace.chunks)
    selection.usage = {"external_service": False, "source_tokens": source_tokens,
                       "body_tokens_approx": workspace.total_tokens,
                       "input_chunk_set_id": workspace.chunk_set_id,
                       "selection_cost": 0.0, "cost_status": "calculated"}
    return selection


def select_rag(workspace: Workspace, question: Question, config: Settings,
               embedder: Any | None = None, reranker: Any | None = None) -> Selection:
    selection = Selection(new_id("sel"), question.id, workspace.id, "rag", "rag-bge-v1", "processing",
                          model_name=f"{config.rag_embedding_model}+{config.rag_reranker_model}",
                          chunk_set_id=workspace.chunk_set_id,
                          rule_params={"configured": config.standard_rag_configured,
                                       "recall_k": config.rag_recall_k, "top_k": config.rag_top_k,
                                       "embedding_model": config.rag_embedding_model,
                                       "reranker_model": config.rag_reranker_model,
                                       **_workspace_rule_params(workspace)},
                          prompt_version="rag-bge-v1",
                          input_validity="valid" if workspace.chunk_rule_fingerprint else "legacy")
    valid, reason = _validate_selection_inputs(workspace, question)
    if not config.standard_rag_configured:
        selection.status, selection.reason = "failed", "Standard retrieval is not configured"
        selection.strategy_version = "rag-unconfigured-v1"
        selection.prompt_version = "unconfigured"
        selection.usage = {"configured": False, "input_chunk_set_id": workspace.chunk_set_id,
                           "cost_status": "not_calculated"}
        return selection
    if not valid or not workspace.chunks:
        selection.status, selection.reason = "failed", reason or workspace.reason or "No chunks available"
        selection.usage = {"configured": True, "input_chunk_set_id": workspace.chunk_set_id,
                           "cost_status": "not_calculated"}
        return selection
    started = time.perf_counter()
    try:
        from .retrieval import retrieve_standard
        ranked = retrieve_standard(workspace.chunks, question.rewritten or question.text, config, embedder=embedder, reranker=reranker)
        selection.selected_chunks = ranked["selected"]
        selection.total_tokens = sum(row["token_count"] for row in ranked["selected"])
        selection.status = "complete" if ranked["selected"] else "empty"
        selection.reason = None if ranked["selected"] else "No chunk was returned by standard retrieval"
        selection.usage = {"configured": True, "input_chunk_set_id": workspace.chunk_set_id,
                           "source_tokens": sum(c.token_count for c in workspace.chunks),
                           "body_tokens_approx": workspace.total_tokens,
                           "candidate_scores": ranked["candidates"],
                           "recall_k": config.rag_recall_k, "top_k": config.rag_top_k,
                           "selection_cost": 0.0, "cost_status": "calculated"}
    except Exception as exc:
        selection.status, selection.reason = "failed", _safe_error(exc, config)
        selection.usage = {"configured": True, "input_chunk_set_id": workspace.chunk_set_id,
                           "cost_status": "not_calculated"}
    selection.elapsed_ms = round((time.perf_counter() - started) * 1000)
    return selection


def select_with_jev(workspace: Workspace, question: Question | str, config: Settings,
                    call: Callable[[str, str, dict[str, Any], float], dict[str, Any]] | None = None) -> Selection:
    if isinstance(question, str):
        question = Question(new_id("q"), workspace.id, question, now())
    selection = Selection(new_id("sel"), question.id, workspace.id, "jev", config.selection_strategy_version,
                          "processing", threshold=config.jev_threshold, budget=config.evidence_token_budget,
                          model_name=config.jev_model, chunk_set_id=workspace.chunk_set_id,
                          rule_params={"threshold": config.jev_threshold,
                                       "evidence_token_budget": config.evidence_token_budget,
                                       **_workspace_rule_params(workspace)},
                          prompt_version="jev-relevance-v1", input_validity="valid" if workspace.chunk_rule_fingerprint else "legacy")
    valid, reason = _validate_selection_inputs(workspace, question)
    if not valid or not workspace.chunks:
        selection.status, selection.reason = "failed", reason or workspace.reason or "No chunks available"
        return selection
    if not config.jev_api_key:
        selection.status, selection.reason = "failed", "JEV_API_KEY is not configured"
        return selection
    started = time.perf_counter()
    call = call or _default_call()
    candidates: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    try:
        for chunk in workspace.chunks:
            payload = {"state": chunk.text, "model": config.jev_model,
                       "questions": {"relevance": {"type": "noul", "instructions":
                                      f"Is this chunk useful evidence to answer: {question.rewritten or question.text}"}}}
            try:
                response = call(_jev_url(config), config.jev_api_key, payload, config.timeout_seconds)
                score = float(response["answers"]["relevance"]["noul"])
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Jev relevance probability must be finite and within [0, 1]")
                candidates.append({**_selected_row(chunk, score), "usage": response.get("usage", {}),
                                   "decision": "scored"})
            except (requests.RequestException, KeyError, TypeError, ValueError, AttributeError) as exc:
                errors.append({"chunk_id": chunk.id, "error": _safe_error(exc, config)})
        # The source document/chunk order is the deterministic tie breaker.
        source_order = {doc.id: index for index, doc in enumerate(workspace.documents)}
        candidates.sort(key=lambda row: (-row["score"], source_order.get(row["document_id"], len(source_order)), row["ordinal"], row["chunk_id"]))
        used = 0
        cropped: list[str] = []
        for row in candidates:
            if row["score"] >= config.jev_threshold and used + row["token_count"] <= config.evidence_token_budget:
                selection.selected_chunks.append(row)
                used += row["token_count"]
                row["decision"] = "selected"
            elif row["score"] >= config.jev_threshold:
                cropped.append(row["chunk_id"])
                row["decision"] = "budget_cropped"
            else:
                row["decision"] = "below_threshold"
        selection.total_tokens = used
        selection.status = "complete" if selection.selected_chunks else "empty"
        selection.reason = None if selection.selected_chunks else ("No chunk fits the evidence token budget" if cropped else "No chunk reached the Jev threshold")
        if errors and not candidates:
            selection.status, selection.reason = "failed", "Jev failed for every candidate chunk"
        elif errors and candidates:
            selection.status, selection.reason = "partial", "Some chunks could not be scored"
        selection.usage.update({"per_chunk": [{"chunk_id": row["chunk_id"], "score": row["score"],
                                                "decision": row["decision"], **row.get("usage", {})} for row in candidates],
                                "candidate_scores": candidates, "errors": errors,
                                "budget_cropped_chunk_ids": cropped,
                                "input_chunk_set_id": workspace.chunk_set_id,
                                "source_tokens": sum(c.token_count for c in workspace.chunks),
                                "body_tokens_approx": workspace.total_tokens,
                                "cost_status": "not_calculated"})
    except Exception as exc:
        selection.status, selection.reason = "failed", _safe_error(exc, config)
    selection.elapsed_ms = round((time.perf_counter() - started) * 1000)
    return selection


def select_with_fallback(workspace: Workspace, question: Question, config: Settings,
                         call: Callable[[str, str, dict[str, Any], float], dict[str, Any]] | None = None,
                         embedder: Any | None = None, reranker: Any | None = None) -> Selection:
    """Use Jev, and use standard retrieval only when Jev cannot complete."""
    selection = select_with_jev(workspace, question, config, call)
    if selection.status == "complete" or not config.standard_rag_configured:
        return selection
    fallback = select_rag(workspace, question, config, embedder, reranker)
    fallback.usage["fallback_from"] = selection.status
    fallback.usage["fallback_reason"] = selection.reason
    return fallback


def _default_call() -> Callable[[str, str, dict[str, Any], float], dict[str, Any]]:
    # Tests replace services.post_json. Read it when the call happens.
    from . import services
    return services.post_json


def _jev_url(config: Settings) -> str:
    return f"{config.jev_base_url.rstrip('/')}/{config.jev_endpoint.lstrip('/')}"


def _workspace_rule_params(workspace: Workspace) -> dict[str, Any]:
    if workspace.chunk_rule_snapshot and workspace.chunk_rule_fingerprint:
        return {"chunk_rule_snapshot": workspace.chunk_rule_snapshot,
                "chunk_rule_fingerprint": workspace.chunk_rule_fingerprint,
                "token_count_method": workspace.token_count_method}
    return {"chunk_rule_snapshot": "unknown", "chunk_rule_fingerprint": "legacy",
            "token_count_method": "legacy"}


def _safe_error(message: Any, config: Settings) -> str:
    text = str(message)
    for secret in (config.jev_api_key, config.deepseek_api_key, config.siliconflow_api_key):
        if secret:
            text = text.replace(secret, "[redacted]")
    return text


def _selected_row(chunk: Chunk, score: float | None) -> dict[str, Any]:
    return {"chunk_id": chunk.id, "score": score, "text": chunk.text, "token_count": chunk.token_count,
            "document_id": chunk.document_id, "ordinal": chunk.ordinal, "kind": chunk.kind,
            "source": chunk.source, "has_image": bool(chunk.image)}


def _validate_selection_inputs(workspace: Workspace, question: Question) -> tuple[bool, str | None]:
    if workspace.status != "ready":
        return False, workspace.reason or "Workspace is not ready"
    if question.workspace_id != workspace.id:
        return False, "Question does not belong to this workspace"
    if workspace.chunk_set_id is None:
        return False, "Workspace has no active chunk set"
    if any(chunk.chunk_set_id != workspace.chunk_set_id for chunk in workspace.chunks):
        return False, "Workspace contains chunks outside the active chunk set"
    if any(chunk.document_id not in {doc.id for doc in workspace.documents} for chunk in workspace.chunks):
        return False, "Workspace contains chunks for an unknown document"
    return True, None
