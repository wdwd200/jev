from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Callable

import requests

from .answering import answer_with_deepseek as answer_with_deepseek
from .chunking import chunk_document, chunk_documents
from .config import Settings
from .documents import blocks_from_text, extract_pdf
from .http_client import post_json as _post_json
from .models import Document, Question, Workspace, new_id, now
from .selection import (_safe_error, select_full as select_full, select_rag as select_rag,
                       select_with_fallback as select_with_fallback, select_with_jev as select_with_jev)
from .tokens import METHOD, count_tokens


def approximate_tokens(text: str) -> int:
    return count_tokens(text)


def chunk_rule_snapshot(config: Settings) -> dict[str, Any]:
    return {"chunk_size": config.chunk_size, "chunk_overlap": 0,
            "chunk_version": config.chunk_version, "token_count_method": METHOD}


def chunk_rule_fingerprint(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def add_document(workspace: Workspace, filename: str, content: bytes, config: Settings) -> Workspace:
    """Add one PDF to the current conversation without rechunking the others."""
    if workspace.status == "rejected" and workspace.total_tokens == 0:
        raise ValueError(workspace.reason or "Workspace is not ready")
    document = extract_pdf(filename, content, workspace.id, config.max_pdf_pages)
    added = document.token_count if document.extraction_ok else 0
    if workspace.total_tokens + added > config.workspace_token_limit:
        raise ValueError("Workspace text exceeds the token limit")
    documents = [*workspace.documents, document]
    total = sum(item.token_count for item in documents if item.extraction_ok)
    status, reason = ("rejected", "No PDF text could be extracted") if total == 0 else ("ready", None)
    chunks = list(workspace.chunks)
    if status == "ready" and document.extraction_ok and workspace.chunk_set_id:
        chunks.extend(chunk_document(document, config, workspace.chunk_set_id))
    return replace(workspace, total_tokens=total, status=status, reason=reason, documents=documents, chunks=chunks)


def remove_document(workspace: Workspace, document_id: str) -> Workspace:
    """Remove one PDF and only the chunks that belong to it."""
    documents = [item for item in workspace.documents if item.id != document_id]
    if len(documents) == len(workspace.documents):
        raise ValueError("Document was not found")
    chunks = [item for item in workspace.chunks if item.document_id != document_id]
    total = sum(item.token_count for item in documents if item.extraction_ok)
    status, reason = ("rejected", "No PDF text could be extracted") if total == 0 else ("ready", None)
    return replace(workspace, total_tokens=total, status=status, reason=reason, documents=documents, chunks=chunks)


def create_workspace_from_text(name: str, text: str, config: Settings) -> Workspace:
    """Build a workspace from text that a benchmark has already extracted.

    This is the same chunking path as an uploaded PDF. It exists so a text
    benchmark is not squeezed through a one-page synthetic PDF.
    """
    workspace_id = new_id("ws")
    blocks = blocks_from_text(text)
    body = "\n\n".join(block.text for block in blocks)
    document = Document(new_id("doc"), workspace_id, name, body, approximate_tokens(body),
                        bool(blocks), None, b"", blocks)
    return _workspace_from_documents(workspace_id, [document], config)


def create_workspace(files: list[tuple[str, bytes]], config: Settings) -> Workspace:
    workspace_id = new_id("ws")
    docs = [extract_pdf(name, content, workspace_id, config.max_pdf_pages) for name, content in files]
    return _workspace_from_documents(workspace_id, docs, config)


def _workspace_from_documents(workspace_id: str, docs: list[Document], config: Settings) -> Workspace:
    total = sum(doc.token_count for doc in docs)
    if total == 0:
        status, reason = "rejected", "No PDF text could be extracted"
    elif total > config.workspace_token_limit:
        status, reason = "rejected", "Workspace text exceeds the token limit"
    else:
        status, reason = "ready", None
    chunk_set_id = new_id("chunks") if status == "ready" else None
    chunks = chunk_documents(docs, config, chunk_set_id) if chunk_set_id else []
    snapshot = chunk_rule_snapshot(config)
    return Workspace(workspace_id, now(), total, status, reason, chunk_set_id, docs, chunks,
                     snapshot, chunk_rule_fingerprint(snapshot), snapshot["token_count_method"])


def rechunk_workspace(workspace: Workspace, config: Settings) -> Workspace:
    if workspace.status != "ready":
        raise ValueError(workspace.reason or "Workspace is not ready")
    snapshot = chunk_rule_snapshot(config)
    chunk_set_id = new_id("chunks")
    chunks = chunk_documents(workspace.documents, config, chunk_set_id)
    return replace(workspace, chunk_set_id=chunk_set_id, chunks=chunks,
                   chunk_rule_snapshot=snapshot,
                   chunk_rule_fingerprint=chunk_rule_fingerprint(snapshot),
                   token_count_method=snapshot["token_count_method"])


def create_question(workspace: Workspace, text: str, history: list[tuple[str, str]] | None = None,
                    benchmark: bool = False,
                    call: Callable[[str, str, dict[str, Any], float], dict[str, Any]] | None = None,
                    config: Settings | None = None) -> Question:
    if workspace.status != "ready":
        raise ValueError(workspace.reason or "Workspace is not ready")
    if not text.strip():
        raise ValueError("Question is required")
    question = Question(new_id("q"), workspace.id, text.strip(), now(), rewritten=text.strip())
    if benchmark or not history:
        return question
    config = config or Settings()
    call = call or post_json
    transcript = "\n".join(f"User: {item}\nAnswer: {answer}" for item, answer in history)
    payload = {"model": config.deepseek_model, "temperature": 0, "stream": False,
               "thinking": {"type": "disabled"},
               "messages": [{"role": "user", "content":
                             "Rewrite the latest user question as one standalone question. "
                             "Only fill pronouns and omissions from the conversation. "
                             "Do not answer and do not add facts.\n"
                             f"{transcript}\nLatest question: {question.text}"}]}
    try:
        response = call(f"{config.deepseek_base_url.rstrip('/')}/{config.deepseek_endpoint.lstrip('/')}",
                        config.deepseek_api_key, payload, config.timeout_seconds)
        content = response["choices"][0]["message"]["content"].strip()
        if not content:
            raise ValueError("empty rewrite")
        question.rewritten = content
        question.rewrite_status = "rewritten"
        question.rewrite_usage = response.get("usage", {})
    except Exception as exc:
        question.rewrite_status = "fallback"
        question.rewrite_usage = {"error": _safe_error(exc, config)}
    return question


def post_json(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    """Compatibility entry. Tests replace this name on the services module."""
    return _post_json(url, api_key, payload, timeout)


