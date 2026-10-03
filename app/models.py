from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4



def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


@dataclass
class Block:
    kind: str
    text: str
    page: int
    index: int
    headings: list[str] = field(default_factory=list)
    image: bytes = b""


@dataclass
class Document:
    id: str
    workspace_id: str
    filename: str
    text: str
    token_count: int
    extraction_ok: bool
    error: str | None = None
    content: bytes = b""
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    chunk_set_id: str
    document_id: str
    ordinal: int
    text: str
    token_count: int
    version: str
    kind: str = "text"
    source: dict[str, Any] = field(default_factory=dict)
    image: bytes = b""


@dataclass
class Workspace:
    id: str
    created_at: str
    total_tokens: int
    status: str
    reason: str | None = None
    chunk_set_id: str | None = None
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    chunk_rule_snapshot: dict[str, Any] = field(default_factory=dict)
    chunk_rule_fingerprint: str | None = None
    token_count_method: str = "approximate_words-v1"


@dataclass
class Question:
    id: str
    workspace_id: str
    text: str
    created_at: str
    status: str = "created"
    rewritten: str = ""
    rewrite_status: str = "original"
    rewrite_usage: dict[str, Any] = field(default_factory=dict)


@dataclass
class Selection:
    id: str
    question_id: str
    workspace_id: str
    strategy: str
    strategy_version: str
    status: str
    selected_chunks: list[dict[str, Any]] = field(default_factory=list)
    total_tokens: int = 0
    usage: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    threshold: float | None = None
    budget: int | None = None
    model_name: str | None = None
    elapsed_ms: int | None = None
    # Immutable input snapshot fields.  They are appended for backwards
    # compatibility with callers that construct Selection positionally.
    chunk_set_id: str | None = None
    rule_params: dict[str, Any] = field(default_factory=dict)
    prompt_version: str | None = None
    input_validity: str = "unknown"


@dataclass
class AnswerRun:
    id: str
    selection_id: str
    status: str
    answer: str = ""
    model_name: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None
    elapsed_ms: int | None = None
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    cost_status: str = "not_calculated"
    prompt_version: str | None = None


@dataclass
class Comparison:
    id: str
    question_id: str
    workspace_id: str
    created_at: str
    status: str
    reason: str | None = None
    records: list[dict[str, Any]] = field(default_factory=list)
    answer_f1: dict[str, Any] = field(default_factory=lambda: {"status": "not_run", "value": None})
    evidence_f1: dict[str, Any] = field(default_factory=lambda: {"status": "not_run", "value": None})
