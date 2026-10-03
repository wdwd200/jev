from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    jev_api_key: str = os.getenv("JEV_API_KEY", "")
    jev_base_url: str = os.getenv("JEV_BASE_URL", "https://api.typesafe.ai")
    jev_endpoint: str = os.getenv("JEV_ENDPOINT", "/v1/systemone")
    jev_model: str = os.getenv("JEV_MODEL", "jev-latest")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    deepseek_endpoint: str = os.getenv("DEEPSEEK_ENDPOINT", "/chat/completions")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
    timeout_seconds: float = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE_TOKENS", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "0"))
    jev_threshold: float = float(os.getenv("JEV_THRESHOLD", "0.5"))
    evidence_token_budget: int = int(os.getenv("EVIDENCE_TOKEN_BUDGET", "6000"))
    workspace_token_limit: int = int(os.getenv("WORKSPACE_TOKEN_LIMIT", "800000"))
    database_path: str = os.getenv("DATABASE_PATH", str(ROOT / ".runtime" / "jev_context.sqlite3"))
    chunk_version: str = os.getenv("CHUNK_RULE_VERSION", "hybrid-512-v1")
    selection_strategy_version: str = os.getenv("SELECTION_STRATEGY_VERSION", "jev-threshold-v1")
    siliconflow_api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    siliconflow_base_url: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    standard_rag_configured: bool = os.getenv("STANDARD_RAG_CONFIGURED", "0").lower() in {"1", "true", "yes"}
    rag_embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
    rag_reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    rag_recall_k: int = int(os.getenv("RAG_RECALL_K", "20"))
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    max_pdf_pages: int = int(os.getenv("MAX_PDF_PAGES", "100"))


settings = Settings()
