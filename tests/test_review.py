from __future__ import annotations

import json

import pymupdf
from fastapi.testclient import TestClient

from app.config import settings
from app.database import Database
from app.main import app
from app.models import new_id
from app.review import accept_uploads, work_once


def _pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def _call(url: str, key: str, payload: dict, timeout: float) -> dict:
    prompt = payload["messages"][0]["content"]
    if "只输出 JSON" in prompt:
        content = json.dumps({"action": "stop", "next_query": "", "conclusion": "找到", "reason": "页内有约定"}, ensure_ascii=False)
    else:
        content = "结论是找到。依据第1页。不能推出付款金额。"
    return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 12, "completion_tokens": 8}}


def test_review_keeps_one_page_and_finishes_one_risk(monkeypatch, tmp_path):
    database = Database(tmp_path / "review.sqlite3")
    monkeypatch.setattr("app.retrieval.retrieve_standard", lambda chunks, question, config: {
        "selected": [{"text": chunks[0].text, "source": chunks[0].source, "kind": "text", "has_image": False}],
        "candidates": [{"text": chunks[0].text, "source": chunks[0].source}],
    })
    progress = accept_uploads([("合同.pdf", _pdf("甲方应于十日内付款"))], "commercial", database, settings)
    batch_id = progress["id"]
    chunk_set = new_id("chunks")
    assert work_once(database, settings, chunk_set, _call)
    assert work_once(database, settings, chunk_set, _call)
    assert database.review_progress(batch_id)["risks"]
    assert work_once(database, settings, chunk_set, _call)
    results = database.review_results(batch_id)
    assert results[0]["conclusion"] == "找到"
    assert results[0]["pages"] == [1]
    assert "第1页" in results[0]["answer"]
    assert not database.review_progress(batch_id)["failed_pages"]


def test_home_starts_contract_review():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert "合同审查" in response.text
    assert "开始审查" in response.text
