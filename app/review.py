from __future__ import annotations

import json
import time
from typing import Any, Callable

import requests

from .chunking import _chunk_block
from .config import Settings
from .database import Database
from .documents import page_count, recognize_page
from .models import Block, Document, new_id, now
from .retrieval import retrieve_standard
from .selection import _safe_error
from .tokens import count_tokens


CHECKLISTS = {
    "commercial": ["合同当事人", "生效日期和期限", "付款金额和付款时间", "交付内容", "违约责任", "赔偿和责任限制", "解除和终止", "争议解决"],
    "rental": ["当事人", "租赁物", "租期", "租金和支付", "押金", "维修责任", "解除", "违约"],
    "labor": ["当事人", "合同期限", "工作内容", "工作时间和休息", "劳动报酬", "社会保险", "解除和终止", "违约和竞业"],
}
CONCLUSIONS = {"找到", "冲突", "缺失"}
MAX_ROUNDS = 3


def accept_uploads(files: list[tuple[str, bytes]], contract_type: str, database: Database, config: Settings,
                   removed: list[str] | None = None, added: list[str] | None = None) -> dict[str, Any]:
    if contract_type not in CHECKLISTS:
        raise ValueError("未知的合同类型")
    batch_id = new_id("batch")
    stored = []
    for filename, content in files:
        if len(content) > config.max_upload_bytes:
            raise ValueError("PDF 超过 20MB")
        count = page_count(content)
        if count > config.max_pdf_pages:
            raise ValueError("PDF 超过 100 页")
        file_id = new_id("file")
        pages = [(new_id("page"), number) for number in range(1, count + 1)]
        stored.append((file_id, filename or "contract.pdf", content, count, pages))
    database.create_review_batch(batch_id, now(), contract_type, config.workspace_token_limit, stored)
    if removed or added:
        database.save_risks(_risk_rows(batch_id, contract_type, removed or [], added or []))
    return database.review_progress(batch_id)


def _risk_rows(batch_id: str, contract_type: str, removed: list[str], added: list[str]) -> list[tuple]:
    names = [name for name in CHECKLISTS[contract_type] if name not in set(removed)]
    rows = [(new_id("risk"), batch_id, name, name, name, index, 0) for index, name in enumerate(names, start=1)]
    for offset, text in enumerate(added, start=1):
        cleaned = " ".join(text.split())
        if cleaned:
            rows.append((new_id("risk"), batch_id, cleaned, cleaned, cleaned, len(names) + offset, 1))
    return rows


def work_once(database: Database, config: Settings, chunk_set_id: str,
              call: Callable[[str, str, dict[str, Any], float], dict[str, Any]] | None = None) -> bool:
    page = database.claim_review_page()
    if page:
        _recognize(database, page)
        return True
    chunk_page = database.claim_chunk_page(chunk_set_id)
    if chunk_page:
        _chunk_page(database, config, chunk_set_id, chunk_page)
        database.refresh_batch_tokens(chunk_page["batch_id"], chunk_set_id, config.workspace_token_limit)
        if database.batch_needs_risks(chunk_page["batch_id"]):
            batch = database.review_batch(chunk_page["batch_id"])
            database.save_risks(_risk_rows(chunk_page["batch_id"], batch["contract_type"], [], []))
        return True
    risk = database.claim_risk()
    if risk:
        _review_risk(database, config, risk, call)
        return True
    return False


def _recognize(database: Database, page: Any) -> None:
    try:
        blocks = recognize_page(bytes(page["content"]), int(page["page_number"]))
        database.finish_review_page(page["id"], blocks)
    except Exception as exc:
        database.fail_review_page(page["id"], "这一页没有识别出来")


def _chunk_page(database: Database, config: Settings, chunk_set_id: str, page: Any) -> None:
    content = database.review_page_content(page["id"])
    document = Document(page["file_id"], page["batch_id"], "", "", 0, True)
    rows = []
    ordinal = 0
    for item in content:
        block = Block(item["kind"], item.get("text", ""), int(page["page_number"]), int(item["position"]), item.get("headings") or [])
        for chunk in _chunk_block(document, block, config, chunk_set_id, ordinal):
            source = dict(chunk.source)
            source["page_id"] = page["id"]
            source["file_id"] = page["file_id"]
            rows.append((chunk.id, page["batch_id"], chunk_set_id, page["file_id"], page["id"], chunk.ordinal,
                         chunk.kind, chunk.text, chunk.token_count, json.dumps(source, ensure_ascii=False),
                         chunk.image or None, chunk.version))
            ordinal += 1
    if not rows:
        rows.append((new_id("chunk"), page["batch_id"], chunk_set_id, page["file_id"], page["id"], 0,
                     "image", "", 0, json.dumps({"page_id": page["id"], "page_start": page["page_number"], "page_end": page["page_number"], "position": 1, "kind": "image"}, ensure_ascii=False),
                     None, config.chunk_version))
    database.replace_page_chunks(page["id"], chunk_set_id, rows)


def _review_risk(database: Database, config: Settings, risk: Any,
                 call: Callable[[str, str, dict[str, Any], float], dict[str, Any]] | None) -> None:
    from .models import Chunk

    chunks = []
    for row in database.batch_chunks(risk["batch_id"], risk["active_chunk_set_id"]):
        chunks.append(Chunk(row["id"], row["chunk_set_id"], row["file_id"], row["ordinal"], row["text"],
                            row["token_count"], row["version"], row["kind"], json.loads(row["source"]),
                            bytes(row["image"]) if row["image"] else b""))
    prior = database.prior_queries(risk["id"])
    round_number = len(prior) + 1
    if round_number > MAX_ROUNDS:
        database.finish_risk(risk["id"], "failed")
        return
    started = time.perf_counter()
    try:
        found = retrieve_standard(chunks, risk["retrieval_query"], config)
        status, error = ("empty" if not found["selected"] else "complete"), None
        calls = 2 if chunks else 0
    except Exception as exc:
        found = {"selected": [], "candidates": []}
        status, error, calls = "failed", _safe_error(exc, config), 0
    elapsed = round((time.perf_counter() - started) * 1000)
    database.save_retrieval((new_id("ret"), risk["id"], round_number, risk["retrieval_query"], "rerank", status,
                             json.dumps(found["candidates"], ensure_ascii=False), json.dumps(found["selected"], ensure_ascii=False),
                             error, calls, elapsed))
    if status == "failed":
        database.save_decision((new_id("dec"), risk["id"], round_number, "stop", "", None, "检索失败", "[]", "failed"))
        database.finish_risk(risk["id"], "failed")
        return
    decision = _decide(risk, found["selected"], prior, round_number, config, call)
    database.save_decision((new_id("dec"), risk["id"], round_number, decision["action"], decision["next_query"],
                            decision["conclusion"], decision["reason"], json.dumps(decision["pages"]), "complete"))
    if decision["action"] == "continue" and round_number < MAX_ROUNDS:
        database.finish_risk(risk["id"], "waiting", decision["next_query"])
        return
    database.finish_risk(risk["id"], "complete")
    _answer(database, config, risk, decision, found["selected"], call)


def _decide(risk: Any, selected: list[dict[str, Any]], prior: list[str], round_number: int, config: Settings,
            call: Callable[[str, str, dict[str, Any], float], dict[str, Any]] | None) -> dict[str, Any]:
    pages = sorted({int(row["source"].get("page_start", 0)) for row in selected if row.get("source")})
    if not selected and round_number >= 2:
        return {"action": "stop", "next_query": "", "conclusion": "缺失", "reason": "两轮都没有留下块", "pages": []}
    if round_number >= MAX_ROUNDS and not selected:
        return {"action": "stop", "next_query": "", "conclusion": "缺失", "reason": "第三轮证据不足", "pages": []}
    if call is None:
        from . import services
        call = services.post_json
    evidence = "\n\n".join(f"第{row['source'].get('page_start')}页\n{row['text']}" for row in selected) or "这一轮没有留下块"
    payload = {"model": config.deepseek_model, "temperature": 0, "stream": False, "thinking": {"type": "disabled"},
               "messages": [{"role": "user", "content":
                             "只输出 JSON。字段是 action、next_query、conclusion、reason。"
                             "action 只能是 stop 或 continue。conclusion 只能是找到、冲突、缺失。"
                             "不能要求改变块数。下一句不能和已查过的句子相同。\n"
                             f"风险项：{risk['name']}\n已查：{' | '.join(prior + [risk['retrieval_query']])}\n证据：\n{evidence}"}]}
    try:
        response = call(f"{config.deepseek_base_url.rstrip('/')}/{config.deepseek_endpoint.lstrip('/')}",
                        config.deepseek_api_key, payload, config.timeout_seconds)
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content[content.find("{"):content.rfind("}") + 1])
        action = parsed.get("action") if parsed.get("action") in {"stop", "continue"} else "stop"
        conclusion = parsed.get("conclusion") if parsed.get("conclusion") in CONCLUSIONS else ("缺失" if not selected else "找到")
        next_query = " ".join(str(parsed.get("next_query", "")).split())
        if action == "continue" and (not next_query or next_query in prior or next_query == risk["retrieval_query"] or round_number >= MAX_ROUNDS):
            action, next_query = "stop", ""
            if conclusion not in CONCLUSIONS:
                conclusion = "缺失"
        return {"action": action, "next_query": next_query, "conclusion": conclusion,
                "reason": str(parsed.get("reason", ""))[:500], "pages": pages}
    except (requests.RequestException, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError):
        conclusion = "缺失" if not selected or round_number >= MAX_ROUNDS else "找到"
        return {"action": "stop", "next_query": "", "conclusion": conclusion, "reason": "模型没有给出可用决定", "pages": pages if conclusion != "缺失" else []}


def _answer(database: Database, config: Settings, risk: Any, decision: dict[str, Any], selected: list[dict[str, Any]],
            call: Callable[[str, str, dict[str, Any], float], dict[str, Any]] | None) -> None:
    if call is None:
        from . import services
        call = services.post_json
    context = "\n\n".join(f"第{row['source'].get('page_start')}页\n{row['text']}" for row in selected)
    prompt = ("用引用内容说明三件事：结论、支撑结论的页码、不能从引用推出的部分。不能补充引用里没有的条款。\n"
              f"风险项：{risk['name']}\n原问法：{risk['original_query']}\n结论：{decision['conclusion']}\n引用：\n{context}")
    started = time.perf_counter()
    text, error, usage = "", None, {"input_tokens_source": "unavailable", "output_tokens_source": "unavailable", "images_sent": False}
    status = "failed"
    in_tokens = out_tokens = 0
    try:
        response = call(f"{config.deepseek_base_url.rstrip('/')}/{config.deepseek_endpoint.lstrip('/')}",
                        config.deepseek_api_key, {"model": config.deepseek_model, "temperature": 0, "stream": False,
                                                  "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": prompt}]},
                        config.timeout_seconds)
        text = response["choices"][0]["message"]["content"].strip()
        reported = response.get("usage") or {}
        in_tokens = int(reported.get("prompt_tokens", count_tokens(prompt)))
        out_tokens = int(reported.get("completion_tokens", count_tokens(text)))
        usage["input_tokens_source"] = "reported" if "prompt_tokens" in reported else "estimated"
        usage["output_tokens_source"] = "reported" if "completion_tokens" in reported else "estimated"
        status = "complete"
    except (requests.RequestException, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        error = _safe_error(exc, config)
    database.save_review_answer((new_id("ans"), risk["id"], status, text, config.deepseek_model, "contract-review-v1",
                                 in_tokens, out_tokens, json.dumps(usage, ensure_ascii=False), "not_calculated",
                                 round((time.perf_counter() - started) * 1000), error, 0))
