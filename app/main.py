from __future__ import annotations

from html import escape
from typing import Annotated, Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from .config import settings
from .comparison import build_comparison, comparison_payload, score_run
from .database import db
from .models import AnswerRun, Question, Selection
from .review import accept_uploads
from .services import (add_document, answer_with_deepseek, create_question, create_workspace,
                       remove_document, rechunk_workspace, select_full, select_rag, select_with_fallback,
                       select_with_jev)

app = FastAPI(title="Jev Context Selector", version="0.2.0")


def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    content = file.file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(413, "PDF exceeds the upload size limit")
    return file.filename or "document.pdf", content


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)
    mode: Literal["full", "jev", "rag", "all"] = "jev"
    references: list[str] = Field(default_factory=list)
    evidence_positions: list[Any] = Field(default_factory=list)


def _workspace_payload(workspace: Any) -> dict[str, Any]:
    return {"id": workspace.id, "created_at": workspace.created_at, "status": workspace.status,
            "reason": workspace.reason, "total_tokens": workspace.total_tokens,
            "chunk_count": len(workspace.chunks), "chunk_set_id": workspace.chunk_set_id,
            "body_tokens_approx": workspace.total_tokens,
            "chunk_rule_snapshot": workspace.chunk_rule_snapshot,
            "chunk_rule_fingerprint": workspace.chunk_rule_fingerprint,
            "token_count_method": workspace.token_count_method,
            "documents": [{"id": d.id, "filename": d.filename, "status": "ok" if d.extraction_ok else "failed",
                           "token_count": d.token_count, "error": d.error} for d in workspace.documents],
            "questions": [{"id": q.id, "question": q.text, "created_at": q.created_at}
                          for q in db.list_questions(workspace.id)]}


def _selection_payload(selection: Selection, answers: list[AnswerRun] | None = None,
                       question: Question | None = None) -> dict[str, Any]:
    history = [a for a in (answers or []) if a.selection_id == selection.id]
    answer = history[-1] if history else None
    source_tokens = selection.usage.get("source_tokens")
    result = {"id": selection.id, "strategy": selection.strategy, "strategy_version": selection.strategy_version,
              "status": selection.status, "reason": selection.reason, "threshold": selection.threshold,
              "budget": selection.budget, "model_name": selection.model_name, "selected_chunks": selection.selected_chunks,
              "selected_tokens": selection.total_tokens, "source_tokens": source_tokens,
              "savings_ratio": (round(1 - selection.total_tokens / source_tokens, 4) if source_tokens else None),
              "chunk_set_id": selection.chunk_set_id or selection.usage.get("input_chunk_set_id"),
              "rule_params": selection.rule_params,
              "prompt_version": selection.prompt_version,
              "input_validity": selection.input_validity,
              "usage": selection.usage, "elapsed_ms": selection.elapsed_ms}
    result["metrics"] = {"body_tokens_approx": selection.usage.get("body_tokens_approx"),
                          "full_context_tokens_approx": source_tokens,
                          "selected_context_tokens_approx": selection.total_tokens,
                          "selection_elapsed_ms": selection.elapsed_ms,
                          "answer_elapsed_ms": answer.elapsed_ms if answer else None,
                          "measured_stage_total_ms": (selection.elapsed_ms + answer.elapsed_ms) if answer and selection.elapsed_ms is not None and answer.elapsed_ms is not None else None,
                          **(score_run(question, selection, answer) if question else
                             {"answer_f1": {"status": "not_run", "value": None},
                              "evidence_f1": {"status": "not_run", "value": None}})}
    def answer_payload(item: AnswerRun) -> dict[str, Any]:
        return {"id": item.id, "status": item.status, "text": item.answer,
                "model_name": item.model_name, "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens, "cost": item.cost,
                "elapsed_ms": item.elapsed_ms, "error": item.error,
                "usage": item.usage, "cost_status": item.cost_status,
                "prompt_version": item.prompt_version}
    result["answer"] = answer_payload(answer) if answer else None
    result["answers"] = [answer_payload(item) for item in history]
    return result


def _comparison_payload_live(comparison: Any) -> dict[str, Any]:
    payload = comparison_payload(comparison)
    workspace = db.get_workspace(comparison.workspace_id)
    live: dict[str, str] = {}
    for record in comparison.records:
        selection = db.get_selection(record["selection_id"])
        if not workspace or not selection:
            live[record["selection_id"]] = "unknown"
        elif selection.status == "obsolete" or selection.input_validity != "valid" or selection.chunk_set_id != workspace.chunk_set_id:
            live[record["selection_id"]] = "stale"
        else:
            live[record["selection_id"]] = "current"
    payload["current_validity"] = live
    return payload


def _run_modes(question: Question, workspace: Any, mode: str) -> list[Selection]:
    requested = [mode] if mode in {"full", "jev", "rag"} else ["full", "jev", "rag"]
    selections: list[Selection] = []
    for strategy in requested:
        if strategy == "full":
            selection = select_full(workspace, question, settings)
        elif strategy == "jev":
            selection = select_with_fallback(workspace, question, settings)
        else:
            selection = select_rag(workspace, question, settings)
        selection.usage["source_tokens"] = sum(c.token_count for c in workspace.chunks)
        selection.usage["body_tokens_approx"] = workspace.total_tokens
        db.save_selection(selection)
        if selection.status == "complete":
            answer = answer_with_deepseek(selection, question, settings)
            db.save_answer(answer)
        selections.append(selection)
    return selections


def _runs_html(workspace: Any, selections: list[Selection]) -> str:
    answers = db.list_answers([s.id for s in selections])
    blocks: list[str] = []
    for selection in selections:
        payload = _selection_payload(selection, answers, db.get_question(selection.question_id))
        answer_payload = payload.get("answer") or {}
        selected_text = "\n\n".join(row["text"] for row in selection.selected_chunks)
        error = answer_payload.get("error") or selection.reason or ""
        saved = payload.get("savings_ratio")
        saved_text = "0" if saved is None else f"{round(float(saved) * 100, 1)}%"
        blocks.append(f"<section><p class='muted'>{escape(selection.strategy)}</p><h2>回答</h2>"
                      f"<div class='metrics'><div class='metric'><b>{workspace.total_tokens}</b><span>原文约 token</span></div>"
                      f"<div class='metric'><b>{selection.total_tokens}</b><span>本次送入 token</span></div>"
                      f"<div class='metric'><b>{saved_text}</b><span>少送比例</span></div></div>"
                      f"<pre>{escape(answer_payload.get('text', '') or '这次没有生成回答')}</pre>"
                      f"<h2>选用的内容</h2><pre>{escape(selected_text)}</pre>"
                      f"<p class='muted'>状态：{escape(selection.status)} / {escape(str(answer_payload.get('status') or 'none'))}</p>"
                      f"<p class='muted'>{escape(str(error))}</p></section>")
    return "<header><div><h1>结果</h1><p class='muted'>少送比例按这次参与选择的全部块计算。</p></div></header>" + "".join(blocks)


PAGE_STYLE = """
:root { color-scheme: light; --ink:#172033; --muted:#5d6b82; --line:#d9e2ef; --paper:#f5f7fb; --card:#fff; --accent:#2457d6; }
* { box-sizing: border-box; }
body { margin:0; font:16px/1.6 "Segoe UI", "PingFang SC", sans-serif; color:var(--ink); background:radial-gradient(1200px 500px at 10% -10%, #e7efff, transparent 60%), var(--paper); }
main { width:min(980px, calc(100% - 32px)); margin:32px auto 64px; }
header { display:flex; justify-content:space-between; gap:24px; align-items:end; margin-bottom:24px; }
h1 { font-size:34px; line-height:1.2; margin:0 0 8px; }
h2 { margin:0 0 12px; font-size:20px; }
p { margin:0; }
.muted { color:var(--muted); }
.card, section, article { background:var(--card); border:1px solid var(--line); border-radius:18px; padding:22px; box-shadow:0 10px 30px rgba(23,32,51,.04); }
form { display:grid; gap:16px; }
label { display:grid; gap:6px; font-weight:650; }
input, select, textarea { width:100%; border:1px solid var(--line); border-radius:12px; padding:12px 14px; font:inherit; background:#fff; }
textarea { min-height:120px; resize:vertical; }
button, .button { border:0; border-radius:999px; padding:12px 18px; background:var(--accent); color:white; font:inherit; text-decoration:none; display:inline-block; }
.metrics { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin:16px 0; }
.metric { padding:14px; border-radius:14px; background:#f7f9fd; }
.metric b { display:block; font-size:22px; }
pre { white-space:pre-wrap; background:#f7f9fd; border-radius:12px; padding:14px; }
a { color:var(--accent); }
@media (max-width: 720px) { header { display:block; } .metrics { grid-template-columns:1fr; } h1 { font-size:28px; } }
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)}</title>
    <style>{PAGE_STYLE}</style></head><body><main>{body}</main></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return _page("合同审查", """
    <header><div><p class="muted">上传后按清单逐项核对</p><h1>合同审查</h1>
    <p class="muted">每项只给出找到、冲突或缺失，并回到页码。识别和检索在后台进行。</p></div></header>
    <section class="card"><form action="/reviews" method="post" enctype="multipart/form-data">
    <label>合同 PDF<input type="file" name="files" accept="application/pdf,.pdf" multiple required></label>
    <label>合同类型<select name="contract_type"><option value="commercial">商业合同</option><option value="rental">房屋租赁</option><option value="labor">劳动合同</option></select></label>
    <button type="submit">开始审查</button></form></section>
    <p class="muted"><a href="/ask-page">原来的提问页</a></p>""")


@app.get("/ask-page", response_class=HTMLResponse)
def ask_page() -> str:
    return _page("Jev Context Selector", """
    <header><div><p class="muted">同一批 PDF，两种读取方式</p><h1>Jev Context Selector</h1>
    <p class="muted">上传文档后直接提问。Jev 只把用到的内容送进模型；整篇对照会送入全部内容。</p></div></header>
    <section class="card"><form action="/ask" method="post" enctype="multipart/form-data">
    <label>PDF<input type="file" name="files" accept="application/pdf" multiple required></label>
    <label>读取方式<select name="mode"><option value="jev">Jev 直选</option><option value="full">整篇对照</option></select></label>
    <label>问题<textarea name="question" placeholder="输入一个可以独立理解的问题" required></textarea></label>
    <button type="submit">开始提问</button></form></section>""")


@app.post("/reviews", response_model=None)
async def create_review(files: Annotated[list[UploadFile], File()],
                        contract_type: Annotated[str, Form()] = "commercial") -> HTMLResponse | RedirectResponse:
    try:
        progress = accept_uploads([_read_upload(item) for item in files], contract_type, db, settings)
    except ValueError as exc:
        return HTMLResponse(_page("无法建立审查", f"<h1>无法建立审查</h1><p>{escape(str(exc))}</p><p><a href='/'>返回</a></p>"))
    return RedirectResponse(f"/reviews/{progress['id']}", status_code=303)


@app.get("/reviews/{batch_id}", response_class=HTMLResponse)
def review_page(batch_id: str) -> str:
    progress = db.review_progress(batch_id)
    if not progress:
        raise HTTPException(404, "审查不存在")
    progress["results"] = db.review_results(batch_id)
    return _page("审查进度", _review_html(progress))


def _review_html(progress: dict[str, Any]) -> str:
    counts = progress.get("pages") or {}
    total = sum(counts.values())
    done = counts.get("complete", 0)
    failed = counts.get("failed", 0)
    risks = progress.get("results") or progress.get("risks") or []
    items = []
    for risk in risks:
        conclusion = risk.get("conclusion") or risk.get("status") or ""
        pages = "、".join(str(page) for page in risk.get("pages") or [])
        answer = risk.get("answer") or ""
        items.append(f"<article><h2>{escape(risk['name'])}</h2><p>{escape(str(conclusion))}</p>"
                     f"<p class='muted'>页码：{escape(pages or '还没有')}</p><pre>{escape(answer)}</pre></article>")
    finished = progress["status"] in {"ready", "rejected"} and risks and all(item.get("status") in {"complete", "failed"} for item in risks)
    refresh = "" if finished else "<meta http-equiv='refresh' content='3'>"
    return (f"{refresh}<header><div><h1>审查进度</h1><p class='muted'>页面 {done}/{total}，失败 {failed}</p></div></header>"
            f"<section><p>状态：{escape(progress['status'])}</p><p class='muted'>{escape(progress.get('reason') or '')}</p></section>"
            + "".join(items) + "<p><a href='/'>返回</a></p>")


@app.post("/workspaces")
async def create_workspace_api(files: Annotated[list[UploadFile], File()]) -> dict[str, Any]:
    contents = [_read_upload(file) for file in files]
    workspace = create_workspace(contents, settings)
    db.save_workspace(workspace)
    return _workspace_payload(workspace)


@app.get("/workspaces/{workspace_id}")
def get_workspace_api(workspace_id: str) -> dict[str, Any]:
    workspace = db.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    return _workspace_payload(workspace)


@app.post("/workspaces/{workspace_id}/documents")
async def add_document_api(workspace_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    workspace = db.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    before = {chunk.id for chunk in workspace.chunks}
    try:
        filename, content = _read_upload(file)
        updated = add_document(workspace, filename, content, settings)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    db.save_documents_changed(updated, [chunk for chunk in updated.chunks if chunk.id not in before])
    return _workspace_payload(updated)


@app.delete("/workspaces/{workspace_id}/documents/{document_id}")
def remove_document_api(workspace_id: str, document_id: str) -> dict[str, Any]:
    workspace = db.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    try:
        updated = remove_document(workspace, document_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    db.save_documents_changed(updated, removed_document_id=document_id)
    return _workspace_payload(updated)


@app.post("/workspaces/{workspace_id}/rechunk")
def rechunk_workspace_api(workspace_id: str) -> dict[str, Any]:
    workspace = db.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    try:
        updated = rechunk_workspace(workspace, settings)
        db.save_rechunk(workspace, updated)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return _workspace_payload(updated)


@app.post("/workspaces/{workspace_id}/questions")
def create_question_api(workspace_id: str, request: QuestionRequest) -> dict[str, Any]:
    workspace = db.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    try:
        question = create_question(workspace, request.question)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    if request.references or request.evidence_positions:
        question.rewrite_usage = {**question.rewrite_usage, "references": request.references,
                                  "evidence_positions": request.evidence_positions}
    db.save_question(question)
    selections = _run_modes(question, workspace, request.mode)
    answers = db.list_answers([s.id for s in selections])
    return {"question": {"id": question.id, "workspace_id": question.workspace_id, "text": question.text,
                         "created_at": question.created_at},
            "runs": [_selection_payload(s, answers, question) for s in selections]}


@app.get("/questions/{question_id}/runs")
def get_question_runs(question_id: str) -> dict[str, Any]:
    question = db.get_question(question_id)
    if not question:
        raise HTTPException(404, "Question not found")
    selections = db.list_selections(question_id)
    answers = db.list_answers([s.id for s in selections])
    return {"question": {"id": question.id, "workspace_id": question.workspace_id, "text": question.text},
            "runs": [_selection_payload(s, answers, question) for s in selections]}


@app.post("/questions/{question_id}/runs/{selection_id}/retry")
def retry_question_run(question_id: str, selection_id: str) -> dict[str, Any]:
    question = db.get_question(question_id)
    previous = db.get_selection(selection_id)
    if not question or not previous or previous.question_id != question_id:
        raise HTTPException(404, "Run not found")
    workspace = db.get_workspace(question.workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    if previous.status == "obsolete" or previous.input_validity != "valid":
        raise HTTPException(409, "Selection input is obsolete or legacy")
    if previous.chunk_set_id != workspace.chunk_set_id:
        raise HTTPException(409, "Selection input is no longer the active chunk set")
    runs = _run_modes(question, workspace, previous.strategy)
    answers = db.list_answers([s.id for s in runs])
    return {"question": {"id": question.id, "workspace_id": question.workspace_id, "text": question.text},
            "runs": [_selection_payload(s, answers, question) for s in runs], "superseded_selection_id": selection_id}


@app.post("/selections/{selection_id}/answers/retry")
def retry_answer(selection_id: str) -> dict[str, Any]:
    selection = db.get_selection(selection_id)
    if not selection:
        raise HTTPException(404, "Selection not found")
    question = db.get_question(selection.question_id)
    workspace = db.get_workspace(selection.workspace_id)
    if not question or not workspace:
        raise HTTPException(404, "Selection context not found")
    if selection.status != "complete":
        raise HTTPException(409, "Only complete selections can be answered")
    if selection.status == "obsolete" or selection.input_validity != "valid":
        raise HTTPException(409, "Selection input is obsolete or legacy")
    if selection.chunk_set_id != workspace.chunk_set_id:
        raise HTTPException(409, "Selection input is no longer the active chunk set")
    if not selection.prompt_version or selection.rule_params.get("chunk_rule_fingerprint") in (None, "legacy"):
        raise HTTPException(409, "Selection input version cannot be confirmed")
    answer = answer_with_deepseek(selection, question, settings)
    db.save_answer(answer)
    answers = db.list_answers([selection.id])
    payload = _selection_payload(selection, answers, question)
    return {"selection": payload, "answer": payload["answer"], "answers": payload["answers"]}


@app.post("/questions/{question_id}/comparisons")
def create_comparison(question_id: str) -> dict[str, Any]:
    question = db.get_question(question_id)
    if not question:
        raise HTTPException(404, "Question not found")
    workspace = db.get_workspace(question.workspace_id)
    selections = db.list_selections(question_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    if not selections:
        raise HTTPException(409, "No runs available for comparison")
    answers = db.list_answers([selection.id for selection in selections])
    comparison = build_comparison(question, workspace, selections, answers)
    db.save_comparison(comparison)
    return _comparison_payload_live(comparison)


@app.get("/questions/{question_id}/comparisons")
def list_comparisons(question_id: str) -> dict[str, Any]:
    if not db.get_question(question_id):
        raise HTTPException(404, "Question not found")
    return {"comparisons": [_comparison_payload_live(item) for item in db.list_comparisons(question_id)]}


@app.get("/comparisons/{comparison_id}")
def get_comparison(comparison_id: str) -> dict[str, Any]:
    comparison = db.get_comparison(comparison_id)
    if not comparison:
        raise HTTPException(404, "Comparison not found")
    return _comparison_payload_live(comparison)


@app.post("/selections/{selection_id}/obsolete")
def obsolete_selection(selection_id: str) -> dict[str, Any]:
    if not db.mark_selection_obsolete(selection_id, "Marked obsolete by caller"):
        raise HTTPException(404, "Selection not found")
    return {"id": selection_id, "status": "obsolete"}


@app.post("/ask", response_class=HTMLResponse)
async def ask(files: Annotated[list[UploadFile], File()],
              question: Annotated[str, Form()],
              mode: Annotated[Literal["full", "jev", "rag", "all"], Form()] = "jev",
              workspace_id: Annotated[str | None, Form()] = None) -> str:
    contents = [_read_upload(file) for file in files]
    workspace = create_workspace(contents, settings)
    db.save_workspace(workspace)
    if workspace.status != "ready":
        return _page("文档不可用", f"<h1>文档不可用</h1><p>{escape(workspace.reason or '')}</p><p><a href='/'>返回</a></p>")
    try:
        q = create_question(workspace, question)
    except ValueError as exc:
        return _page("问题不可用", f"<h1>问题不可用</h1><p>{escape(str(exc))}</p><p><a href='/'>返回</a></p>")
    db.save_question(q)
    selections = _run_modes(q, workspace, mode)
    answers = db.list_answers([s.id for s in selections])
    return _page("结果", _runs_html(workspace, selections) + f"<p><a class='button' href='/workspaces/{workspace.id}/ask'>继续提问</a></p>")


@app.get("/workspaces/{workspace_id}/ask", response_class=HTMLResponse)
def workspace_ask_form(workspace_id: str) -> str:
    workspace = db.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    history = "".join(f"<li><a href='/questions/{q.id}/history'>{escape(q.text)}</a></li>" for q in db.list_questions(workspace_id))
    return _page("继续提问", f"<header><div><h1>继续提问</h1><p class='muted'>仍使用这一批还在的 PDF。</p></div></header>"
            f"<section><ul>{history}</ul><form action='/workspaces/{workspace.id}/ask' method='post'>"
            "<label>读取方式<select name='mode'><option value='jev'>Jev 直选</option><option value='full'>整篇对照</option></select></label>"
            "<label>问题<textarea name='question' required></textarea></label><button type='submit'>提问</button></form></section>")


@app.get("/questions/{question_id}/history", response_class=HTMLResponse)
def question_history(question_id: str) -> str:
    question = db.get_question(question_id)
    if not question:
        raise HTTPException(404, "Question not found")
    selections = db.list_selections(question_id)
    answers = db.list_answers([item.id for item in selections])
    body = [f"<header><div><h1>提问记录</h1><p>{escape(question.text)}</p></div></header>"]
    for selection in selections:
        payload = _selection_payload(selection, answers, question)
        reason = selection.reason or ""
        selected_text = "\n\n".join(row["text"] for row in selection.selected_chunks)
        body.append(f"<section><p class='muted'>{escape(selection.strategy)} · {escape(selection.status)}</p>"
                    f"<p class='muted'>{escape(reason)}</p>"
                    f"<pre>{escape(selected_text)}</pre>")
        for item in payload["answers"]:
            detail = item.get("error") or ""
            body.append(f"<article><pre>{escape(item['text'])}</pre><p class='muted'>状态：{escape(item['status'])}</p><p class='muted'>{escape(str(detail))}</p></article>")
        body.append("</section>")
    return _page("提问记录", "".join(body))


@app.post("/workspaces/{workspace_id}/ask", response_class=HTMLResponse)
def workspace_ask(workspace_id: str, question: Annotated[str, Form()],
                  mode: Annotated[Literal["full", "jev", "rag", "all"], Form()] = "jev") -> str:
    workspace = db.get_workspace(workspace_id)
    if not workspace:
        raise HTTPException(404, "Workspace not found")
    try:
        created = create_question(workspace, question)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    db.save_question(created)
    return _page("结果", _runs_html(workspace, _run_modes(created, workspace, mode)) + f"<p><a class='button' href='/workspaces/{workspace.id}/ask'>继续提问</a></p>")
