from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

from app import services
from app.config import settings
from app.database import Database
from app.main import app
from app.models import Question


def pdf_bytes(text: str) -> bytes:
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [b"<</Type /Catalog /Pages 2 0 R>>",
               b"<</Type /Pages /Kids [3 0 R] /Count 1>>",
               b"<</Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
               b"/Resources<</Font<</F1 4 0 R>>>> /Contents 5 0 R>>",
               b"<</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>",
               f"<</Length {len(stream)}>>\nstream\n".encode() + stream + b"\nendstream"]
    result = b"%PDF-1.4\n"
    offsets = [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    startxref = len(result)
    result += b"xref\n0 6\n0000000000 65535 f \n"
    result += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:])
    result += f"trailer<</Size 6 /Root 1 0 R>>\nstartxref\n{startxref}\n%%EOF\n".encode()
    return result


def test_pdf_extract_and_invalid():
    document = services.extract_pdf("ok.pdf", pdf_bytes("alpha beta"), "ws")
    assert document.extraction_ok and "alpha" in document.text
    broken = services.extract_pdf("bad.pdf", b"not a pdf", "ws")
    assert not broken.extraction_ok
    empty = services.extract_pdf("empty.pdf", pdf_bytes(""), "ws")
    assert not empty.extraction_ok


def test_pdf_extract_keeps_table_cells_and_image_marker():
    import io
    import pymupdf
    from PIL import Image
    image = Image.new("RGB", (80, 30), "white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Body revenue sentence.")
    shape = page.new_shape()
    for step in range(3):
        y = 120 + step * 30
        shape.draw_line((72, y), (300, y))
    for step in range(3):
        x = 72 + step * 114
        shape.draw_line((x, 120), (x, 180))
    shape.finish(width=0.8)
    shape.commit()
    page.insert_text((80, 140), "2018")
    page.insert_text((194, 140), "1577")
    page.insert_image(pymupdf.Rect(72, 220, 220, 280), stream=buffer.getvalue())
    content = pdf.tobytes()
    pdf.close()
    document = services.extract_pdf("table.pdf", content, "ws")
    assert document.extraction_ok
    assert "Body revenue sentence." in document.text
    assert "| 2018 | 1577 |" in document.text
    assert any(block.kind == "image" and block.image for block in document.blocks)
    assert "[Image 1.1] image text was not recognized" not in document.text


def test_chunk_version_and_workspace_limit():
    config = replace(settings, chunk_size=2, chunk_overlap=1, workspace_token_limit=2, chunk_version="test-v1")
    workspace = services.create_workspace([("a.pdf", pdf_bytes("one two three"))], config)
    assert workspace.status == "rejected"
    config = replace(config, workspace_token_limit=100)
    workspace = services.create_workspace([("a.pdf", pdf_bytes("one two three"))], config)
    assert workspace.status == "ready" and workspace.chunks
    assert all(chunk.version == "test-v1" for chunk in workspace.chunks)


def test_workspace_keeps_partial_document_failure():
    config = replace(settings, workspace_token_limit=100)
    workspace = services.create_workspace([("ok.pdf", pdf_bytes("one two")), ("bad.pdf", b"broken")], config)
    assert workspace.status == "ready"
    assert sum(document.extraction_ok for document in workspace.documents) == 1
    assert len(workspace.chunks) > 0


def ready_workspace(config):
    return services.create_workspace([("a.pdf", pdf_bytes("alpha beta gamma delta"))], config)


def test_jev_threshold_empty_budget_and_service_error():
    config = replace(settings, jev_api_key="secret", chunk_size=2, chunk_overlap=0,
                     jev_threshold=.8, evidence_token_budget=2)
    workspace = ready_workspace(config)
    question = Question("q1", workspace.id, "What?", "now")
    low = services.select_with_jev(workspace, question, config,
                                   lambda *args: {"answers": {"relevance": {"noul": .1}}})
    assert low.status == "empty" and low.selected_chunks == []
    budget = services.select_with_jev(workspace, question, replace(config, jev_threshold=.1),
                                      lambda *args: {"answers": {"relevance": {"noul": .9}}})
    assert budget.status == "complete" and budget.total_tokens <= 2
    failed = services.select_with_jev(workspace, question, config,
                                      lambda *args: (_ for _ in ()).throw(services.requests.RequestException("offline")))
    assert failed.status == "failed"


def test_api_persists_continuous_questions_and_isolates_answer_failures(tmp_path, monkeypatch):
    import app.main as main
    local_db = Database(tmp_path / "test.sqlite3")
    monkeypatch.setattr(main, "db", local_db)
    config = replace(settings, jev_api_key="", deepseek_api_key="", chunk_size=20,
                     standard_rag_configured=False)
    monkeypatch.setattr(main, "settings", config)
    client = TestClient(app)
    response = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("alpha beta"), "application/pdf")})
    assert response.status_code == 200
    workspace_id = response.json()["id"]
    run = client.post(f"/workspaces/{workspace_id}/questions", json={"question": "What?", "mode": "full"})
    assert run.status_code == 200
    assert run.json()["runs"][0]["strategy"] == "full"
    assert run.json()["runs"][0]["answer"]["status"] == "failed"
    second = client.post(f"/workspaces/{workspace_id}/questions", json={"question": "Again", "mode": "jev"})
    assert second.status_code == 200
    assert len(local_db.list_questions(workspace_id)) == 2
    history = client.get(f"/questions/{second.json()['question']['id']}/runs")
    assert history.status_code == 200 and history.json()["runs"][0]["strategy"] == "jev"


def test_api_all_modes_and_page(tmp_path, monkeypatch):
    import app.main as main
    local_db = Database(tmp_path / "all.sqlite3")
    monkeypatch.setattr(main, "db", local_db)
    config = replace(settings, jev_api_key="jev", deepseek_api_key="deep", chunk_size=20,
                     standard_rag_configured=False)
    monkeypatch.setattr(main, "settings", config)

    def fake_call(url, key, payload, timeout):
        if "typesafe" in url:
            return {"answers": {"relevance": {"noul": .9}}, "usage": {"input_tokens": 2}}
        return {"choices": [{"message": {"content": "mock answer"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2}}

    monkeypatch.setattr(services, "post_json", fake_call)
    client = TestClient(app)
    assert client.get("/").status_code == 200
    created = client.post("/workspaces", files={"files": ("a.pdf", pdf_bytes("alpha beta"), "application/pdf")})
    runs = client.post(f"/workspaces/{created.json()['id']}/questions",
                       json={"question": "What?", "mode": "all"})
    assert runs.status_code == 200
    by_mode = {run["strategy"]: run for run in runs.json()["runs"]}
    assert by_mode["full"]["status"] == "complete"
    assert by_mode["jev"]["status"] == "complete"
    assert by_mode["full"]["answer"]["status"] == "complete"
    assert by_mode["rag"]["status"] == "failed"


def test_add_and_remove_document_changes_only_its_chunks():
    config = replace(settings, chunk_size=20, workspace_token_limit=100)
    workspace = services.create_workspace([("a.pdf", pdf_bytes("alpha beta"))], config)
    kept = {chunk.id for chunk in workspace.chunks}
    added = services.add_document(workspace, "b.pdf", pdf_bytes("gamma delta"), config)
    assert {chunk.id for chunk in added.chunks} >= kept
    assert len(added.documents) == 2
    new_ids = {chunk.id for chunk in added.chunks} - kept
    removed = services.remove_document(added, added.documents[-1].id)
    assert {chunk.id for chunk in removed.chunks} == kept
    assert new_ids.isdisjoint({chunk.id for chunk in removed.chunks})


def test_document_changes_persist_and_evidence_uses_position(tmp_path):
    from app.database import Database
    from app.scoring import evidence_position_f1
    config = replace(settings, chunk_size=20, workspace_token_limit=100)
    workspace = services.create_workspace([("a.pdf", pdf_bytes("alpha beta"))], config)
    database = Database(tmp_path / "docs.sqlite3")
    database.save_workspace(workspace)
    added = services.add_document(workspace, "b.pdf", pdf_bytes("gamma delta"), config)
    database.save_documents_changed(added, [chunk for chunk in added.chunks if chunk.document_id == added.documents[-1].id])
    loaded = database.get_workspace(workspace.id)
    assert len(loaded.documents) == 2
    removed = services.remove_document(loaded, loaded.documents[-1].id)
    database.save_documents_changed(removed, removed_document_id=loaded.documents[-1].id)
    final = database.get_workspace(workspace.id)
    assert len(final.documents) == 1
    assert all(chunk.document_id == final.documents[0].id for chunk in final.chunks)
    selected = [{"page": 1, "kind": "text", "position": 1}, {"page": 2, "kind": "table", "position": 1}]
    evidence = [{"page": 1, "kind": "text", "position": 1}]
    assert evidence_position_f1(selected, evidence) == 100 * 2 / 3


def test_proof_scores_three_paths_and_leaves_plain_questions_unscored():
    from app.comparison import build_comparison
    from app.models import AnswerRun, Selection
    config = replace(settings, chunk_size=50, workspace_token_limit=100)
    workspace = services.create_workspace([("paper.pdf", pdf_bytes("alpha beta"))], config)
    question = services.create_question(workspace, "What is alpha?", benchmark=True)
    question.rewrite_usage = {
        "references": ["alpha beta"],
        "evidence_positions": [[{"page": 1, "kind": "text", "position": 1}],
                               [{"page": 9, "kind": "table", "position": 3}]],
    }
    chunk = workspace.chunks[0]
    selected = [{"chunk_id": chunk.id, "text": chunk.text, "token_count": chunk.token_count,
                 "kind": chunk.kind, "source": chunk.source}]
    selections = []
    answers = []
    for strategy, text, chunks in (
        ("full", "alpha beta", selected),
        ("jev", "unrelated", []),
        ("rag", "", selected),
    ):
        selection = Selection(f"sel-{strategy}", question.id, workspace.id, strategy, f"{strategy}-v1",
                              "complete", chunks, sum(item["token_count"] for item in chunks),
                              {"source_tokens": workspace.total_tokens}, chunk_set_id=workspace.chunk_set_id,
                              input_validity="valid")
        selections.append(selection)
        status = "complete" if strategy != "rag" else "failed"
        answers.append(AnswerRun(f"ans-{strategy}", selection.id, status, text, "same-model", 4, 1,
                                 prompt_version="prompt-v1"))
    comparison = build_comparison(question, workspace, selections, answers)
    scores = {row["strategy"]: row for row in comparison.records}
    assert scores["full"]["answer_f1"] == {"status": "scored", "value": 100.0}
    assert scores["full"]["evidence_f1"]["status"] == "scored"
    assert scores["full"]["evidence_f1"]["value"] == 100.0
    assert scores["jev"]["answer_f1"]["value"] == 0.0
    assert scores["jev"]["evidence_f1"]["value"] == 0.0
    assert scores["rag"]["answer_f1"]["status"] == "unanswered"
    assert scores["rag"]["evidence_f1"]["status"] == "unanswered"
    assert comparison.answer_f1["status"] == "scored"
    assert comparison.answer_f1["value"] == 50.0
    plain = services.create_question(workspace, "后来呢")
    plain_selection = Selection("sel-plain", plain.id, workspace.id, "full", "full-v1", "complete",
                                selected, 1, {"source_tokens": 1}, chunk_set_id=workspace.chunk_set_id,
                                input_validity="valid")
    plain_answer = AnswerRun("ans-plain", plain_selection.id, "complete", "alpha", "same-model", 1, 1,
                             prompt_version="prompt-v1")
    plain_comparison = build_comparison(plain, workspace, [plain_selection], [plain_answer])
    assert plain_comparison.records[0]["answer_f1"]["status"] == "not_run"


def test_follow_up_is_rewritten_and_benchmark_question_is_not():
    config = replace(settings, deepseek_api_key="deep")
    workspace = services.create_workspace([("a.pdf", pdf_bytes("alpha beta"))], config)

    def fake_call(url, key, payload, timeout):
        return {"choices": [{"message": {"content": "What happened to Alpha later?"}}], "usage": {"prompt_tokens": 8, "completion_tokens": 4}}

    follow = services.create_question(workspace, "后来呢", [("Alpha 是什么", "一家公司")], call=fake_call, config=config)
    assert follow.rewrite_status == "rewritten"
    assert follow.text == "后来呢"
    benchmark = services.create_question(workspace, "What is Alpha?", benchmark=True, call=fake_call, config=config)
    assert benchmark.rewritten == "What is Alpha?"
    assert benchmark.rewrite_status == "original"


def test_html_form_sends_question_as_form_field(tmp_path, monkeypatch):
    import app.main as main
    local_db = Database(tmp_path / "form.sqlite3")
    monkeypatch.setattr(main, "db", local_db)
    monkeypatch.setattr(main, "settings", replace(settings, jev_api_key="", deepseek_api_key="",
                                                  standard_rag_configured=False))
    client = TestClient(app)
    response = client.post("/ask", data={"question": "What?", "mode": "jev"},
                           files={"files": ("a.pdf", pdf_bytes("alpha beta"), "application/pdf")})
    assert response.status_code == 200
    assert "结果" in response.text
    assert "Field required" not in response.text
