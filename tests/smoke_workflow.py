"""Explicit live PDF/API workflow check using synthetic text and a temporary DB.

Run: python -X utf8 tests/smoke_workflow.py
Makes at most two Jev and three generation calls (one chunk per document).
Output contains statuses only, never keys, source documents or full responses.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    report = {"kind": "live_synthetic_pdf_api_workflow", "quality_evaluation": False,
              "browser_visual_check": False}
    with tempfile.TemporaryDirectory(prefix="workflow-smoke-", dir=runtime) as folder:
        client = None
        previous_path = os.environ.get("DATABASE_PATH")
        os.environ["DATABASE_PATH"] = str(Path(folder) / "workflow.sqlite3")
        try:
            from app import main as application
            from app.config import settings
            from app.database import Database
            from fastapi.testclient import TestClient
            from test_local import pdf_bytes

            if not settings.jev_api_key or not settings.deepseek_api_key:
                report["status"] = "skipped_missing_configuration"
            else:
                application.settings = replace(settings, timeout_seconds=20)
                client = TestClient(application.app)
                pdf = pdf_bytes("The project codename is Aurora. The access code is cobalt.")
                upload = client.post("/workspaces", files={"files": ("synthetic.pdf", pdf, "application/pdf")})
                upload.raise_for_status()
                workspace = upload.json()
                report["upload_status"] = workspace["status"]
                report["chunk_count"] = workspace["chunk_count"]
                if workspace["status"] != "ready" or workspace["chunk_count"] != 1:
                    report["status"] = "not_run_requires_one_ready_chunk"
                else:
                    original_chunk_set = workspace["chunk_set_id"]
                    q1 = client.post(f"/workspaces/{workspace['id']}/questions",
                                     json={"question": "What is the project codename?", "mode": "all"})
                    q1.raise_for_status()
                    q2 = client.post(f"/workspaces/{workspace['id']}/questions",
                                     json={"question": "What is the access code?", "mode": "jev"})
                    q2.raise_for_status()
                    runs = q1.json()["runs"] + q2.json()["runs"]
                    report["runs"] = [{"strategy": run["strategy"], "selection_status": run["status"],
                                       "answer_status": (run["answer"] or {}).get("status"),
                                       "answer_nonempty": bool((run["answer"] or {}).get("text", "").strip())}
                                      for run in runs]
                    application.db = Database(os.environ["DATABASE_PATH"])
                    reloaded = client.get(f"/workspaces/{workspace['id']}").json()
                    report["question_count_after_reopen"] = len(reloaded["questions"])
                    report["chunk_set_reused"] = reloaded["chunk_set_id"] == original_chunk_set
                    historical = client.get(f"/questions/{q1.json()['question']['id']}/runs").json()
                    report["history_preserved"] = historical["runs"] == q1.json()["runs"]
                    report["status"] = "passed" if (
                        report["question_count_after_reopen"] == 2 and report["chunk_set_reused"]
                        and report["history_preserved"]
                        and all(run["selection_status"] == "complete" and run["answer_status"] == "complete"
                                and run["answer_nonempty"] for run in report["runs"] if run["strategy"] != "rag")
                        and all(run["selection_status"] == "failed" and run["answer_status"] is None
                                for run in report["runs"] if run["strategy"] == "rag")
                    ) else "failed"
        except Exception as exc:
            report["status"] = "failed"
            report["exception_type"] = type(exc).__name__
        finally:
            if client is not None:
                client.close()
            # Release any library references before removing the isolated folder.
            gc.collect()
            if previous_path is None:
                os.environ.pop("DATABASE_PATH", None)
            else:
                os.environ["DATABASE_PATH"] = previous_path
        text = json.dumps(report, ensure_ascii=False, indent=2)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        (runtime / f"live-workflow-{stamp}.json").write_text(text, encoding="utf-8")
    print(text)
    return 0 if report["status"] == "passed" else 2 if report["status"].startswith("skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
