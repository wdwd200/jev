"""Connectivity smoke test; prints response metadata but never API keys."""
from __future__ import annotations

import json
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv


def post(base: str, endpoint: str, key: str, payload: dict[str, Any], timeout: float) -> Any:
    response = requests.post(f"{base.rstrip('/')}/{endpoint.lstrip('/')}",
                             headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def main() -> int:
    load_dotenv()
    required = ["JEV_API_KEY", "DEEPSEEK_API_KEY"]
    missing = [name for name in required if not os.getenv(name, "").strip()]
    if missing:
        print("Smoke test skipped; missing: " + ", ".join(missing))
        return 2
    timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "120"))
    try:
        jev = post(os.getenv("JEV_BASE_URL", "https://api.typesafe.ai"), os.getenv("JEV_ENDPOINT", "/v1/systemone"),
                   os.environ["JEV_API_KEY"], {"state": "A short test document.", "model": os.getenv("JEV_MODEL", "jev-latest"),
                   "questions": {"relevance": {"type": "noul", "instructions": "Is this relevant?"}}}, timeout)
        deepseek = post(os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"), os.getenv("DEEPSEEK_ENDPOINT", "/chat/completions"),
                        os.environ["DEEPSEEK_API_KEY"], {"model": os.getenv("DEEPSEEK_MODEL", "deepseek-flash"),
                        "messages": [{"role": "user", "content": "Answer with one word: What is the capital of France?"}],
                        "temperature": 0, "stream": False}, timeout)
        print(json.dumps({"jev": {"keys": list(jev) if isinstance(jev, dict) else type(jev).__name__},
                          "deepseek": {"keys": list(deepseek) if isinstance(deepseek, dict) else type(deepseek).__name__}}, ensure_ascii=False))
    except (requests.RequestException, ValueError) as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
