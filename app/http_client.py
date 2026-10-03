from __future__ import annotations

from typing import Any

import requests


def post_json(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                             json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()
