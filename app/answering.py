from __future__ import annotations

import time
from typing import Any, Callable

import requests

from .config import Settings
from .models import AnswerRun, Question, Selection, new_id
from .selection import _safe_error
from .tokens import count_tokens


def answer_with_deepseek(selection: Selection, question: Question, config: Settings,
                         call: Callable[[str, str, dict[str, Any], float], dict[str, Any]] | None = None) -> AnswerRun:
    answer = AnswerRun(new_id("ans"), selection.id, "processing", model_name=config.deepseek_model,
                       prompt_version="deepseek-context-v1")
    if selection.workspace_id != question.workspace_id or selection.question_id != question.id:
        answer.status, answer.error = "failed", "Selection does not belong to this question"
        answer.usage.update({"input_tokens_source": "unavailable", "output_tokens_source": "unavailable"})
        return answer
    if selection.input_validity != "valid":
        answer.status, answer.error = "failed", "Selection input is not confirmed valid"
        return answer
    if selection.status != "complete":
        answer.status, answer.error = "failed", selection.reason or "Selection is not ready for answering"
        return answer
    if not config.deepseek_api_key:
        answer.status, answer.error = "failed", "DEEPSEEK_API_KEY is not configured"
        return answer
    context = "\n\n".join(row["text"] for row in selection.selected_chunks)
    images_held = any(row.get("kind") == "image" and row.get("has_image") for row in selection.selected_chunks)
    payload = {"model": config.deepseek_model,
               "messages": [{"role": "user", "content":
                             "Answer briefly using only the context. Do not explain.\n"
                             f"Question: {question.rewritten or question.text}\nContext:\n{context}"}],
               "temperature": 0, "stream": False, "thinking": {"type": "disabled"}}
    started = time.perf_counter()
    if call is None:
        from . import services
        call = services.post_json
    try:
        response = call(f"{config.deepseek_base_url.rstrip('/')}/{config.deepseek_endpoint.lstrip('/')}",
                        config.deepseek_api_key, payload, config.timeout_seconds)
        if not isinstance(response, dict):
            raise TypeError("Provider response must be an object")
        usage = response.get("usage", {})
        in_value = usage.get("prompt_tokens", usage.get("input_tokens")) if isinstance(usage, dict) else None
        out_value = usage.get("completion_tokens", usage.get("output_tokens")) if isinstance(usage, dict) else None
        if in_value is not None:
            answer.input_tokens = int(in_value)
            answer.usage["input_tokens_source"] = "reported"
        else:
            answer.input_tokens = count_tokens(payload["messages"][0]["content"])
            answer.usage["input_tokens_source"] = "estimated"
        if out_value is not None:
            answer.output_tokens = int(out_value)
            answer.usage["output_tokens_source"] = "reported"
        else:
            # Parse the answer before estimating completion usage so malformed
            # provider structures remain a failed run with known usage kept.
            answer.output_tokens = 0
            answer.usage["output_tokens_source"] = "unavailable"
        try:
            content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        except (AttributeError, IndexError, TypeError) as exc:
            raise ValueError("Provider response has an invalid choices structure") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Provider returned an empty answer")
        answer.answer = content
        if out_value is None:
            answer.output_tokens = count_tokens(answer.answer)
            answer.usage["output_tokens_source"] = "estimated"
        answer.cost_status = "not_calculated"
        answer.usage["images_sent"] = False
        answer.usage["images_held"] = images_held
        answer.status = "complete"
    except (requests.RequestException, KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        answer.status, answer.error = "failed", _safe_error(exc, config)
        answer.usage.setdefault("input_tokens_source", "unavailable")
        answer.usage.setdefault("output_tokens_source", "unavailable")
        answer.cost_status = "not_calculated"
    answer.elapsed_ms = round((time.perf_counter() - started) * 1000)
    return answer
