from __future__ import annotations

import os
from pathlib import Path

METHOD = "whitespace-cjk-v1"
_tokenizer = None


def _load_tokenizer() -> None:
    global METHOD, _tokenizer
    configured = os.getenv("DEEPSEEK_TOKENIZER_JSON", "")
    if not configured:
        return
    path = Path(configured)
    if not path.is_file():
        return
    from tokenizers import Tokenizer

    _tokenizer = Tokenizer.from_file(str(path))
    METHOD = "deepseek-tokenizer-v1"


_load_tokenizer()


def count_tokens(text: str) -> int:
    """Count text the way selection and the workspace limit must agree.

    A configured DeepSeek tokenizer file is authoritative. Without that file,
    each CJK character is one token and each whitespace-separated word is one
    token. This is not billed usage; answer records keep the provider count.
    """
    if not text:
        return 0
    if _tokenizer is not None:
        return len(_tokenizer.encode(text).ids)
    return len(pieces(text))


def pieces(text: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            parts.append("".join(buf))
            buf.clear()

    for char in text:
        if char.isspace():
            flush()
        elif "\u4e00" <= char <= "\u9fff":
            flush()
            parts.append(char)
        else:
            buf.append(char)
    flush()
    return parts


def join_pieces(parts: list[str]) -> str:
    if not parts:
        return ""
    out = [parts[0]]
    for previous, current in zip(parts, parts[1:]):
        if not _cjk(previous) and not _cjk(current):
            out.append(" ")
        out.append(current)
    return "".join(out)


def split_text(text: str, size: int) -> list[str]:
    parts = pieces(text)
    if not parts:
        return []
    step = max(1, size)
    return [join_pieces(parts[start : start + step]) for start in range(0, len(parts), step)]


def _cjk(text: str) -> bool:
    return bool(text) and all("\u4e00" <= char <= "\u9fff" for char in text)
