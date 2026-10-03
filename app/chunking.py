from __future__ import annotations

from typing import Any

from .models import Block, Chunk, Document, new_id
from .tokens import count_tokens, split_text


def chunk_document(document: Document, config: Any, chunk_set_id: str) -> list[Chunk]:
    if not document.extraction_ok:
        return []
    blocks = document.blocks or [Block("text", document.text, 1, 1)]
    result: list[Chunk] = []
    for block in blocks:
        result.extend(_chunk_block(document, block, config, chunk_set_id, len(result)))
    return result


def chunk_documents(documents: list[Document], config: Any, chunk_set_id: str | None = None) -> list[Chunk]:
    chunk_set_id = chunk_set_id or new_id("chunks")
    result: list[Chunk] = []
    for document in documents:
        result.extend(chunk_document(document, config, chunk_set_id))
    return result


def _chunk_block(document: Document, block: Block, config: Any, chunk_set_id: str, start: int) -> list[Chunk]:
    size = max(1, config.chunk_size)
    if block.kind == "image":
        pieces = split_text(block.text, size) or [""]
    elif block.kind == "table":
        pieces = _table_pieces(block.text, size)
    else:
        pieces = _text_pieces(block.text, block.headings, size)
    chunks: list[Chunk] = []
    for offset, text in enumerate(pieces):
        source = {"filename": document.filename, "page_start": block.page, "page_end": block.page,
                  "position": block.index, "headings": block.headings, "kind": block.kind}
        chunks.append(Chunk(new_id("chunk"), chunk_set_id, document.id, start + offset, text,
                            count_tokens(text), config.chunk_version, block.kind, source,
                            block.image if block.kind == "image" else b""))
    return chunks


def _text_pieces(text: str, headings: list[str], size: int) -> list[str]:
    prefix = " > ".join(headings).strip()
    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not paragraphs:
        return []
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph if not current else f"{current}\n\n{paragraph}"
        labeled = f"{prefix}\n\n{candidate}" if prefix else candidate
        if current and count_tokens(labeled) > size:
            pieces.append(f"{prefix}\n\n{current}" if prefix else current)
            current = paragraph
        else:
            current = candidate
    if current:
        labeled = f"{prefix}\n\n{current}" if prefix else current
        if count_tokens(labeled) <= size:
            pieces.append(labeled)
        else:
            pieces.extend(split_text(labeled, size))
    return pieces


def _table_pieces(markdown: str, size: int) -> list[str]:
    lines = [line for line in markdown.splitlines() if line.strip()]
    if not lines:
        return []
    header = lines[:2] if len(lines) > 1 and set(lines[1].replace("|", "").replace("-", "").replace(":", "").strip()) == set() else lines[:1]
    body = lines[len(header):]
    pieces: list[str] = []
    current: list[str] = []
    for row in body or []:
        candidate = header + current + [row]
        if current and count_tokens("\n".join(candidate)) > size:
            pieces.append("\n".join(header + current))
            current = [row]
        else:
            current.append(row)
    if current or not body:
        pieces.append("\n".join(header + current))
    return [piece for piece in pieces if piece.strip()]
