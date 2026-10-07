from __future__ import annotations

import re

from typing import Any

from .models import Block, Document, new_id
from .tokens import count_tokens


def blocks_from_text(text: str) -> list[Block]:
    """Turn benchmark text into paragraph blocks with stable positions.

    Blank lines separate paragraphs. A file with no blank lines, such as a
    copied financial statement, uses one block per line. Page is 1 because
    these sources do not carry PDF page numbers.
    """
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) <= 1:
        paragraphs = [part.strip() for part in text.splitlines() if part.strip()]
    return [Block("text", part, 1, index) for index, part in enumerate(paragraphs, start=1)]


def page_count(content: bytes) -> int:
    import pymupdf
    document = pymupdf.open(stream=content, filetype="pdf")
    try:
        return document.page_count
    finally:
        document.close()


def extract_pdf(filename: str, content: bytes, workspace_id: str = "", max_pages: int | None = None) -> Document:
    document_id = new_id("doc")
    try:
        if max_pages is not None and page_count(content) > max_pages:
            return Document(document_id, workspace_id, filename, "", 0, False,
                            f"PDF exceeds {max_pages} pages", content, [])
        blocks = _parse(content)
        text = "\n\n".join(block.text for block in blocks if block.text).strip()
        if not text and not any(block.kind == "image" and block.image for block in blocks):
            return Document(document_id, workspace_id, filename, "", 0, False, "PDF contains no extractable text", content, [])
        return Document(document_id, workspace_id, filename, text, count_tokens(text), True, None, content, blocks)
    except Exception as exc:
        return Document(document_id, workspace_id, filename, "", 0, False, f"PDF parsing failed: {exc}", content, [])


def _parse(content: bytes) -> list[Block]:
    docling_blocks = _parse_docling(content)
    if docling_blocks is not None:
        return docling_blocks
    return _parse_pdf_blocks(content)


def _parse_docling(content: bytes) -> list[Block] | None:
    """Use Docling when it is installed. The local shape below matches its blocks."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        return None
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
        handle.write(content)
        path = handle.name
    try:
        result = DocumentConverter().convert(path)
        document = result.document
    except Exception:
        return None
    finally:
        import os
        os.unlink(path)
    blocks: list[Block] = []
    for index, table in enumerate(getattr(document, "tables", []) or [], start=1):
        markdown = ""
        exporter = getattr(table, "export_to_markdown", None)
        if exporter:
            markdown = exporter() or ""
        blocks.append(Block("table", markdown.strip(), _page_of(table), index))
    for index, picture in enumerate(getattr(document, "pictures", []) or [], start=1):
        caption = " ".join(getattr(item, "text", "") for item in getattr(picture, "captions", []) or [])
        blocks.append(Block("image", caption.strip(), _page_of(picture), index))
    markdown = document.export_to_markdown() if hasattr(document, "export_to_markdown") else ""
    if markdown.strip():
        blocks.insert(0, Block("text", markdown.strip(), 1, 1))
    return blocks or None


def recognize_page(content: bytes, page_number: int) -> list[dict[str, Any]]:
    """Recognize one PDF page. A page with no text layer still returns its images."""
    import pymupdf

    document = pymupdf.open(stream=content, filetype="pdf")
    try:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError("Page is outside the PDF")
        repeated = _repeated_margins([_page_lines(page) for page in document])
        page = document[page_number - 1]
        lines = [line for line in _page_lines(page) if line.strip() not in repeated]
        blocks: list[dict[str, Any]] = []
        text = _unwrap(lines)
        if text:
            blocks.append({"kind": "text", "text": text, "position": 1, "headings": []})
        finder = page.find_tables()
        for index, table in enumerate(finder.tables, start=1):
            rendered = _markdown_table(table.extract())
            if rendered:
                blocks.append({"kind": "table", "text": rendered, "position": index, "headings": []})
        for index, image in enumerate(page.get_images(full=True), start=1):
            payload = document.extract_image(image[0])
            raw = payload.get("image", b"")
            blocks.append({"kind": "image", "text": "", "position": index, "headings": [], "image": raw})
        if not blocks:
            blocks.append({"kind": "image", "text": "", "position": 1, "headings": [], "image": b""})
        return blocks
    finally:
        document.close()


def _page_of(item: Any) -> int:
    provenance = getattr(item, "prov", None) or []
    if provenance:
        return int(getattr(provenance[0], "page_no", 1) or 1)
    return 1


def _parse_pdf_blocks(content: bytes) -> list[Block]:
    import pymupdf

    document = pymupdf.open(stream=content, filetype="pdf")
    try:
        page_lines = [_page_lines(page) for page in document]
        repeated = _repeated_margins(page_lines)
        blocks: list[Block] = []
        for number, page in enumerate(document, start=1):
            lines = [line for line in page_lines[number - 1] if line.strip() not in repeated]
            text = _unwrap(lines)
            if text:
                blocks.append(Block("text", text, number, 1))
            finder = page.find_tables()
            for index, table in enumerate(finder.tables, start=1):
                rendered = _markdown_table(table.extract())
                if rendered:
                    blocks.append(Block("table", rendered, number, index))
            for index, image in enumerate(page.get_images(full=True), start=1):
                payload = document.extract_image(image[0])
                blocks.append(Block("image", "", number, index, image=payload.get("image", b"")))
        return blocks
    finally:
        document.close()


def _page_lines(page: Any) -> list[str]:
    lines: list[str] = []
    for raw in page.get_text("text").splitlines():
        lines.append(raw.strip())
    return lines


def _repeated_margins(page_lines: list[list[str]]) -> set[str]:
    if len(page_lines) < 2:
        return set()
    from collections import Counter

    tops: Counter[str] = Counter()
    bottoms: Counter[str] = Counter()
    for lines in page_lines:
        kept = [line for line in lines if line]
        if not kept:
            continue
        tops[kept[0]] += 1
        bottoms[kept[-1]] += 1
    threshold = max(2, len(page_lines) // 2)
    return {line for line, count in (tops + bottoms).items() if count >= threshold}


def _unwrap(lines: list[str]) -> str:
    paragraphs: list[str] = []
    buffer: list[str] = []
    for line in lines:
        if not line:
            if buffer:
                paragraphs.append(_join_wrapped(buffer))
                buffer = []
            continue
        buffer.append(line)
    if buffer:
        paragraphs.append(_join_wrapped(buffer))
    return "\n\n".join(paragraphs).strip()


def _join_wrapped(lines: list[str]) -> str:
    text = lines[0]
    for nxt in lines[1:]:
        if text.endswith("-") and nxt[:1].islower():
            text = text[:-1] + nxt
        elif text[-1:] in ".!?。！？":
            text = f"{text} {nxt}"
        else:
            text = f"{text} {nxt}"
    return text


def _markdown_table(rows: list[list[Any]]) -> str:
    rendered = []
    for row in rows:
        cells = [" ".join(str(cell or "").split()) for cell in row]
        if any(cells):
            rendered.append(cells)
    if not rendered:
        return ""
    width = max(len(row) for row in rendered)

    def form(row: list[str]) -> str:
        padded = row + [""] * (width - len(row))
        return "| " + " | ".join(padded) + " |"

    lines = [form(rendered[0]), "| " + " | ".join("---" for _ in range(width)) + " |"]
    lines.extend(form(row) for row in rendered[1:])
    return "\n".join(lines)
