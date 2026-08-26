"""
Step 2b: Semantic / section-based chunking.

Takes the normalized output of extract.extract_document() and produces a
flat list of chunk dicts. Strategy:

  1. Build a document-wide sequence of "lines" (from PDF text-spans, or
     plain text lines for HTML/OCR'd/docx pages where span info isn't
     available).
  2. Detect the body font size (the mode) and flag lines that are visually
     larger/bolder and short as section headings.
  3. Group lines into sections bounded by headings (a doc with too few
     detected headings falls back to one big section = paragraph-only
     chunking).
  4. Within each section, split on paragraph boundaries and pack into
     chunks up to MAX_CHUNK_CHARS, never fixed-token windows.
  5. Tables detected by PyMuPDF's find_tables() become their own chunks
     (chunk_type="table"), tagged with the nearest preceding heading.
"""

import re
import statistics
from pathlib import Path

from extract import _looks_like_real_table

MAX_CHUNK_CHARS = 1600
MIN_HEADINGS_FOR_SECTION_MODE = 2


def _build_lines(extraction: dict) -> list:
    lines = []
    for page in extraction["pages"]:
        if page["spans"]:
            for span in page["spans"]:
                text = span["text"].strip()
                if text:
                    lines.append({"text": text, "size": span["size"], "bold": span["bold"], "page": page["page_num"]})
        else:
            for raw_line in page["text"].splitlines():
                text = raw_line.strip()
                if text:
                    lines.append({"text": text, "size": None, "bold": False, "page": page["page_num"]})
    return lines


def _body_size(lines: list):
    sizes = [round(l["size"]) for l in lines if l["size"]]
    if not sizes:
        return None
    return statistics.mode(sizes)


_PAGE_NUM_RE = re.compile(r"^[\d\.\s\-–—]+$")
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{3,}")


def _is_heading(line: dict, body_size) -> bool:
    if body_size is None or line["size"] is None:
        return False
    text = line["text"]
    if not text or len(text) > 120:
        return False
    if _PAGE_NUM_RE.match(text):
        return False
    # Math/equation spans often render with oversized fonts (symbols,
    # fractions, superscripts) and would otherwise be misread as headings.
    # Require at least one real word so we only promote actual prose headers.
    if not _ALPHA_WORD_RE.search(text):
        return False
    if line["size"] >= body_size * 1.15:
        return True
    if line["bold"] and line["size"] >= body_size * 1.05:
        return True
    return False


def _group_sections(lines: list, body_size) -> list:
    sections = []
    current = {"heading": None, "page_start": None, "page_end": None, "text_lines": []}

    def flush():
        if current["text_lines"] or current["heading"]:
            sections.append(dict(current))

    for line in lines:
        if _is_heading(line, body_size):
            flush()
            current = {"heading": line["text"], "page_start": line["page"], "page_end": line["page"], "text_lines": []}
        else:
            if current["page_start"] is None:
                current["page_start"] = line["page"]
            current["page_end"] = line["page"]
            current["text_lines"].append(line["text"])
    flush()
    return sections


def _hard_split(text: str, max_chars: int) -> list:
    """Last-resort word-boundary split. Guarantees no piece exceeds max_chars,
    even if `text` has no punctuation/whitespace split points at all (e.g. a
    PyMuPDF line-join artifact that swallowed a space after a period)."""
    words = text.split(" ")
    pieces = []
    buf = ""
    for w in words:
        if len(buf) + len(w) + 1 <= max_chars:
            buf = (buf + " " + w).strip()
        else:
            if buf:
                pieces.append(buf)
                buf = ""
            if len(w) > max_chars:
                for i in range(0, len(w), max_chars):
                    pieces.append(w[i:i + max_chars])
            else:
                buf = w
    if buf:
        pieces.append(buf)
    return pieces


def _pack_sentences(text: str, max_chars: int) -> list:
    sentences = re.split(r"(?<=[.?!])\s+", text)
    chunks = []
    sbuf = ""
    for s in sentences:
        if len(sbuf) + len(s) + 1 <= max_chars:
            sbuf = (sbuf + " " + s).strip()
        else:
            if sbuf:
                chunks.append(sbuf)
                sbuf = ""
            if len(s) > max_chars:
                chunks.extend(_hard_split(s, max_chars))
            else:
                sbuf = s
    if sbuf:
        chunks.append(sbuf)
    return chunks


def _split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paras:
        paras = [text.strip()] if text.strip() else []

    chunks = []
    buf = ""
    for p in paras:
        if len(p) > max_chars:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_pack_sentences(p, max_chars))
            continue

        if len(buf) + len(p) + 1 <= max_chars:
            buf = (buf + "\n" + p).strip()
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return chunks


def _row_to_text(row: list) -> str:
    cells = [str(c).strip() if c is not None else "" for c in row]
    return " | ".join(cells)


def _table_to_chunks(table_rows: list, max_chars: int = MAX_CHUNK_CHARS) -> list:
    """Render a table to text, splitting into multiple pieces by row if it's
    too large for one chunk. The header row (first row) is repeated on every
    piece so each stays independently interpretable."""
    if not table_rows:
        return []
    header = _row_to_text(table_rows[0])
    body_rows = [_row_to_text(r) for r in table_rows[1:]]

    if not body_rows:
        return [header]

    full = "\n".join([header] + body_rows)
    if len(full) <= max_chars:
        return [full]

    pieces = []
    buf_rows = [header]
    buf_len = len(header)
    for row_text in body_rows:
        if buf_len + len(row_text) + 1 > max_chars and len(buf_rows) > 1:
            pieces.append("\n".join(buf_rows))
            buf_rows = [header]
            buf_len = len(header)
        buf_rows.append(row_text)
        buf_len += len(row_text) + 1
    if len(buf_rows) > 1:
        pieces.append("\n".join(buf_rows))
    return pieces


def chunk_document(extraction: dict, source_root: Path) -> list:
    path = Path(extraction["path"])
    try:
        rel = str(path.relative_to(source_root))
    except ValueError:
        rel = str(path)

    lines = _build_lines(extraction)
    body_size = _body_size(lines)
    sections = _group_sections(lines, body_size)

    n_headings = sum(1 for s in sections if s["heading"])
    section_mode = n_headings >= MIN_HEADINGS_FOR_SECTION_MODE

    if not section_mode:
        all_text = "\n\n".join(l["text"] for l in lines)
        page_start = lines[0]["page"] if lines else 0
        page_end = lines[-1]["page"] if lines else 0
        sections = [{"heading": None, "page_start": page_start, "page_end": page_end, "text_lines": [all_text]}]

    chunks = []
    doc_id = rel.replace("\\", "/")

    for sec in sections:
        full_text = "\n".join(sec["text_lines"]) if section_mode else "\n\n".join(sec["text_lines"])
        for part_idx, piece in enumerate(_split_long_text(full_text)):
            if len(piece.strip()) < 20:
                continue
            chunks.append({
                "chunk_id": f"{doc_id}::text::{len(chunks)}",
                "source_file": doc_id,
                "doc_type": extraction["doc_type"],
                "chunk_type": "text",
                "heading": sec["heading"],
                "page_start": sec["page_start"],
                "page_end": sec["page_end"],
                "text": piece,
                "char_count": len(piece),
            })

    running_heading = None
    heading_by_page = {}
    for sec in sections:
        if sec["heading"]:
            running_heading = sec["heading"]
        if sec["page_start"] is not None:
            for p in range(sec["page_start"], (sec["page_end"] or sec["page_start"]) + 1):
                heading_by_page.setdefault(p, running_heading)

    for page in extraction["pages"]:
        for t_idx, table in enumerate(page.get("tables", [])):
            if not _looks_like_real_table(table):
                continue  # cached extraction predates the false-positive-table filter
            for part_idx, table_text in enumerate(_table_to_chunks(table)):
                if len(table_text.strip()) < 10:
                    continue
                chunks.append({
                    "chunk_id": f"{doc_id}::table::{page['page_num']}::{t_idx}::{part_idx}",
                    "source_file": doc_id,
                    "doc_type": extraction["doc_type"],
                    "chunk_type": "table",
                    "heading": heading_by_page.get(page["page_num"]),
                    "page_start": page["page_num"],
                    "page_end": page["page_num"],
                    "text": table_text,
                    "char_count": len(table_text),
                })

    return chunks
