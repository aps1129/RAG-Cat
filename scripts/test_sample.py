"""Sanity-test extract.py + chunk.py on a handful of representative files."""
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import extract_document
from chunk import chunk_document

SOURCE = Path(r"C:\Users\ARNAV\Desktop\extrass\CAT")

SAMPLES = [
    ("clean formula sheet", SOURCE / "Prep Club/Cracku Formula/CAT Number System Formulas PDF.pdf"),
    ("clean DI w/ tables", SOURCE / "Prep Club/TIME Material/TIME-2021/Data Interpretation Basic and Exercise with solution/DI chapter 1 - Tables .pdf"),
    ("scanned book (5pg)", SOURCE / "Prep Club/QA/QUANTUM CAT By Sarvesh Verma.pdf"),
    ("garbled font", SOURCE / "Prep Club/CL Practice Sheets/QA/QA-20 Algebra 4 with Solutions.pdf"),
    ("html", SOURCE / "Prep Club/IMS Materials/IMS Mocks_/IMS CAT 2020 Mocks Sectionals Topic Questions/Practice Material/Quant/Algebra/Basic.htm"),
    ("docx", SOURCE / "Prep Club/LOKESHVA/Parajumble_/PJ Practice 1.docx"),
]

for label, path in SAMPLES:
    print(f"\n{'=' * 70}\n{label}: {path.relative_to(SOURCE)}\n{'=' * 70}")
    if not path.exists():
        print("  FILE NOT FOUND")
        continue
    t0 = time.time()
    max_pages = 5 if "scanned" in label else None
    extraction = extract_document(path, max_pages=max_pages)
    t1 = time.time()
    print(f"  extracted {len(extraction['pages'])} pages in {t1 - t0:.1f}s, errors={extraction['errors']}")
    methods = {p["method"] for p in extraction["pages"]}
    print(f"  methods used: {methods}")

    chunks = chunk_document(extraction, SOURCE)
    t2 = time.time()
    print(f"  chunked into {len(chunks)} chunks in {t2 - t1:.1f}s")

    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    if text_chunks:
        avg_len = sum(c["char_count"] for c in text_chunks) / len(text_chunks)
        print(f"  text chunks: {len(text_chunks)} (avg {avg_len:.0f} chars), table chunks: {len(table_chunks)}")

    for c in chunks[:4]:
        preview = c["text"][:150].replace("\n", " ")
        print(f"    [{c['chunk_type']}] heading={c['heading']!r} pages={c['page_start']}-{c['page_end']} chars={c['char_count']}")
        print(f"        {preview}...")
