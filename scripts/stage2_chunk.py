"""
Step 2 (stage 2): Chunk all cached extractions.

Reads every cached extraction JSON from output/extracted/ (produced by
stage1_extract.py) and runs chunk_document() on it. Fast (no OCR/PDF
parsing), so this rebuilds output/chunks.jsonl from scratch every run --
safe to re-run freely while iterating on chunking logic.

Usage:
    python stage2_chunk.py [--source PATH]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk import chunk_document

DEFAULT_SOURCE = r"C:\Users\ARNAV\Desktop\extrass\CAT"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"
EXTRACTED_DIR = OUT_DIR / "extracted"
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"


def canonical_rank(source_file: str) -> tuple:
    """Deterministic preference for which copy of duplicated content to keep:
    shallowest path first, then shortest, then lexicographic. Keeps the
    top-level `Prep Club/QA/foo.pdf` over a nested archive re-extraction."""
    return (source_file.count("/"), len(source_file), source_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--no-dedupe", action="store_true",
                        help="keep byte-identical chunks from duplicate files (default is to drop them)")
    args = parser.parse_args()

    source = Path(args.source)
    cache_files = sorted(EXTRACTED_DIR.glob("*.json"))
    print(f"Cached extractions: {len(cache_files)}")

    n_docs = 0
    n_errors = 0
    all_chunks = []

    for cf in cache_files:
        try:
            extraction = json.loads(cf.read_text(encoding="utf-8"))
            all_chunks.extend(chunk_document(extraction, source))
            n_docs += 1
        except Exception as e:
            n_errors += 1
            print(f"  ERROR chunking {cf.name}: {type(e).__name__}: {e}")

    n_raw = len(all_chunks)
    n_dropped = 0

    if not args.no_dedupe:
        # The corpus ships the same books at several paths (archive
        # re-extractions, PYQ folders mirroring subject folders). Identical
        # chunk text would otherwise occupy multiple top-k slots for one
        # underlying passage, so keep a single canonical copy of each.
        best = {}
        for c in all_chunks:
            key = hashlib.sha1(c["text"].encode("utf-8")).digest()
            prev = best.get(key)
            if prev is None or canonical_rank(c["source_file"]) < canonical_rank(prev["source_file"]):
                best[key] = c
        deduped = list(best.values())
        n_dropped = n_raw - len(deduped)
        # Preserve document order rather than dict-insertion order.
        deduped.sort(key=lambda c: (c["source_file"], c["page_start"] or 0, c["chunk_id"]))
        all_chunks = deduped

    max_chunk_chars = 0
    with open(CHUNKS_PATH, "w", encoding="utf-8") as out_f:
        for c in all_chunks:
            out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
            max_chunk_chars = max(max_chunk_chars, c["char_count"])

    print(f"\nDone. docs={n_docs} errors={n_errors}")
    print(f"  chunks before dedupe: {n_raw}")
    print(f"  duplicate chunks dropped: {n_dropped}")
    print(f"  total_chunks: {len(all_chunks)}  max_chunk_chars={max_chunk_chars}")
    print(f"Chunks: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
