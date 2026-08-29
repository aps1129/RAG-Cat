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
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chunk import chunk_document
from link_qa import load_links as load_qa_links

DEFAULT_SOURCE = r"C:\Users\ARNAV\Desktop\extrass\CAT"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"
EXTRACTED_DIR = OUT_DIR / "extracted"
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"


def canonical_rank(source_file: str) -> tuple:
    """Deterministic preference for which copy of duplicated content to keep:
    shallowest path first, then shortest, then lexicographic. Keeps the
    top-level `Prep Club/QA/foo.pdf` over a nested archive re-extraction."""
    return (source_file.count("/"), len(source_file), source_file)


def dedup_chunks_and_alias(all_chunks: list) -> tuple:
    """Dedup drops an entire duplicate FILE's worth of chunks in favor of an
    identical copy at another path (whichever canonical_rank prefers). Any
    file-path-keyed metadata computed on the pre-dedup tree -- like
    qa_links.json -- silently breaks for a file that got dropped, since it's
    now attributed to its surviving twin instead. Recover this by voting:
    for each original source_file, find which surviving file most of its
    chunks now belong to, and treat that as an alias.

    Returns (deduped_chunks, alias_map)."""
    best = {}
    for c in all_chunks:
        key = hashlib.sha1(c["text"].encode("utf-8")).digest()
        prev = best.get(key)
        if prev is None or canonical_rank(c["source_file"]) < canonical_rank(prev["source_file"]):
            best[key] = c

    votes = defaultdict(Counter)
    for c in all_chunks:
        key = hashlib.sha1(c["text"].encode("utf-8")).digest()
        winner_file = best[key]["source_file"]
        votes[c["source_file"]][winner_file] += 1

    alias = {}
    for original_file, counter in votes.items():
        winner_file, _ = counter.most_common(1)[0]
        if winner_file != original_file:
            alias[original_file] = winner_file

    return list(best.values()), alias


def remap_qa_links(qa_links: dict, alias: dict) -> dict:
    def canon(path):
        return alias.get(path, path)

    remapped = {}
    for path, info in qa_links.items():
        new_key, new_linked = canon(path), canon(info["linked_to"])
        if new_key == new_linked:
            continue  # both sides deduped onto the same surviving file -- no real link left
        remapped[new_key] = {"role": info["role"], "linked_to": new_linked}
    return remapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--no-dedupe", action="store_true",
                        help="keep byte-identical chunks from duplicate files (default is to drop them)")
    args = parser.parse_args()

    source = Path(args.source)
    cache_files = sorted(EXTRACTED_DIR.glob("*.json"))
    print(f"Cached extractions: {len(cache_files)}")

    qa_links = load_qa_links()
    print(f"Q/A links loaded: {len(qa_links)} files ({len(qa_links) // 2} pairs)")

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
    n_aliased_links = 0

    if not args.no_dedupe:
        # The corpus ships the same books at several paths (archive
        # re-extractions, PYQ folders mirroring subject folders). Identical
        # chunk text would otherwise occupy multiple top-k slots for one
        # underlying passage, so keep a single canonical copy of each.
        deduped, alias = dedup_chunks_and_alias(all_chunks)
        n_dropped = n_raw - len(deduped)
        deduped.sort(key=lambda c: (c["source_file"], c["page_start"] or 0, c["chunk_id"]))
        all_chunks = deduped

        # qa_links.json was built against the pre-dedup file tree. Remap
        # through the alias BEFORE annotating, or any link whose file got
        # dropped in favor of a duplicate elsewhere silently disappears.
        remapped = remap_qa_links(qa_links, alias)
        n_aliased_links = sum(1 for p in qa_links if p in alias)
        qa_links = remapped

    # Annotate AFTER dedup: source_file on a surviving chunk is now guaranteed
    # to be the canonical path, matching qa_links's (now-remapped) keys.
    for c in all_chunks:
        link = qa_links.get(c["source_file"])
        c["qa_role"] = link["role"] if link else None
        c["linked_source"] = link["linked_to"] if link else None

    max_chunk_chars = 0
    with open(CHUNKS_PATH, "w", encoding="utf-8") as out_f:
        for c in all_chunks:
            out_f.write(json.dumps(c, ensure_ascii=False) + "\n")
            max_chunk_chars = max(max_chunk_chars, c["char_count"])

    n_linked_chunks = sum(1 for c in all_chunks if c.get("linked_source"))
    print(f"\nDone. docs={n_docs} errors={n_errors}")
    print(f"  chunks before dedupe: {n_raw}")
    print(f"  duplicate chunks dropped: {n_dropped}")
    print(f"  total_chunks: {len(all_chunks)}  max_chunk_chars={max_chunk_chars}")
    print(f"  qa_links remapped through dedup alias: {n_aliased_links}")
    print(f"  chunks with a linked question/answer file: {n_linked_chunks} "
          f"({len({c['source_file'] for c in all_chunks if c.get('linked_source')})} files)")
    print(f"Chunks: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
