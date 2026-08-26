"""
Step 3: Embed all chunks and load them into a persistent ChromaDB collection.

Reads output/chunks.jsonl (produced by stage2_chunk.py), embeds the text of
every chunk with a local sentence-transformers model, and upserts into a
persistent Chroma collection at output/chroma_db/. Safe to re-run: existing
collection is dropped and rebuilt fresh (embedding is fast enough on CPU
that incremental resumability isn't worth the complexity here).

Usage:
    python stage3_embed.py [--model BAAI/bge-small-en-v1.5] [--batch-size 128]
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

torch.set_num_threads(os.cpu_count())

import chromadb
from sentence_transformers import SentenceTransformer

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
CHUNKS_PATH = OUT_DIR / "chunks.jsonl"
CHROMA_DIR = OUT_DIR / "chroma_db"
COLLECTION_NAME = "cat_prep"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


def load_chunks() -> list:
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def sanitize_metadata(chunk: dict) -> dict:
    return {
        "source_file": chunk["source_file"],
        "doc_type": chunk["doc_type"],
        "chunk_type": chunk["chunk_type"],
        "heading": chunk["heading"] or "",
        "page_start": chunk["page_start"] if chunk["page_start"] is not None else -1,
        "page_end": chunk["page_end"] if chunk["page_end"] is not None else -1,
        "char_count": chunk["char_count"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="chunks per collection.add() call")
    parser.add_argument("--encode-batch-size", type=int, default=64,
                        help="chunks per forward pass through the embedding model")
    parser.add_argument("--limit", type=int, default=None, help="only embed the first N chunks (for testing)")
    parser.add_argument("--resume", action="store_true",
                        help="keep the existing collection and only embed chunk_ids not already in it")
    args = parser.parse_args()

    print(f"Loading chunks from {CHUNKS_PATH} ...")
    chunks = load_chunks()
    if args.limit:
        chunks = chunks[: args.limit]
    print(f"Loaded {len(chunks)} chunks")

    print(f"Opening persistent Chroma store at {CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if args.resume:
        collection = client.get_or_create_collection(
            COLLECTION_NAME,
            metadata={"embedding_model": args.model, "hnsw:space": "cosine"},
        )
        existing = set()
        offset = 0
        page = 20000
        while True:
            got = collection.get(limit=page, offset=offset, include=[])
            ids = got.get("ids", [])
            if not ids:
                break
            existing.update(ids)
            offset += len(ids)
        before = len(chunks)
        chunks = [c for c in chunks if c["chunk_id"] not in existing]
        print(f"Resume: {len(existing)} already embedded, {len(chunks)} of {before} remaining")
        if not chunks:
            print(f"\nNothing to do. {collection.count()} vectors in collection '{COLLECTION_NAME}'.")
            return
    else:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(
            COLLECTION_NAME,
            metadata={"embedding_model": args.model, "hnsw:space": "cosine"},
        )

    print(f"Loading embedding model: {args.model} ...")
    model = SentenceTransformer(args.model)

    # Sort globally by text length before batching. Transformer cost scales with
    # the LONGEST sequence in a batch (everything else is padded up to it), and
    # chunk lengths here range from ~10 to 1600 chars. Batching in corpus order
    # makes almost every batch pad to near the 1600-char maximum; length-sorting
    # makes each batch homogeneous and removes most of that wasted compute.
    # Insertion order is irrelevant to Chroma, so this is free.
    chunks.sort(key=lambda c: len(c["text"]))

    t_start = time.time()
    n = len(chunks)
    for i in range(0, n, args.batch_size):
        batch = chunks[i:i + args.batch_size]
        texts = [c["text"] for c in batch]
        # Passages are embedded RAW. Per the BGE model card, the retrieval
        # instruction goes on the QUERY side only (see retrieve.py) -- "in all
        # cases, no instruction needs to be added to passages". The E5-style
        # "passage: " prefix would misalign the query/passage vector spaces.
        embeddings = model.encode(texts, batch_size=args.encode_batch_size,
                                  show_progress_bar=False, normalize_embeddings=True)

        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=[sanitize_metadata(c) for c in batch],
        )

        if (i // args.batch_size) % 5 == 0 or i + args.batch_size >= n:
            done = min(i + args.batch_size, n)
            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            # flush=True so progress is visible when stdout is redirected to a
            # log file (block buffering otherwise hides it until process exit).
            print(f"  [{done}/{n}] elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m rate={rate:.0f} chunks/s",
                  flush=True)

    print(f"\nDone. {collection.count()} vectors in collection '{COLLECTION_NAME}'.")
    print(f"Chroma store: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
