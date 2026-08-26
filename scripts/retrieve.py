"""
Retrieval interface over the ChromaDB collection built by stage3_embed.py.

Importable (`from retrieve import Retriever`) and runnable as a CLI:

    python retrieve.py "how to find number of trailing zeros in a factorial"
    python retrieve.py --k 5 --chunk-type table "speed distance time formulas"
"""

import argparse
import os
import sys
from pathlib import Path

# Corpus text carries typographic unicode (thin spaces, dashes) that the
# default cp1252 Windows console cannot encode.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

torch.set_num_threads(os.cpu_count())

import chromadb
from sentence_transformers import SentenceTransformer

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
CHROMA_DIR = OUT_DIR / "chroma_db"
COLLECTION_NAME = "cat_prep"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# bge models expect this instruction prefix on the QUERY side only; the
# corpus side was embedded with the "passage: " prefix in stage3_embed.py.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Retriever:
    def __init__(self, model_name: str = DEFAULT_MODEL, chroma_dir: Path = CHROMA_DIR):
        self.model_name = model_name
        self.is_bge = "bge" in model_name.lower()
        self.model = SentenceTransformer(model_name)
        client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = client.get_collection(COLLECTION_NAME)

    def _embed_query(self, query: str):
        text = f"{BGE_QUERY_PREFIX}{query}" if self.is_bge else query
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

    def search(self, query: str, k: int = 8, chunk_type: str | None = None,
               source_contains: str | None = None) -> list:
        where = {}
        if chunk_type:
            where["chunk_type"] = chunk_type
        # Chroma needs a slightly wider net when we post-filter by substring.
        n_results = k * 5 if source_contains else k

        res = self.collection.query(
            query_embeddings=[self._embed_query(query)],
            n_results=n_results,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            if source_contains and source_contains.lower() not in meta["source_file"].lower():
                continue
            hits.append({
                "text": doc,
                "source_file": meta["source_file"],
                "heading": meta.get("heading") or None,
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "chunk_type": meta.get("chunk_type"),
                "distance": dist,
                "score": 1.0 - dist,  # cosine space -> similarity
            })
            if len(hits) >= k:
                break
        return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--chunk-type", choices=["text", "table"], default=None)
    parser.add_argument("--source-contains", default=None)
    parser.add_argument("--full", action="store_true", help="print full chunk text instead of a preview")
    args = parser.parse_args()

    r = Retriever()
    hits = r.search(args.query, k=args.k, chunk_type=args.chunk_type,
                    source_contains=args.source_contains)

    print(f"\nQuery: {args.query}\n{'=' * 78}")
    for i, h in enumerate(hits, 1):
        loc = f"p{h['page_start']}" if h["page_start"] == h["page_end"] else f"p{h['page_start']}-{h['page_end']}"
        print(f"\n[{i}] score={h['score']:.3f}  {h['chunk_type']}  {h['source_file']} ({loc})")
        if h["heading"]:
            print(f"    heading: {h['heading']}")
        body = h["text"] if args.full else h["text"][:400].replace("\n", " ")
        print(f"    {body}{'' if args.full else '...'}")


if __name__ == "__main__":
    main()
