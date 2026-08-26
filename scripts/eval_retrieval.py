"""
Step 4a: Measure retrieval quality against the hand-built eval set.

Reports hit@k for two independent checks per question:
  * content hit - did any retrieved chunk contain the expected material?
  * source hit  - did any retrieved chunk come from an expected document?

Content hit rate is the primary number: it measures whether the retrieved
context could actually answer the question. Source hit is a weaker sanity
check (the same concept is legitimately taught in several books).

Usage:
    python eval_retrieval.py [--k 5] [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retrieve import Retriever

EVAL_PATH = Path(__file__).resolve().parent.parent / "eval" / "eval_set.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "eval" / "results.json"


def content_hit(hits: list, needles: list) -> bool:
    for h in hits:
        low = h["text"].lower()
        if any(n.lower() in low for n in needles):
            return True
    return False


def source_hit(hits: list, fragments: list) -> bool:
    for h in hits:
        low = h["source_file"].lower()
        if any(fr.lower() in low for fr in fragments):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    spec = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    items = spec["items"]

    print(f"Loading retriever ...")
    r = Retriever()
    print(f"Evaluating {len(items)} questions at k={args.k}\n")

    n_content = 0
    n_source = 0
    per_section = {}
    misses = []
    records = []

    for item in items:
        hits = r.search(item["question"], k=args.k)
        c_ok = content_hit(hits, item["must_contain_any"])
        s_ok = source_hit(hits, item["expected_source_any"])
        n_content += c_ok
        n_source += s_ok

        sec = item["section"]
        st = per_section.setdefault(sec, {"n": 0, "content": 0, "source": 0})
        st["n"] += 1
        st["content"] += c_ok
        st["source"] += s_ok

        flag = "OK  " if c_ok else "MISS"
        print(f"[{flag}] {item['id']:<9} content={'Y' if c_ok else 'n'} source={'Y' if s_ok else 'n'}  "
              f"top={hits[0]['score']:.3f}  {item['topic']}")
        if not c_ok:
            misses.append(item)
        if args.verbose or not c_ok:
            for i, h in enumerate(hits[:3], 1):
                print(f"        {i}. {h['score']:.3f} {h['source_file'][:70]} p{h['page_start']}")
                print(f"           {h['text'][:130].replace(chr(10), ' ')}...")

        records.append({
            "id": item["id"],
            "section": sec,
            "topic": item["topic"],
            "question": item["question"],
            "content_hit": bool(c_ok),
            "source_hit": bool(s_ok),
            "top_score": hits[0]["score"] if hits else None,
            "retrieved": [
                {"source_file": h["source_file"], "page_start": h["page_start"],
                 "score": round(h["score"], 4), "chunk_type": h["chunk_type"]}
                for h in hits
            ],
        })

    n = len(items)
    print(f"\n{'=' * 62}")
    print(f"content hit@{args.k}: {n_content}/{n} = {100 * n_content / n:.0f}%   <- primary")
    print(f"source  hit@{args.k}: {n_source}/{n} = {100 * n_source / n:.0f}%")
    print(f"\nBy section:")
    for sec, st in sorted(per_section.items()):
        print(f"  {sec:<6} content {st['content']}/{st['n']}   source {st['source']}/{st['n']}")

    if misses:
        print(f"\nContent misses ({len(misses)}):")
        for m in misses:
            print(f"  {m['id']} [{m['topic']}] {m['question']}")

    RESULTS_PATH.write_text(json.dumps({
        "k": args.k,
        "n_questions": n,
        "content_hit_rate": round(n_content / n, 4),
        "source_hit_rate": round(n_source / n, 4),
        "by_section": per_section,
        "records": records,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
