"""
Step 2 (stage 1): Extraction only, cached to disk.

Runs extract_document() on every supported file and caches the raw result
(pages, text, spans, tables) as one JSON file per source doc under
output/extracted/. This is the slow, OCR-heavy stage. Once cached, chunking
logic can be iterated on cheaply via stage2_chunk.py without re-running OCR.

Resumable: a file is skipped if its cache JSON already exists.

Usage:
    python stage1_extract.py [--source PATH] [--workers 4] [--limit N]
"""

import argparse
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

DEFAULT_SOURCE = r"C:\Users\ARNAV\Desktop\extrass\CAT"
OUT_DIR = Path(__file__).resolve().parent.parent / "output"
EXTRACTED_DIR = OUT_DIR / "extracted"
LOG_PATH = OUT_DIR / "extract_log.jsonl"
SUPPORTED_EXTS = {".pdf", ".htm", ".html", ".docx"}


def cache_path(source: Path, path: Path) -> Path:
    rel = str(path.relative_to(source)).replace("\\", "/")
    h = hashlib.sha1(rel.encode("utf-8")).hexdigest()
    return EXTRACTED_DIR / f"{h}.json"


def process_one(args):
    path_str, source_str = args
    from extract import extract_document

    path = Path(path_str)
    source = Path(source_str)
    out_path = cache_path(source, path)
    t0 = time.time()
    try:
        extraction = extract_document(path)
        extraction["source_rel"] = str(path.relative_to(source)).replace("\\", "/")
        out_path.write_text(json.dumps(extraction, ensure_ascii=False), encoding="utf-8")
        return {
            "path": path_str,
            "status": "ok",
            "n_pages": len(extraction["pages"]),
            "extract_errors": extraction["errors"],
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        return {
            "path": path_str,
            "status": "error",
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.time() - t0, 2),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source = Path(args.source)
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)

    all_files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    todo = [p for p in all_files if not cache_path(source, p).exists()]
    if args.limit:
        todo = todo[: args.limit]

    print(f"Total supported files: {len(all_files)}")
    print(f"Already cached:        {len(all_files) - len(todo) if not args.limit else 'n/a'}")
    print(f"To process this run:   {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    n_ok = 0
    n_err = 0
    n_pool_restarts = 0
    t_start = time.time()
    total_todo = len(todo)
    completed_total = 0
    max_pool_restarts = 8

    with open(LOG_PATH, "a", encoding="utf-8") as log_f:
        while todo:
            try:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    futures = {pool.submit(process_one, (str(p), str(source))): p for p in todo}
                    for fut in as_completed(futures):
                        result = fut.result()
                        completed_total += 1
                        log_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        log_f.flush()

                        if result["status"] == "ok":
                            n_ok += 1
                        else:
                            n_err += 1
                            print(f"  ERROR [{result['path']}]: {result['error']}")

                        if completed_total % 25 == 0 or completed_total == total_todo:
                            elapsed = time.time() - t_start
                            rate = completed_total / elapsed if elapsed > 0 else 0
                            eta = (total_todo - completed_total) / rate if rate > 0 else float("inf")
                            print(f"  [{completed_total}/{total_todo}] ok={n_ok} err={n_err} "
                                  f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m")
                todo = []

            except BrokenProcessPool:
                n_pool_restarts += 1
                if n_pool_restarts > max_pool_restarts:
                    print(f"\nToo many pool restarts ({n_pool_restarts}), aborting. Re-run to resume.")
                    sys.exit(1)
                todo = [p for p in all_files if not cache_path(source, p).exists()]
                if args.limit:
                    todo = todo[: args.limit]
                print(f"\nWorker pool crashed (restart #{n_pool_restarts}). Resuming with {len(todo)} files remaining.\n")

    print(f"\nDone. ok={n_ok} err={n_err} pool_restarts={n_pool_restarts} time={(time.time() - t_start) / 60:.1f}m")
    print(f"Cache dir: {EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
