"""
Step 1: Survey the CAT prep source folder.

- Walks the source directory and reports a breakdown of file types/counts/sizes.
- Samples N random PDFs, extracts text with PyMuPDF, and flags each as
  clean / garbled / empty (likely-scanned) based on simple heuristics.

Usage:
    python step1_survey.py [--source PATH] [--sample-size 10] [--seed 42] [--max-pages 20]
"""

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF

# Windows consoles default to cp1252, which chokes on emoji/unicode file names.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SOURCE = r"C:\Users\ARNAV\Desktop\extrass\CAT"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Heuristic thresholds
MIN_CHARS_PER_PAGE = 40       # below this average -> likely empty/scanned
MIN_ALNUM_RATIO = 0.55        # below this -> likely garbled encoding
MIN_PRINTABLE_RATIO = 0.85    # below this -> likely garbled/binary junk


def human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f}{unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f}PB"


def walk_and_survey(source: Path):
    ext_counts = Counter()
    ext_sizes = defaultdict(int)
    folder_count = 0
    file_count = 0
    total_size = 0
    all_pdfs = []
    errors = []

    for root, dirs, files in Path.walk(source) if hasattr(Path, "walk") else _os_walk(source):
        folder_count += 1
        for name in files:
            fpath = root / name
            file_count += 1
            ext = fpath.suffix.lower() or "(no extension)"
            ext_counts[ext] += 1
            try:
                size = fpath.stat().st_size
            except OSError as e:
                errors.append(f"stat failed: {fpath} ({e})")
                size = 0
            ext_sizes[ext] += size
            total_size += size
            if ext == ".pdf":
                all_pdfs.append(fpath)

    return {
        "folder_count": folder_count,
        "file_count": file_count,
        "total_size": total_size,
        "ext_counts": ext_counts,
        "ext_sizes": ext_sizes,
        "all_pdfs": all_pdfs,
        "errors": errors,
    }


def _os_walk(source: Path):
    import os
    for dirpath, _dirnames, filenames in os.walk(source):
        yield Path(dirpath), _dirnames, filenames


def classify_pdf(fpath: Path, max_pages: int) -> dict:
    result = {
        "path": str(fpath),
        "size": None,
        "page_count": None,
        "pages_sampled": 0,
        "total_chars": 0,
        "avg_chars_per_page": 0.0,
        "alnum_ratio": None,
        "printable_ratio": None,
        "verdict": None,
        "note": "",
    }
    try:
        result["size"] = fpath.stat().st_size
        doc = fitz.open(fpath)
        result["page_count"] = doc.page_count
        n_pages = min(max_pages, doc.page_count) if doc.page_count else 0
        result["pages_sampled"] = n_pages

        text_parts = []
        for i in range(n_pages):
            page = doc.load_page(i)
            text_parts.append(page.get_text("text"))
        doc.close()

        full_text = "".join(text_parts)
        stripped = full_text.strip()
        result["total_chars"] = len(stripped)
        result["avg_chars_per_page"] = len(stripped) / n_pages if n_pages else 0.0

        if not stripped or result["avg_chars_per_page"] < MIN_CHARS_PER_PAGE:
            result["verdict"] = "EMPTY/SCANNED"
            result["note"] = "little or no extractable text -> needs OCR"
            return result

        alnum = sum(1 for c in stripped if c.isalnum())
        printable = sum(1 for c in stripped if c.isprintable() or c in "\n\t")
        alnum_ratio = alnum / len(stripped)
        printable_ratio = printable / len(stripped)
        result["alnum_ratio"] = round(alnum_ratio, 3)
        result["printable_ratio"] = round(printable_ratio, 3)

        # Heuristic: lots of replacement/control chars or too few alnum chars
        # relative to length suggests broken font encoding -> garbled text.
        weird_char_ratio = len(re.findall(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]", stripped)) / len(stripped)

        if printable_ratio < MIN_PRINTABLE_RATIO or weird_char_ratio > 0.02 or alnum_ratio < MIN_ALNUM_RATIO:
            result["verdict"] = "GARBLED"
            result["note"] = f"low alnum/printable ratio (alnum={alnum_ratio:.2f}, printable={printable_ratio:.2f}) -> likely broken encoding, needs OCR fallback"
        else:
            result["verdict"] = "CLEAN"
            result["note"] = "text extraction looks good"

    except Exception as e:
        result["verdict"] = "ERROR"
        result["note"] = f"{type(e).__name__}: {e}"

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-pages", type=int, default=20,
                         help="max pages to scan per sampled PDF (perf guard for huge eBooks)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"Source folder not found: {source}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {source} ...")
    survey = walk_and_survey(source)

    print("\n=== File type breakdown ===")
    print(f"Folders: {survey['folder_count']}")
    print(f"Files:   {survey['file_count']}")
    print(f"Total size: {human_size(survey['total_size'])}")
    print()
    print(f"{'Extension':<18}{'Count':>8}{'Total Size':>14}")
    for ext, count in survey["ext_counts"].most_common():
        print(f"{ext:<18}{count:>8}{human_size(survey['ext_sizes'][ext]):>14}")

    if survey["errors"]:
        print(f"\n{len(survey['errors'])} stat errors encountered (see report JSON).")

    all_pdfs = survey["all_pdfs"]
    print(f"\n=== PDF sampling ===")
    print(f"Total PDFs found: {len(all_pdfs)}")

    if not all_pdfs:
        print("No PDFs found, skipping extraction sample.")
        sample = []
    else:
        rng = random.Random(args.seed)
        sample_size = min(args.sample_size, len(all_pdfs))
        sample = rng.sample(all_pdfs, sample_size)
        print(f"Sampling {sample_size} PDFs (seed={args.seed}), scanning up to {args.max_pages} pages each...\n")

    results = []
    for fpath in sample:
        r = classify_pdf(fpath, args.max_pages)
        results.append(r)
        rel = fpath.relative_to(source)
        print(f"[{r['verdict']:<13}] {rel}")
        print(f"    size={human_size(r['size'] or 0):<10} pages={r['page_count']} sampled={r['pages_sampled']} "
              f"avg_chars/page={r['avg_chars_per_page']:.1f} alnum_ratio={r['alnum_ratio']} printable_ratio={r['printable_ratio']}")
        print(f"    {r['note']}\n")

    verdict_counts = Counter(r["verdict"] for r in results)
    print("=== Sample summary ===")
    for v, c in verdict_counts.most_common():
        print(f"  {v}: {c}/{len(results)}")

    if verdict_counts:
        needs_ocr = verdict_counts.get("EMPTY/SCANNED", 0) + verdict_counts.get("GARBLED", 0)
        pct = 100 * needs_ocr / len(results)
        print(f"\nEstimated OCR-fallback rate from this sample: {pct:.0f}% "
              f"({needs_ocr}/{len(results)} likely need OCR or table-aware extraction)")

    # Save full report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"step1_survey_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report = {
        "source": str(source),
        "timestamp": datetime.now().isoformat(),
        "seed": args.seed,
        "sample_size_requested": args.sample_size,
        "max_pages_per_pdf": args.max_pages,
        "folder_count": survey["folder_count"],
        "file_count": survey["file_count"],
        "total_size_bytes": survey["total_size"],
        "ext_counts": dict(survey["ext_counts"]),
        "ext_sizes_bytes": dict(survey["ext_sizes"]),
        "total_pdfs": len(all_pdfs),
        "errors": survey["errors"],
        "sample_results": results,
        "verdict_counts": dict(verdict_counts),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    main()
