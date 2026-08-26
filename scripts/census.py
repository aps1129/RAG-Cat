"""Fast census: page count per PDF, no text extraction. Used to size the full pipeline run."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import fitz

SOURCE = Path(r"C:\Users\ARNAV\Desktop\extrass\CAT")

rows = []
errors = []
for p in SOURCE.rglob("*.pdf"):
    try:
        doc = fitz.open(p)
        rows.append((doc.page_count, p))
        doc.close()
    except Exception as e:
        errors.append((p, str(e)))

rows.sort(reverse=True)
total_pages = sum(r[0] for r in rows)
print(f"Total PDFs: {len(rows)}  Total pages: {total_pages}  Errors: {len(errors)}")
print()
for bucket, lo, hi in [("0-20", 0, 20), ("21-50", 21, 50), ("51-100", 51, 100), ("101-300", 101, 300), ("301-600", 301, 600), ("600+", 601, 999999)]:
    n = sum(1 for c, _ in rows if lo <= c <= hi)
    pages = sum(c for c, _ in rows if lo <= c <= hi)
    print(f"  {bucket:<10} pages: {n:>5} files, {pages:>7} total pages")

print("\nTop 25 largest PDFs:")
for c, p in rows[:25]:
    print(f"  {c:>5} pages  {p.relative_to(SOURCE)}")

with open(Path(__file__).resolve().parent.parent / "reports" / "page_census.txt", "w", encoding="utf-8") as f:
    for c, p in rows:
        f.write(f"{c}\t{p}\n")
