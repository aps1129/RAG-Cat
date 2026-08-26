"""Peek inside zip/rar archives under the CAT source folder without extracting."""
import sys
import zipfile
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SOURCE = Path(r"C:\Users\ARNAV\Desktop\extrass\CAT")

for zpath in sorted(SOURCE.rglob("*.zip")):
    print(f"\n=== {zpath.relative_to(SOURCE)} ===")
    try:
        with zipfile.ZipFile(zpath) as zf:
            names = zf.namelist()
            ext_counts = Counter(Path(n).suffix.lower() for n in names if not n.endswith("/"))
            print(f"  {len(names)} entries")
            for ext, count in ext_counts.most_common():
                print(f"    {ext or '(no ext)'}: {count}")
    except zipfile.BadZipFile as e:
        print(f"  ERROR: {e}")

for rpath in sorted(SOURCE.rglob("*.rar")):
    print(f"\n=== {rpath.relative_to(SOURCE)} (RAR) ===")
    try:
        import rarfile
        with rarfile.RarFile(rpath) as rf:
            names = rf.namelist()
            ext_counts = Counter(Path(n).suffix.lower() for n in names if not n.endswith("/"))
            print(f"  {len(names)} entries")
            for ext, count in ext_counts.most_common():
                print(f"    {ext or '(no ext)'}: {count}")
    except ImportError:
        print("  (rarfile module not installed - skipping, will list via 7z if available)")
    except Exception as e:
        print(f"  ERROR: {e}")
