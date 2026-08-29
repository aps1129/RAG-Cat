"""
Step 2d: Link question files to their solution/answer files.

A large chunk of this corpus stores a question paper and its worked
solutions as SEPARATE files or folders (Question/07.pdf <-> Answer_/07.pdf;
"IIFT 2020.pdf" <-> "IIFT 2020 Soln.pdf"). Chunking treats these as
unrelated documents. Since the project goal is retrieving a shortcut
alongside the question it explains, a retrieved question chunk whose
matching shortcut lives in a separate "solution" file needs that file
findable -- this stage builds that map.

Two independent matching passes, both driven by the same idea: strip known
question/answer vocabulary out of a name and see what's left.

  1. Sibling-FOLDER pairing: two folders sharing a parent are a pair if their
     names are identical once question/answer keywords are stripped (e.g.
     "Question" and "Answer_" both strip to ""; "CDC- LRDI" and
     "CDC- LRDT Sol_" both strip to "cdc lrdi"). Files inside a matched pair
     are linked by exact filename first, then by the same stripped-signature
     approach.

  2. Sibling-FILE pairing: within a single folder (no recursion), filenames
     are grouped by their stripped signature; a group containing both an
     unambiguous question-side name and an answer-side name is linked
     (e.g. "IIFT 2020.pdf" <-> "IIFT 2020 Soln.pdf").

Output: output/qa_links.json, a flat dict of
    {relative_path: {"role": "question"|"answer", "linked_to": relative_path,
                      "match_type": "..."}}

Usage:
    python link_qa.py [--source PATH]
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_SOURCE = r"C:\Users\ARNAV\Desktop\extrass\CAT"
OUT_PATH = Path(__file__).resolve().parent.parent / "output" / "qa_links.json"
SUPPORTED_EXTS = {".pdf", ".htm", ".html", ".docx"}


def load_links(path: Path = OUT_PATH) -> dict:
    """Load qa_links.json with paths normalized to forward slashes, matching
    the source_file convention used throughout chunk.py / chunks.jsonl.
    Shared by stage2_chunk.py (attaches link metadata to chunks) and
    generate.py (expands retrieval context across a linked file)."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        k.replace("\\", "/"): {"role": v["role"], "linked_to": v["linked_to"].replace("\\", "/")}
        for k, v in raw.items()
    }

# Whole-token matches only (never substring) so "QA" the subject folder is
# never mistaken for a question/answer marker.
QUESTION_TOKENS = {"question", "questions", "ques", "qn", "qns", "q", "paper", "papers"}
ANSWER_TOKENS = {
    "answer", "answers", "ans", "solution", "solutions", "sol", "soln", "sols",
    "explanation", "explanations", "explaination", "explainations", "key", "keys",
}


def tokenize(name: str) -> list:
    return [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]


def classify(tokens: list) -> str:
    has_q = any(t in QUESTION_TOKENS for t in tokens)
    has_a = any(t in ANSWER_TOKENS for t in tokens)
    if has_a and not has_q:
        return "answer"
    if has_q and not has_a:
        return "question"
    if has_a and has_q:
        return "answer"  # e.g. "question paper with solutions" -> treat as the solved side
    return "unmarked"


def strip_signature(name: str) -> str:
    """Core identity of a name with question/answer vocabulary removed."""
    tokens = [t for t in tokenize(name) if t not in QUESTION_TOKENS and t not in ANSWER_TOKENS]
    return " ".join(tokens)


def loose_signature(name: str) -> str:
    """Fallback: same idea but also collapses all whitespace, for names that
    differ only in spacing ("NMAT 1" vs "NMAT1")."""
    return strip_signature(name).replace(" ", "")


def link_pair_with_roles(a: Path, role_a: str, b: Path, role_b: str, links: dict, match_type: str) -> bool:
    if role_a == role_b:
        return False  # can't tell which side is which -- skip rather than guess
    ra = "question" if role_a != "answer" else "answer"
    rb = "answer" if ra == "question" else "question"
    links[str(a)] = {"role": ra, "linked_to": str(b), "match_type": match_type}
    links[str(b)] = {"role": rb, "linked_to": str(a), "match_type": match_type}
    return True


def link_pair(a: Path, b: Path, links: dict, match_type: str) -> bool:
    """Classify role from the FILENAME itself. Only valid when there is no
    higher-level (folder) role signal available -- i.e. the same-folder-file
    pairing pass. Files inside an already-classified Question/Answer folder
    must use link_pair_with_roles instead: a bare "01.pdf" carries no
    question/answer vocabulary of its own, so re-deriving role from the
    filename there silently drops the match."""
    role_a = classify(tokenize(a.stem))
    role_b = classify(tokenize(b.stem))
    return link_pair_with_roles(a, role_a, b, role_b, links, match_type)


def link_sibling_folders(source: Path, all_dirs: list, links: dict, stats: dict):
    by_parent = defaultdict(list)
    for d in all_dirs:
        by_parent[d.parent].append(d)

    for parent, dirs in by_parent.items():
        if len(dirs) < 2:
            continue
        by_sig = defaultdict(list)
        for d in dirs:
            by_sig[strip_signature(d.name)].append(d)

        for sig, group in by_sig.items():
            if len(group) != 2:
                continue  # ambiguous (0, 1, or >2 candidates) -- skip rather than guess
            fa, fb = group
            role_a = classify(tokenize(fa.name))
            role_b = classify(tokenize(fb.name))
            if role_a == role_b:
                continue
            files_a = {f.name: f for f in fa.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS}
            files_b = {f.name: f for f in fb.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS}

            # Pass 1: exact filename match (the common case: 01.pdf <-> 01.pdf).
            # Role comes from the FOLDER classification, not the filename --
            # "01.pdf" carries no question/answer vocabulary of its own.
            for name, path_a in files_a.items():
                if name in files_b:
                    ok = link_pair_with_roles(
                        path_a.relative_to(source), role_a,
                        files_b[name].relative_to(source), role_b,
                        links, "folder_pair+exact_filename")
                    if ok:
                        stats["folder_exact"] += 1

            # Pass 2: fuzzy filename match for the leftovers in this folder pair
            remaining_a = {n: p for n, p in files_a.items() if n not in files_b}
            remaining_b = {n: p for n, p in files_b.items() if n not in files_a}
            link_files_by_signature(remaining_a, role_a, remaining_b, role_b,
                                    source, links, stats, "folder_pair+fuzzy_filename")


def link_files_by_signature(files_a: dict, role_a: str, files_b: dict, role_b: str,
                            source: Path, links: dict, stats: dict, match_type: str):
    sig_a = defaultdict(list)
    for name, p in files_a.items():
        sig_a[strip_signature(p.stem)].append(p)
    sig_b = defaultdict(list)
    for name, p in files_b.items():
        sig_b[strip_signature(p.stem)].append(p)

    for sig in set(sig_a) & set(sig_b):
        if sig and len(sig_a[sig]) == 1 and len(sig_b[sig]) == 1:
            if link_pair_with_roles(sig_a[sig][0].relative_to(source), role_a,
                                    sig_b[sig][0].relative_to(source), role_b,
                                    links, match_type):
                stats[match_type] += 1

    # Loose pass (whitespace-insensitive) on whatever the exact stripped
    # signature above still left unmatched.
    left_a = {n: p for n, p in files_a.items() if strip_signature(p.stem) not in sig_b}
    left_b = {n: p for n, p in files_b.items() if strip_signature(p.stem) not in sig_a}
    loose_a = defaultdict(list)
    for name, p in left_a.items():
        loose_a[loose_signature(p.stem)].append(p)
    loose_b = defaultdict(list)
    for name, p in left_b.items():
        loose_b[loose_signature(p.stem)].append(p)
    for sig in set(loose_a) & set(loose_b):
        if sig and len(loose_a[sig]) == 1 and len(loose_b[sig]) == 1:
            if link_pair_with_roles(loose_a[sig][0].relative_to(source), role_a,
                                    loose_b[sig][0].relative_to(source), role_b,
                                    links, match_type + "+loose"):
                stats[match_type + "_loose"] += 1


def link_sibling_files(source: Path, all_dirs: list, links: dict, stats: dict):
    for d in all_dirs:
        files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS]
        if len(files) < 2:
            continue
        by_sig = defaultdict(list)
        for f in files:
            by_sig[strip_signature(f.stem)].append(f)

        for sig, group in by_sig.items():
            if not sig or len(group) != 2:
                continue
            fa, fb = group
            rel_a, rel_b = fa.relative_to(source), fb.relative_to(source)
            if str(rel_a) in links or str(rel_b) in links:
                continue  # already linked via the folder-pair pass
            if link_pair(rel_a, rel_b, links, "same_folder_filename"):
                stats["same_folder_filename"] += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = Path(args.source)

    all_dirs = [source] + [p for p in source.rglob("*") if p.is_dir()]

    links = {}
    stats = defaultdict(int)

    link_sibling_folders(source, all_dirs, links, stats)
    link_sibling_files(source, all_dirs, links, stats)

    n_files = len({p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS})
    n_linked = len(links)

    # Every successful link_pair call writes exactly 2 dict entries, so the
    # match-event count and the resulting link count must agree. A mismatch
    # here means some match silently failed to write (the bug this guards
    # against: a role tie discarded after being counted as a success).
    n_match_events = sum(stats.values())
    assert n_linked == 2 * n_match_events, (
        f"stats/links mismatch: {n_match_events} successful matches implies "
        f"{2 * n_match_events} linked files, but links has {n_linked}"
    )
    # Every link must be symmetric: A -> B implies B -> A.
    for path, info in links.items():
        counterpart = links.get(info["linked_to"])
        assert counterpart is not None and counterpart["linked_to"] == path, (
            f"asymmetric link: {path} -> {info['linked_to']}"
        )

    print(f"Total supported files in corpus: {n_files}")
    print(f"Files linked to a counterpart:   {n_linked}  ({100 * n_linked / n_files:.1f}%)")
    print(f"Linked pairs:                    {n_linked // 2}")
    print("\nBy match type:")
    for mt, count in sorted(stats.items(), key=lambda kv: -kv[1]):
        print(f"  {mt:<35} {count}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(links, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
