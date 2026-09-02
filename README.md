# CAT-RAG

A retrieval-augmented generation pipeline over ~1,300 real CAT (Common Admission Test)
prep documents — coaching-institute practice sheets, past papers, and reference
textbooks — that retrieves relevant material and generates quiz questions **using the
shortcuts that actually appear in the source material**, as validated structured JSON.

Everything except the final generation call runs locally and free: local embedding
model, local vector store, no paid API for ingestion or retrieval.

---

## Pipeline

```
source corpus (1,266 files, 1.8 GB)
        │
        ├─ stage1_extract.py ──► output/extracted/*.json     (slow: PDF parse + OCR)
        │      PyMuPDF text · pytesseract OCR fallback · table detection
        │
        ├─ link_qa.py ─────────► output/qa_links.json        (fast: pure logic)
        │      pairs question files/folders with their solution counterpart
        │
        ├─ stage2_chunk.py ────► output/chunks.jsonl         (fast: pure logic)
        │      heading-aware section chunking · dedupe · qa_links applied · 1600-char cap
        │
        ├─ stage3_embed.py ────► output/chroma_db/           (bge-small-en-v1.5)
        │
        ├─ retrieve.py ────────► top-k semantic search
        │
        └─ generate.py ────────► validated quiz JSON  (Groq / Gemini)
               │
               └─ expands each retrieved question chunk with its linked
                  solution chunk (or vice versa) before prompting the LLM
```

Extraction and chunking are **deliberately separate stages**. Extraction is the
expensive, OCR-heavy step and is cached per-document; chunking reads that cache and
rebuilds in seconds. Iterating on chunking logic therefore costs seconds, not another
full OCR pass over the corpus.

---

## Corpus

Surveyed with `step1_survey.py` before any pipeline was designed, to measure the
OCR-vs-text ratio rather than assume it.

| Extension | Files | Size |
|---|---:|---:|
| `.pdf` | 1,134 | 1.8 GB |
| `.htm` / `.html` | 128 | 47.9 MB |
| `.docx` / `.doc` | 7 | ~640 KB |
| **Total supported** | **1,266** | |

**~93% of PDFs extract cleanly with PyMuPDF.** The rest split into genuinely scanned
documents (image-only, need OCR — mostly standalone reference textbooks like *Quantum
CAT* and Nishit Sinha's QADI) and a small tail with broken font encodings. OCR is
therefore a *fallback path*, not the main path — which is what made a 1.8 GB corpus
tractable on CPU.

> The stack originally specified `ebooklib` for EPUBs. A full walk of the corpus —
> including inside all 11 zip/rar archives — found **zero `.epub` files**, so that
> dependency was dropped rather than carried dead.

---

## Design decisions worth explaining

**Semantic chunking, not fixed-token windows.** Chunks are bounded by detected section
headings (font size + weight relative to the document's modal body size), then packed
to a 1,600-character cap on paragraph boundaries. 94% of chunks carry a heading in
metadata, which gives retrieval results human-readable provenance.

**A guaranteed size cap.** The splitter degrades paragraph → sentence → word boundary.
The word-boundary fallback exists because real PDFs produce "sentences" of arbitrary
length: PyMuPDF's line joining sometimes swallows the space after a period, defeating
sentence segmentation entirely. Without that last resort, one RC passage produced a
22,658-character chunk that would have been silently truncated at embed time.

**False-positive table rejection.** PyMuPDF's `find_tables()` reads the column
structure of justified prose as a table, emitting a single-cell "table" holding an
entire reading-comprehension passage. Tables are accepted only if they have ≥2 columns,
≥2 rows, and no cell longer than 400 characters.

**Question/solution linkage.** A large fraction of the corpus stores a question
paper and its worked solutions as separate files or folders --
`Question/07.pdf` next to `Answer_/07.pdf`, or `IIFT 2020.pdf` next to
`IIFT 2020 Soln.pdf` -- so a question chunk and the shortcut that solves it can
end up in entirely different documents. Since the whole point of this project
is retrieving a shortcut *alongside* the question it explains, that split
mattered enough to build `link_qa.py`: it pairs files by stripping known
question/answer vocabulary from folder and file names and matching what's
left, using three passes (exact-filename sibling folders, fuzzy-filename
sibling folders, sibling files in the same folder). It links **496 files into
248 pairs (39% of the corpus)**, verified by hand-sampling matches across all
three passes plus every low-confidence fuzzy match -- no false positives found.
`generate.py` uses the map at retrieval time: any retrieved chunk that has a
linked counterpart pulls in that file's nearest-page chunk as extra context,
labeled `[Linked SOLUTION for Excerpt N]` / `[Linked QUESTION for Excerpt N]`
so the model knows which is which.

Two bugs surfaced building this, both instructive:
- **Role came from the wrong level.** A bare `01.pdf` carries no
  question/answer vocabulary of its own -- only its *parent folder*
  (`Question/` vs `Answer_/`) does. The first version re-derived role from the
  filename for every match, which silently discarded 160 of 172 folder-based
  matches (they'd tie as "unmarked" vs "unmarked"). Fixed by threading the
  folder-level role down into the file-matching pass instead of re-guessing it.
  Caught by a self-consistency assertion (match-event count must equal
  linked-file count / 2), not by eyeballing output.
- **Dedup silently breaks file-path-keyed links.** `qa_links.json` is built
  against the pre-dedup file tree, but dedup drops an entire duplicate file's
  chunks in favor of an identical copy elsewhere -- so a link pointing at the
  now-dropped path pointed at nothing. Fixed by voting: for each original
  file, find which surviving file most of its chunks now belong to, and remap
  both sides of every link through that alias before annotating. 221 of 496
  linked files needed remapping -- this wasn't a rare edge case.

**Deduplication.** The corpus ships the same books at multiple paths (archive
re-extractions, PYQ folders mirroring subject folders). 24,310 byte-identical chunks
were dropped — 17% of the raw total — because duplicate passages otherwise consume
multiple top-k slots for one underlying piece of content.

**BGE prefixing is asymmetric.** Per the model card, the retrieval instruction goes on
the **query** side only; passages are embedded raw. The E5-style `"passage: "` prefix
is *wrong* for BGE and misaligns the two vector spaces. (Caught after a full embedding
pass had already been run with it — the index was rebuilt.)

**Length-sorted batching — a 7.7x speedup.** Transformer cost scales with the *longest*
sequence in a batch, since everything shorter is padded up to it. Chunks here range from
10 to 1,600 characters, so batching in corpus order padded nearly every batch to the
maximum. Sorting all chunks by length before batching made each batch homogeneous and
took embedding throughput from **8.9 to 68 vectors/sec** — a projected 3 hours down to
~25 minutes of wall time. Insertion order is irrelevant to a vector store, so this costs
nothing.

> Worth recording how that was found: a first benchmark suggested ~198 chunks/sec, but
> it measured 90-character strings. Real chunks average 829 characters. Benchmarking on
> unrepresentative input produced an ETA that was off by more than an order of
> magnitude. A second, isolated benchmark then ruled out ChromaDB (227–270 inserts/sec,
> never the bottleneck) and correctly localised the cost to the encoder.

---

## Setup

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

System dependencies for the OCR fallback:

- **Tesseract OCR** — `winget install UB-Mannheim.TesseractOCR`
- **Poppler** — prebuilt Windows binaries, unpacked to `tools/poppler_extracted/`
  (required by `pdf2image` to rasterize pages)

Both paths are set at the top of `scripts/extract.py`.

## Running

```bash
# 1. survey the corpus (no pipeline needed)
python scripts/step1_survey.py --sample-size 50

# 2. extract (slow, resumable, crash-resilient — safe to re-run)
python scripts/stage1_extract.py --workers 4

# 3. link question files to their solution counterparts (fast, pure logic)
python scripts/link_qa.py

# 4. chunk (fast, rebuilds from cache every run; applies qa_links.json)
python scripts/stage2_chunk.py

# 5. embed into ChromaDB (--resume to continue an interrupted run)
python scripts/stage3_embed.py

# 6. query
python scripts/retrieve.py "how do I count trailing zeros in a factorial"

# 7. evaluate retrieval
python scripts/eval_retrieval.py --k 5

# 8. generate a quiz  (set GROQ_API_KEY or GEMINI_API_KEY first)
python scripts/generate.py "time speed distance shortcuts" --n 5

# 9. or launch the web UI instead of the CLI
GROQ_API_KEY=... uvicorn webapp.main:app --port 8000   # open http://localhost:8000
```

---

## Web UI

A thin FastAPI layer over `retrieve.py` + `generate.py` -- no pipeline logic is
duplicated, just exposed over HTTP. The embedding model and Q/A link index load once
at startup and are reused across requests rather than reloaded per call.

- Type a topic (or click a suggested one in the sidebar) to generate a quiz.
- Click an option to see it graded inline -- correct/incorrect highlighted, with the
  cited shortcut revealed underneath.
- Every response includes a collapsible **Sources** panel listing exactly which
  chunks (and, where relevant, which linked solution file) produced the quiz, so any
  question is traceable back to its source page.
- Plain grayscale UI by design, in both light and dark mode (follows the OS theme) --
  color is used only for the correct/incorrect grading state, nowhere else.

```bash
GROQ_API_KEY=... uvicorn webapp.main:app --port 8000
```

Then open `http://localhost:8000`. `webapp/main.py` is the whole backend; the frontend
is plain HTML/CSS/JS in `webapp/static/` with no build step and no framework.

---

## Generation contract

`generate.py` retrieves top-k chunks, passes **only** those as context, and forces JSON
output. Every question is validated against a Pydantic schema before it is returned:

```json
{
  "question": "...",
  "options": ["...", "...", "...", "..."],
  "reasoning": "...",
  "correct_answer": "...",
  "shortcut_used": "..."
}
```

Validation is not decorative. It rejects the two failure modes these models actually
exhibit: emitting a `correct_answer` that is not character-for-character one of its own
four options, and returning the wrong number of options. Markdown-fenced and
prose-wrapped JSON are recovered rather than treated as failures.

Output also carries the `retrieved_context` (source file, page, similarity score) that
produced it, so any generated question can be traced back to the page it came from.

### Catching arithmetic errors

The generation eval below originally measured the model getting arithmetic wrong on
about 1 in 6 questions even when retrieval and the cited shortcut were correct. Two
fixes address that, both in `generate.py`:

1. **`reasoning` is a required schema field, ordered before `correct_answer`.** The
   system prompt requires the model to work the arithmetic step by step in `reasoning`
   before committing to `correct_answer`, instead of jumping straight to an answer with
   no scratch space. This is free -- no extra API call -- and targets exactly the
   failure mode the eval caught (correct method, sloppy arithmetic on top of it).
2. **Every question is judged and, if flagged, repaired.** `verify_and_repair()` sends
   each generated question to the same fact-checking prompt `eval_generation.py` uses
   (`judge_question`, moved into `generate.py` so both share one copy): re-derive the
   answer from the source excerpt from scratch and check whether `correct_answer`
   actually holds up. A question the judge flags gets one repair attempt
   (`repair_question`) fed the judge's own note, then one re-check. The result carries
   `verified: true/false/null` and `verification_notes` on every question, so a question
   that still doesn't check out after repair isn't silently shipped as if it were fine --
   the web UI flags it inline, and the CLI prints `[UNVERIFIED]`.

This runs by default (`verify=True` in `generate_quiz`, on in the web UI and CLI); pass
`--no-verify` to `generate.py` or `eval_generation.py` to skip it. Two caveats worth
being honest about: it roughly doubles best-case token usage per question (a judge call,
plus a repair + re-judge pair for anything flagged), so expect it to hit the Groq
free-tier rate limit more often; and by default the judge is the same model that did the
generation, so a mistake the model is consistently confident about can pass its own
check -- pass `--judge-model` (or run the judge on the other provider) for a more
independent read when both API keys are available.

---

## Evaluation

`eval/eval_set.json` holds 30 hand-built questions spanning QA, DILR, VARC and GK. Each
was written **after reading a real chunk in the indexed corpus**, so the ground truth is
verified to exist rather than assumed, and each is phrased the way an aspirant would ask
it rather than as keywords lifted from the source text.

`eval_retrieval.py` reports two independent numbers:

- **content hit@k** (primary) — did any retrieved chunk contain the expected material?
  This measures whether the retrieved context could actually answer the question.
- **source hit@k** — did any retrieved chunk come from an expected document? A weaker
  check, since the same concept is legitimately taught in several books.

### Results (139,710 chunks, bge-small-en-v1.5)

| k | content hit | source hit |
|---:|---:|---:|
| 1 | **90%** (27/30) | 50% (15/30) |
| 3 | **100%** (30/30) | 80% (24/30) |
| 5 | **100%** (30/30) | 100% (30/30) |

Read these honestly:

- **content hit@1 = 90%** is the number that matters — for 27 of 30 questions the single
  top-ranked chunk already contained the answer. The three k=1 misses (remainder
  manipulation, negative work, successive percentage change) all recover by k=3.
- **hit@5 = 100% does not mean retrieval is perfect.** It means this eval saturates at
  k=5. The `must_contain_any` keywords are deliberately short, so the check rewards
  finding the right *material* rather than string-matching the question — which also
  makes it easy to pass. It is a regression guard, not a leaderboard score.
- **Low source hit@1 (50%) is not a failure.** CAT concepts are taught in many books at
  once; the top hit frequently comes from a perfectly good source that simply was not on
  the expected list. This is exactly why content hit is the primary metric.

Typical top-1 cosine similarity sits around 0.75–0.90.

### Generation eval (12 topics across QA/DILR/VARC, `eval_generation.py`)

Retrieval quality doesn't guarantee generation quality -- the LLM still has to reason
correctly from the context it's handed. This runs `generate.py` end-to-end on topics
distinct from the retrieval eval set, then has a *second*, independent LLM call
re-derive the answer from the same source excerpt from scratch and judge whether
`correct_answer` actually holds up and whether `shortcut_used` names something that
genuinely appears in the source rather than a plausible-sounding invention.

| Metric | Result (pre-fix baseline) |
|---|---:|
| Generation attempts (schema-valid JSON) | 12/12 (100%) |
| Judge: answer actually correct | 10/12 (83%) |
| Judge: shortcut genuinely grounded in source | 11/12 (92%) |
| Options structurally distinct | 12/12 (100%) |

This is a judge model checking another model's output, not ground truth -- but it
found two real problems, which is the point of running it:

- A **Time & Work question got the arithmetic wrong** (claimed 27 minutes; the judge's
  independent derivation and re-check gives 25). The retrieved shortcut itself was
  correctly cited; the model's own arithmetic on top of it was not.
- A **Critical Reasoning question misidentified the assumption**, and separately used
  "Assumption Identification" as `shortcut_used` -- a category label, not a technique
  that appears in the source. This is the VARC generic-shortcut limitation below,
  caught empirically rather than just predicted.

**This table is the baseline that motivated the fix in "Catching arithmetic errors"
above (`reasoning` field + judge/repair loop), not its result** -- it predates that
change. `eval_generation.py` now defaults to running with the fix on
(`verify=True`, matching what the CLI and web UI ship), so re-running it
(`python eval_generation.py --topics-per-section 4 --n-per-topic 1`) reports the
current, post-fix numbers, including `n_repaired_by_pipeline` (how many questions the
judge/repair loop actually touched) and, per record, `pipeline_verified` alongside the
eval's own independent `judge` verdict. Pass `--no-verify` to reproduce the table above
for comparison. `eval/generation_results.json` holds whichever run was saved last.

Practical takeaway: **treat generated QA/DILR questions as generally reliable and
generated VARC questions as needing a read-through** -- consistent with VARC material
being more explanatory than technique-based to begin with. Full per-question judge
output (including the two flagged records) is in `eval/generation_results.json`.

---

## Repository layout

```
scripts/
  step1_survey.py      corpus census + OCR-vs-text ratio sampling
  peek_archives.py     inspect zip/rar contents without extracting
  extract.py           per-document extraction (PDF/HTML/docx) + OCR fallback
  chunk.py             heading detection, section chunking, table rendering
  link_qa.py           pairs question files/folders with their solution counterpart
  stage1_extract.py    parallel extraction driver, resumable
  stage2_chunk.py      chunking driver + deduplication + qa_links application
  stage3_embed.py      embedding + ChromaDB load, resumable
  retrieve.py          Retriever class + query CLI
  eval_retrieval.py    hit@k against the eval set
  generate.py          retrieval-grounded quiz generation, schema-validated,
                       expands context across linked question/solution files
  eval_generation.py   generation-quality eval (LLM-as-judge)
webapp/
  main.py              FastAPI backend -- wraps retrieve.py + generate.py over HTTP
  static/              plain HTML/CSS/JS frontend, no build step
eval/
  eval_set.json         30 hand-built retrieval Q/A pairs
  results.json          latest retrieval eval run
  generation_results.json  latest generation-quality eval run
output/
  extracted/           cached per-document extraction (gitignored)
  qa_links.json        question <-> solution file map (248 pairs)
  chunks.jsonl         final chunk store
  chroma_db/           persistent vector store (gitignored)
```

## Known limitations

- **A minority of "solution" files have no real solution text.** Some mock
  tests in the corpus were exported as printed webpages with the solution
  panel still collapsed behind JavaScript, so the PDF's text layer only has
  the restated question and UI labels ("Answer key/Solution", "Bookmark") --
  no amount of better chunking recovers text that was never rendered. This
  surfaced during testing on one CL mock-review export; most linked answer
  files (Arun Sharma, NMAT, IIFT, XAT, TIME) do contain genuine worked
  explanations.
- **VARC has few *named* shortcuts to retrieve.** Unlike QA ("skipping zero
  concept", "alligation"), reading comprehension and verbal material rarely
  names a reusable technique -- `shortcut_used` for VARC questions tends to
  come out generic ("process of elimination") rather than citing something
  specific from the source, simply because the source itself is more
  explanatory than technique-based for this section.
- **Generation had a measured arithmetic error rate of ~1 in 6, now mitigated
  but not eliminated.** `eval_generation.py` (see Generation eval above) found the
  model got arithmetic wrong on ~17% of generated questions even with correct
  retrieved context and a correctly cited shortcut -- the failure was in the LLM's
  own reasoning on top of good context, not in retrieval. `generate.py` now requires
  a step-by-step `reasoning` field before `correct_answer`, and independently
  judges + repairs every question before returning it (see "Catching arithmetic
  errors" above) -- flagged-and-fixed questions carry `verified: true`, and a
  question that still can't be verified after one repair attempt is marked
  `verified: false` (surfaced in the web UI) rather than shipped silently. This
  narrows the gap but the judge is an LLM checking another LLM's work, not ground
  truth, so still treat generated quizzes as a draft to review, especially any
  question flagged unverified.
