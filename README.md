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
        ├─ stage2_chunk.py ────► output/chunks.jsonl         (fast: pure logic)
        │      heading-aware section chunking · dedupe · 1600-char hard cap
        │
        ├─ stage3_embed.py ────► output/chroma_db/           (bge-small-en-v1.5)
        │
        ├─ retrieve.py ────────► top-k semantic search
        │
        └─ generate.py ────────► validated quiz JSON  (Groq Llama 3.3 / Gemini)
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

# 3. chunk (fast, rebuilds from cache every run)
python scripts/stage2_chunk.py

# 4. embed into ChromaDB (--resume to continue an interrupted run)
python scripts/stage3_embed.py

# 5. query
python scripts/retrieve.py "how do I count trailing zeros in a factorial"

# 6. evaluate retrieval
python scripts/eval_retrieval.py --k 5

# 7. generate a quiz  (set GROQ_API_KEY or GEMINI_API_KEY first)
python scripts/generate.py "time speed distance shortcuts" --n 5
```

---

## Generation contract

`generate.py` retrieves top-k chunks, passes **only** those as context, and forces JSON
output. Every question is validated against a Pydantic schema before it is returned:

```json
{
  "question": "...",
  "options": ["...", "...", "...", "..."],
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

---

## Repository layout

```
scripts/
  step1_survey.py      corpus census + OCR-vs-text ratio sampling
  peek_archives.py     inspect zip/rar contents without extracting
  extract.py           per-document extraction (PDF/HTML/docx) + OCR fallback
  chunk.py             heading detection, section chunking, table rendering
  stage1_extract.py    parallel extraction driver, resumable
  stage2_chunk.py      chunking driver + deduplication
  stage3_embed.py      embedding + ChromaDB load, resumable
  retrieve.py          Retriever class + query CLI
  eval_retrieval.py    hit@k against the eval set
  generate.py          retrieval-grounded quiz generation, schema-validated
eval/
  eval_set.json        30 hand-built Q/A pairs
  results.json         latest eval run
output/
  extracted/           cached per-document extraction (gitignored)
  chunks.jsonl         final chunk store
  chroma_db/           persistent vector store (gitignored)
```
