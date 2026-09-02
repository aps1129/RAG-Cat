"""
Step 4b: Retrieval-grounded quiz generation with structured output.

Pipeline: topic query -> retrieve top-k chunks from ChromaDB -> prompt an LLM
with ONLY that context -> parse and validate strict JSON matching the target
schema (question, options, correct_answer, shortcut_used).

Two free-tier backends are supported, selected with --provider:
  groq   - needs GROQ_API_KEY   (default model: llama-3.3-70b-versatile)
  gemini - needs GEMINI_API_KEY (default model: gemini-2.0-flash)

Set the key as an environment variable before running, e.g. in PowerShell:
    $env:GROQ_API_KEY = "..."

Usage:
    python generate.py "time speed distance shortcuts" --n 5
    python generate.py "profit and loss" --provider gemini --n 3 --out quiz.json
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Literal

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retrieve import Retriever
from link_qa import load_links as load_qa_links

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "output" / "chunks.jsonl"


# ----------------------------- schema -----------------------------

class QuizQuestion(BaseModel):
    question: str = Field(min_length=10)
    options: List[str] = Field(min_length=4, max_length=4)
    reasoning: str = Field(min_length=10)
    correct_answer: str
    shortcut_used: str = Field(min_length=5)

    @field_validator("correct_answer")
    @classmethod
    def answer_must_be_an_option(cls, v, info):
        options = info.data.get("options")
        if options and v not in options:
            raise ValueError(f"correct_answer {v!r} is not one of the options")
        return v


class Quiz(BaseModel):
    questions: List[QuizQuestion]


SYSTEM_PROMPT = """You are a CAT (Common Admission Test) exam question writer.

You will be given EXCERPTS from real CAT preparation material. Write quiz \
questions grounded ONLY in those excerpts.

Some excerpts are paired: a "[Linked SOLUTION for Excerpt N]" block gives the \
worked solution to the question in Excerpt N (the reverse, "[Linked QUESTION \
for Excerpt N]", also occurs). When you see a linked pair, prefer it -- the \
solution excerpt is usually where the actual named shortcut lives, not the bare \
question text.

Hard rules:
- Use only shortcuts, formulas and methods that actually appear in the excerpts. \
Never invent a shortcut.
- "shortcut_used" must name the specific technique from the excerpt and briefly \
state it. If an excerpt gives a named shortcut, use that name.
- Exactly 4 options per question. "correct_answer" must be character-for-character \
one of the 4 options.
- The question must be solvable from the excerpt content alone.
- Vary the numbers from the source examples so questions are fresh, but keep the \
underlying method identical.
- Fill in "reasoning" BEFORE deciding "correct_answer": work through the arithmetic \
step by step, plugging in the actual numbers you chose, the way a careful student \
checking their own work would. Then re-check each step against itself -- this is \
where models most often slip. "correct_answer" must be the value your own \
"reasoning" arrives at, not a number that merely looks plausible.

Return ONLY a JSON object of this exact shape, with no prose or markdown. Note the \
key order -- "reasoning" comes before "correct_answer" because you must derive the \
answer before stating it:
{"questions": [{"question": "...", "options": ["...","...","...","..."], \
"reasoning": "step-by-step derivation with the actual numbers ...", \
"correct_answer": "...", "shortcut_used": "..."}]}"""


def load_linked_chunk_index(qa_links: dict) -> dict:
    """Stream chunks.jsonl once, keeping only chunks that belong to a file
    which is the LINKED counterpart of something else (~7% of the corpus --
    see link_qa.py). This is what lets a retrieved question chunk pull in
    its worked-solution chunk even though semantic search alone wouldn't
    rank a mostly-numeric answer page highly for a text query."""
    wanted = {v["linked_to"] for v in qa_links.values()}
    if not wanted:
        return {}
    index = defaultdict(list)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["source_file"] in wanted:
                index[c["source_file"]].append(c)
    for chunks in index.values():
        chunks.sort(key=lambda c: (c["page_start"] or 0, c["chunk_id"]))
    return index


def find_linked_chunks(hit: dict, qa_links: dict, index: dict, max_per_hit: int = 2) -> list:
    """For a retrieved hit whose source file has a known counterpart, return
    the counterpart's chunks with the closest page number -- e.g. a question
    on page 3 of Question/07.pdf should pull the explanation for page 3 of
    Answer_/07.pdf, not page 1."""
    link = qa_links.get(hit["source_file"])
    if not link:
        return []
    candidates = index.get(link["linked_to"], [])
    if not candidates:
        return []
    # link["role"] is the HIT's own role; the counterpart is the opposite side.
    counterpart_role = "answer" if link["role"] == "question" else "question"
    target_page = hit["page_start"] or 0
    candidates = sorted(candidates, key=lambda c: abs((c["page_start"] or 0) - target_page))
    return [{**c, "_linked_role": counterpart_role} for c in candidates[:max_per_hit]]


def build_context(hits: list, qa_links: dict = None, linked_index: dict = None,
                  max_chars: int = 9000) -> tuple:
    """Returns (context_text, provenance_list). Each primary hit is followed
    immediately by its linked question/solution excerpt (if any) so the two
    stay visually paired for the model, rather than the linked material
    showing up in an unrelated position later in the prompt."""
    qa_links = qa_links or {}
    linked_index = linked_index or {}
    parts = []
    provenance = []
    used = 0
    seen_chunk_ids = {h["chunk_id"] for h in hits if "chunk_id" in h}

    def add_block(text: str) -> bool:
        nonlocal used
        if used + len(text) > max_chars:
            return False
        parts.append(text)
        used += len(text)
        return True

    for i, h in enumerate(hits, 1):
        loc = h["source_file"].split("/")[-1]
        if not add_block(f"[Excerpt {i} | {loc} | p{h['page_start']}]\n{h['text']}\n"):
            break
        provenance.append({"source_file": h["source_file"], "page_start": h["page_start"],
                           "score": round(h["score"], 4), "role": "retrieved"})

        for linked in find_linked_chunks(h, qa_links, linked_index):
            if linked.get("chunk_id") in seen_chunk_ids:
                continue  # already independently retrieved -- don't duplicate
            seen_chunk_ids.add(linked.get("chunk_id"))
            loc2 = linked["source_file"].split("/")[-1]
            label = "SOLUTION" if linked["_linked_role"] == "answer" else "QUESTION"
            block = f"[Linked {label} for Excerpt {i} | {loc2} | p{linked['page_start']}]\n{linked['text']}\n"
            if not add_block(block):
                continue
            provenance.append({"source_file": linked["source_file"], "page_start": linked["page_start"],
                               "score": None, "role": f"linked_{linked['_linked_role']}"})

    return "\n".join(parts), provenance


# ----------------------------- providers -----------------------------

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
MAX_RETRIES = 4

_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)s", re.I)


def _request_with_retry(post_fn, max_retries: int = MAX_RETRIES):
    """Retry on rate limits (429) and transient server errors, honoring the
    provider's own suggested wait time when it gives one (Groq's free tier
    reports this and it's usually just a few seconds -- far better than a
    fixed guess). Exponential backoff otherwise."""
    last_err = None
    for attempt in range(max_retries + 1):
        resp = post_fn()
        if resp.status_code == 200:
            return resp
        last_err = resp
        if resp.status_code not in RETRYABLE_STATUS or attempt == max_retries:
            break
        wait = 2 ** attempt
        match = _RETRY_AFTER_RE.search(resp.text)
        if match:
            wait = float(match.group(1)) + 0.5
        print(f"  [retry {attempt + 1}/{max_retries}] {resp.status_code}, waiting {wait:.1f}s ...",
              file=sys.stderr)
        time.sleep(wait)
    return last_err


def call_groq(system: str, user: str, model: str, temperature: float) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")

    def do_post():
        return requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )

    resp = _request_with_retry(do_post)
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API {resp.status_code}: {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


def call_gemini(system: str, user: str, model: str, temperature: float) -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")

    def do_post():
        return requests.post(
            GEMINI_URL.format(model=model),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "responseMimeType": "application/json",
                },
            },
            timeout=120,
        )

    resp = _request_with_retry(do_post)
    if resp.status_code != 200:
        raise RuntimeError(f"Gemini API {resp.status_code}: {resp.text[:400]}")
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


# ----------------------------- parsing -----------------------------

def extract_json(raw: str) -> dict:
    """Both backends are asked for strict JSON, but models still occasionally
    wrap it in a markdown fence or add a stray prose line. Recover from that
    rather than failing the whole generation."""
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fence:
        raw = fence.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            return json.loads(raw[start:end + 1])
        raise


# ----------------------------- verify + repair -----------------------------
#
# The reasoning field above cuts arithmetic slips at generation time, but
# eval_generation.py's LLM-as-judge measured a real residual error rate
# (~1 in 6 questions, see README "Known limitations"). This closes that loop:
# every generated question is independently re-derived by a judge call, and
# anything flagged gets one repair attempt fed the judge's own note before
# the quiz is returned. This is the same judging prompt eval_generation.py
# uses for its offline audit -- defined once here so neither copy drifts.

JUDGE_SYSTEM_PROMPT = """You are a strict CAT exam fact-checker. You will be given a \
source excerpt, a generated quiz question, its 4 options, the claimed correct \
answer, and a claimed "shortcut_used".

Judge two things independently:
1. answer_correct: working the question through yourself from first principles \
(not just trusting the claim), is "correct_answer" actually the right answer?
2. shortcut_grounded: does "shortcut_used" name a technique or fact that \
genuinely appears in the source excerpt, as opposed to a plausible-sounding \
generic phrase ("elimination method", "basic algebra") that isn't actually \
tied to anything specific in the excerpt?

Return ONLY this JSON shape, no prose:
{"answer_correct": true/false, "shortcut_grounded": true/false, \
"confidence": "high"|"medium"|"low", "notes": "one sentence explaining any problem found"}"""

REPAIR_SYSTEM_PROMPT = """You are a meticulous CAT exam question editor. A fact-checker \
flagged a problem with a quiz question you generated from the given source excerpts. \
Re-derive the answer step by step from the excerpts, fix whatever is actually wrong \
(usually an arithmetic slip), and change as little else as possible. Return ONLY the \
corrected JSON object, no prose or markdown."""


def judge_question(question: dict, context: str, provider: str, model: str,
                   temperature: float = 0.0) -> dict:
    """Independently re-derive the answer and check shortcut grounding for one
    generated question. Returns the judge's raw parsed JSON verdict."""
    caller = call_groq if provider == "groq" else call_gemini
    user = (
        f"Source excerpts:\n{context}\n\n"
        f"Generated question: {question['question']}\n"
        f"Options: {question['options']}\n"
        f"Claimed correct_answer: {question['correct_answer']}\n"
        f"Claimed shortcut_used: {question['shortcut_used']}\n"
    )
    raw = caller(JUDGE_SYSTEM_PROMPT, user, model, temperature)
    return extract_json(raw)


def repair_question(question: dict, context: str, notes: str, provider: str,
                    model: str, temperature: float = 0.3) -> dict:
    """Ask the model to fix one flagged question in place. Raises if the
    repaired output doesn't parse or validate -- the caller decides what to
    do with the original in that case."""
    caller = call_groq if provider == "groq" else call_gemini
    user = (
        f"Source excerpts:\n{context}\n\n"
        f"You previously generated this question:\n"
        f"Question: {question['question']}\n"
        f"Options: {question['options']}\n"
        f"Reasoning: {question.get('reasoning', '')}\n"
        f"Correct answer: {question['correct_answer']}\n"
        f"Shortcut used: {question['shortcut_used']}\n\n"
        f"Problem found by the fact-checker: {notes}\n\n"
        f"Return a corrected version of ONLY this one question as a single JSON "
        f'object (not wrapped in a list): {{"question": "...", '
        f'"options": ["...","...","...","..."], "reasoning": "...", '
        f'"correct_answer": "...", "shortcut_used": "..."}}'
    )
    raw = caller(REPAIR_SYSTEM_PROMPT, user, model, temperature)
    data = extract_json(raw)
    return QuizQuestion.model_validate(data).model_dump()


def verify_and_repair(questions: List[dict], context: str, provider: str,
                      model: str, judge_model: str = None) -> List[dict]:
    """Judge every question; give the ones flagged as wrong one repair
    attempt, then re-judge the repair. Every returned question carries
    'verified' (True/False/None) and 'verification_notes' so callers can
    surface residual risk rather than silently trusting the fix."""
    judge_model = judge_model or model
    out = []
    for q in questions:
        try:
            verdict = judge_question(q, context, provider, judge_model)
        except Exception as e:
            result = dict(q, verified=None, verification_notes=f"judge call failed: {e}")
            out.append(result)
            continue

        if verdict.get("answer_correct"):
            out.append(dict(q, verified=True, verification_notes=verdict.get("notes", "")))
            continue

        try:
            repaired = repair_question(q, context, verdict.get("notes", ""), provider, model)
            recheck = judge_question(repaired, context, provider, judge_model)
        except Exception as e:
            out.append(dict(q, verified=False,
                            verification_notes=f"flagged ({verdict.get('notes', '')}); repair failed: {e}"))
            continue

        if recheck.get("answer_correct"):
            out.append(dict(repaired, verified=True,
                            verification_notes=f"repaired after flag: {verdict.get('notes', '')}"))
        else:
            out.append(dict(repaired, verified=False,
                            verification_notes=f"still flagged after repair: {recheck.get('notes', '')}"))
    return out


def generate_quiz(topic: str, n: int, provider: str, model: str, k: int,
                  temperature: float, retriever: Retriever,
                  qa_links: dict = None, linked_index: dict = None,
                  verify: bool = True, judge_model: str = None) -> dict:
    hits = retriever.search(topic, k=k)
    if not hits:
        raise RuntimeError(f"No chunks retrieved for topic: {topic!r}")

    context, provenance = build_context(hits, qa_links, linked_index)
    user = (f"Topic: {topic}\n\nWrite exactly {n} CAT quiz question(s) grounded in "
            f"these excerpts.\n\n{context}")

    caller = call_groq if provider == "groq" else call_gemini
    raw = caller(SYSTEM_PROMPT, user, model, temperature)

    data = extract_json(raw)
    quiz = Quiz.model_validate(data)
    questions = [q.model_dump() for q in quiz.questions]

    if verify:
        questions = verify_and_repair(questions, context, provider, model, judge_model)

    return {
        "topic": topic,
        "provider": provider,
        "model": model,
        "retrieved_context": provenance,
        "questions": questions,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topic")
    parser.add_argument("--n", type=int, default=3, help="number of questions")
    parser.add_argument("--provider", choices=["groq", "gemini"], default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--k", type=int, default=6, help="chunks to retrieve as context")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--out", default=None, help="write the quiz JSON to this path")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="skip the judge+repair pass (faster, cheaper, less accurate)")
    parser.add_argument("--judge-model", default=None,
                        help="model used to verify/repair; defaults to --model")
    args = parser.parse_args()

    model = args.model or (GROQ_DEFAULT_MODEL if args.provider == "groq" else GEMINI_DEFAULT_MODEL)

    print(f"Loading retriever ...")
    retriever = Retriever()
    qa_links = load_qa_links()
    linked_index = load_linked_chunk_index(qa_links)
    if qa_links:
        print(f"Q/A linkage: {len(qa_links) // 2} pairs available "
              f"({len(linked_index)} distinct linked files indexed)")

    try:
        result = generate_quiz(args.topic, args.n, args.provider, model, args.k,
                               args.temperature, retriever, qa_links, linked_index,
                               verify=args.verify, judge_model=args.judge_model)
    except ValidationError as e:
        print(f"\nModel returned JSON that failed schema validation:\n{e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTopic: {result['topic']}   ({result['provider']}/{result['model']})")
    print("Context retrieved from:")
    for c in result["retrieved_context"]:
        score = f"{c['score']:.3f}" if c["score"] is not None else f"({c['role']})"
        print(f"   {score:<10} {c['source_file']} p{c['page_start']}")

    for i, q in enumerate(result["questions"], 1):
        print(f"\n{'-' * 66}\nQ{i}. {q['question']}")
        for j, opt in enumerate(q["options"]):
            mark = "*" if opt == q["correct_answer"] else " "
            print(f"   {mark} {chr(65 + j)}. {opt}")
        print(f"   shortcut: {q['shortcut_used']}")
        if "verified" in q:
            tag = {True: "verified", False: "UNVERIFIED", None: "verification skipped"}[q["verified"]]
            print(f"   [{tag}] {q.get('verification_notes', '')}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
