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
from pathlib import Path
from typing import List, Literal

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retrieve import Retriever

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_DEFAULT_MODEL = "openai/gpt-oss-120b"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


# ----------------------------- schema -----------------------------

class QuizQuestion(BaseModel):
    question: str = Field(min_length=10)
    options: List[str] = Field(min_length=4, max_length=4)
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

Return ONLY a JSON object of this exact shape, with no prose or markdown:
{"questions": [{"question": "...", "options": ["...","...","...","..."], \
"correct_answer": "...", "shortcut_used": "..."}]}"""


def build_context(hits: list, max_chars: int = 9000) -> str:
    parts = []
    used = 0
    for i, h in enumerate(hits, 1):
        loc = h["source_file"].split("/")[-1]
        block = f"[Excerpt {i} | {loc} | p{h['page_start']}]\n{h['text']}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


# ----------------------------- providers -----------------------------

def call_groq(system: str, user: str, model: str, temperature: float) -> str:
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set in the environment.")
    resp = requests.post(
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
    if resp.status_code != 200:
        raise RuntimeError(f"Groq API {resp.status_code}: {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


def call_gemini(system: str, user: str, model: str, temperature: float) -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in the environment.")
    resp = requests.post(
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


def generate_quiz(topic: str, n: int, provider: str, model: str, k: int,
                  temperature: float, retriever: Retriever) -> dict:
    hits = retriever.search(topic, k=k)
    if not hits:
        raise RuntimeError(f"No chunks retrieved for topic: {topic!r}")

    context = build_context(hits)
    user = (f"Topic: {topic}\n\nWrite exactly {n} CAT quiz question(s) grounded in "
            f"these excerpts.\n\n{context}")

    caller = call_groq if provider == "groq" else call_gemini
    raw = caller(SYSTEM_PROMPT, user, model, temperature)

    data = extract_json(raw)
    quiz = Quiz.model_validate(data)

    return {
        "topic": topic,
        "provider": provider,
        "model": model,
        "retrieved_context": [
            {"source_file": h["source_file"], "page_start": h["page_start"],
             "score": round(h["score"], 4)} for h in hits
        ],
        "questions": [q.model_dump() for q in quiz.questions],
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
    args = parser.parse_args()

    model = args.model or (GROQ_DEFAULT_MODEL if args.provider == "groq" else GEMINI_DEFAULT_MODEL)

    print(f"Loading retriever ...")
    retriever = Retriever()

    try:
        result = generate_quiz(args.topic, args.n, args.provider, model,
                               args.k, args.temperature, retriever)
    except ValidationError as e:
        print(f"\nModel returned JSON that failed schema validation:\n{e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"\n{e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTopic: {result['topic']}   ({result['provider']}/{result['model']})")
    print("Context retrieved from:")
    for c in result["retrieved_context"]:
        print(f"   {c['score']:.3f}  {c['source_file']} p{c['page_start']}")

    for i, q in enumerate(result["questions"], 1):
        print(f"\n{'-' * 66}\nQ{i}. {q['question']}")
        for j, opt in enumerate(q["options"]):
            mark = "*" if opt == q["correct_answer"] else " "
            print(f"   {mark} {chr(65 + j)}. {opt}")
        print(f"   shortcut: {q['shortcut_used']}")

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
