"""
Step 4c: Systematic generation-quality eval.

Retrieval is measured by eval_retrieval.py (hit@k). This measures the other
half: given the retrieved context was fine, is the generated QUIZ actually
good? Two layers:

  1. Automated structural checks (cheap, deterministic):
     - did generation succeed schema validation at all (failure rate)?
     - are the 4 options distinct, non-trivial strings?
     - lexical grounding: does shortcut_used / the question share real
       vocabulary with the retrieved context, or does it look invented?

  2. LLM-as-judge pass (a second, independent call): given the ORIGINAL
     context plus the generated question, asks the model to judge whether
     correct_answer is actually correct and whether shortcut_used names a
     technique that genuinely appears in the context rather than a
     plausible-sounding invention. This is the automatable stand-in for the
     "manual spot-check on generated quiz accuracy" step from the original
     plan -- it is a judge model checking another model's work, not ground
     truth, and is reported as such.

Usage:
    python eval_generation.py --topics-per-section 4 --n-per-topic 1
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from retrieve import Retriever
from link_qa import load_links as load_qa_links
from generate import (
    generate_quiz, load_linked_chunk_index, call_groq, extract_json,
    GROQ_DEFAULT_MODEL,
)

OUT_PATH = Path(__file__).resolve().parent.parent / "eval" / "generation_results.json"

# Deliberately different from eval_set.json's questions (that set tests
# retrieval; these exercise generation across the same section spread).
TOPICS = {
    "QA": [
        "trailing zeros in factorial shortcut",
        "successive percentage change shortcut",
        "alligation and mixtures shortcut",
        "time and work negative work concept",
    ],
    "DILR": [
        "games and tournaments points table approach",
        "binary logic true false statements",
        "venn diagram set theory shortcut",
        "data sufficiency approach",
    ],
    "VARC": [
        "para jumble opening sentence strategy",
        "critical reasoning assumption identification",
        "grammar subject verb agreement rule",
        "word analogy bridge method",
    ],
}

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


def word_set(text: str) -> set:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


STOPWORDS = {"that", "this", "with", "from", "have", "will", "shall", "which",
             "there", "their", "about", "into", "such", "than", "then", "when"}


def grounding_score(question_obj: dict, context: str) -> float:
    """Rough lexical-overlap heuristic: fraction of distinctive words in
    shortcut_used that also appear somewhere in the retrieved context. Not a
    substitute for the LLM judge, but free and catches obvious invention."""
    shortcut_words = word_set(question_obj["shortcut_used"]) - STOPWORDS
    if not shortcut_words:
        return 0.0
    context_words = word_set(context)
    overlap = shortcut_words & context_words
    return len(overlap) / len(shortcut_words)


def judge_question(question_obj: dict, context: str, model: str) -> dict:
    user = (
        f"Source excerpts:\n{context}\n\n"
        f"Generated question: {question_obj['question']}\n"
        f"Options: {question_obj['options']}\n"
        f"Claimed correct_answer: {question_obj['correct_answer']}\n"
        f"Claimed shortcut_used: {question_obj['shortcut_used']}\n"
    )
    raw = call_groq(JUDGE_SYSTEM_PROMPT, user, model, temperature=0.0)
    return extract_json(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topics-per-section", type=int, default=4)
    parser.add_argument("--n-per-topic", type=int, default=1)
    parser.add_argument("--provider", choices=["groq", "gemini"], default="groq")
    parser.add_argument("--model", default=None)
    parser.add_argument("--judge-model", default=None,
                        help="defaults to the same model as --model")
    args = parser.parse_args()

    model = args.model or GROQ_DEFAULT_MODEL
    judge_model = args.judge_model or model

    print("Loading retriever ...")
    retriever = Retriever()
    qa_links = load_qa_links()
    linked_index = load_linked_chunk_index(qa_links)

    results = []
    n_generation_failures = 0
    n_attempts = 0

    for section, topics in TOPICS.items():
        for topic in topics[: args.topics_per_section]:
            n_attempts += 1
            print(f"\n[{section}] {topic}")
            try:
                out = generate_quiz(topic, args.n_per_topic, args.provider, model,
                                    k=6, temperature=0.4, retriever=retriever,
                                    qa_links=qa_links, linked_index=linked_index)
            except Exception as e:
                n_generation_failures += 1
                print(f"  GENERATION FAILED: {type(e).__name__}: {e}")
                results.append({"section": section, "topic": topic, "status": "generation_failed",
                                "error": str(e)})
                continue

            context_text = " ".join(c["source_file"] for c in out["retrieved_context"])
            # Re-derive the actual excerpt text for the judge from retrieval,
            # since generate_quiz doesn't return the raw context string.
            hits = retriever.search(topic, k=6)
            context_for_judge = "\n\n".join(h["text"] for h in hits)

            for q in out["questions"]:
                score = grounding_score(q, context_for_judge)
                distinct_options = len(set(q["options"])) == len(q["options"])

                judge = None
                try:
                    judge = judge_question(q, context_for_judge, judge_model)
                except Exception as e:
                    print(f"  judge call failed: {type(e).__name__}: {e}")

                rec = {
                    "section": section,
                    "topic": topic,
                    "status": "ok",
                    "question": q["question"],
                    "correct_answer": q["correct_answer"],
                    "shortcut_used": q["shortcut_used"],
                    "distinct_options": distinct_options,
                    "lexical_grounding_score": round(score, 2),
                    "judge": judge,
                }
                results.append(rec)

                flag = "OK" if (judge and judge.get("answer_correct") and judge.get("shortcut_grounded")) else "FLAG"
                print(f"  [{flag}] {q['question'][:70]}...")
                print(f"        distinct_options={distinct_options} lexical_grounding={score:.2f} "
                      f"judge={judge}")

            # Free-tier TPM is tight (8k/min on the model this was built against)
            # and a generation+judge pair alone can use ~7-8k tokens. The retry
            # wrapper in generate.py's call_groq handles occasional overshoot,
            # but pacing proactively here means most topics never need it.
            time.sleep(12)

    ok_records = [r for r in results if r["status"] == "ok"]
    n_answer_correct = sum(1 for r in ok_records if r["judge"] and r["judge"].get("answer_correct"))
    n_shortcut_grounded = sum(1 for r in ok_records if r["judge"] and r["judge"].get("shortcut_grounded"))
    n_distinct = sum(1 for r in ok_records if r["distinct_options"])
    avg_lexical = sum(r["lexical_grounding_score"] for r in ok_records) / len(ok_records) if ok_records else 0

    summary = {
        "n_generation_attempts": n_attempts,
        "n_generation_failures": n_generation_failures,
        "n_questions_generated": len(ok_records),
        "judge_answer_correct_rate": round(n_answer_correct / len(ok_records), 3) if ok_records else None,
        "judge_shortcut_grounded_rate": round(n_shortcut_grounded / len(ok_records), 3) if ok_records else None,
        "distinct_options_rate": round(n_distinct / len(ok_records), 3) if ok_records else None,
        "avg_lexical_grounding_score": round(avg_lexical, 3),
        "model": model,
        "judge_model": judge_model,
        "records": results,
    }

    print(f"\n{'=' * 62}")
    print(f"Generation attempts: {n_attempts}  (failures: {n_generation_failures})")
    print(f"Questions generated: {len(ok_records)}")
    print(f"Judge: answer correct     {summary['judge_answer_correct_rate']}")
    print(f"Judge: shortcut grounded  {summary['judge_shortcut_grounded_rate']}")
    print(f"Distinct options rate     {summary['distinct_options_rate']}")
    print(f"Avg lexical grounding     {summary['avg_lexical_grounding_score']}")

    OUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
