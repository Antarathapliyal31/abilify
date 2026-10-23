"""End-to-end eval runner for the Abilify Q&A graph.

Usage:
    python tests/run_eval.py                  # run all questions
    python tests/run_eval.py --limit 5        # run first 5
    python tests/run_eval.py --id clinical-01 # run a single question by id

Scoring is intentionally simple so it runs offline without extra LLM calls:
  - routing_correct: did the pipeline hit the expected agent?
    (approximated via `previous_agent` in state, or `supervisor_agent` for OOS)
  - keyword_recall: fraction of `expected_keywords` present (case-insensitive)
    in the final answer
  - disclaimer_ok: `must_include_disclaimer` matches whether answer contains
    "consult" + "medical professional"/"doctor"
  - redaction_ok: for PII rows, `redaction_must_remove` tokens are absent
    from the redacted query and `redaction_must_keep` tokens are present

Writes tests/eval_results.json and prints a summary table.
"""
import argparse
import json
import logging
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()


def _keyword_recall(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    lower = answer.lower()
    hits = sum(1 for k in keywords if k.lower() in lower)
    return hits / len(keywords)


def _has_disclaimer(answer: str) -> bool:
    lower = answer.lower()
    return "consult" in lower and ("doctor" in lower or "medical professional" in lower)


def _route_ok(previous_agent: str | None, final_answer: str, expected: str) -> bool:
    if expected == "supervisor_agent":
        return "medical professional" in final_answer.lower() or "consult" in final_answer.lower()
    return previous_agent == expected


def run(questions: list[dict]) -> list[dict]:
    from graph import app
    from redaction import redact

    results = []
    for i, q in enumerate(questions, 1):
        qid = q["id"]
        print(f"[{i}/{len(questions)}] {qid}: {q['question'][:70]}...")
        thread_id = str(uuid.uuid4())
        t0 = time.time()
        try:
            final_state = app.invoke(
                {"query": q["question"]},
                config={"configurable": {"thread_id": thread_id}},
            )
            elapsed = time.time() - t0
            answer = final_state.get("final_answer") or final_state.get("current_answer") or ""
            prev_agent = final_state.get("previous_agent")
            redacted = final_state.get("query", "")

            row = {
                "id": qid,
                "category": q["category"],
                "elapsed_s": round(elapsed, 2),
                "expected_agent": q["expected_agent"],
                "actual_previous_agent": prev_agent,
                "routing_correct": _route_ok(prev_agent, answer, q["expected_agent"]),
                "keyword_recall": round(_keyword_recall(answer, q.get("expected_keywords", [])), 2),
                "disclaimer_ok": (
                    _has_disclaimer(answer) if q.get("must_include_disclaimer", False) else True
                ),
                "redaction_ok": True,
                "answer_preview": answer[:200],
            }

            # PII rows: verify redacted query
            must_remove = q.get("redaction_must_remove", [])
            must_keep = q.get("redaction_must_keep", [])
            if must_remove or must_keep:
                redaction_only = redact(q["question"])
                removed = all(tok not in redaction_only for tok in must_remove)
                kept = all(tok in redaction_only for tok in must_keep)
                row["redaction_ok"] = removed and kept
                row["redacted_query"] = redaction_only

            results.append(row)
            print(
                f"    route={row['routing_correct']} "
                f"kw_recall={row['keyword_recall']:.2f} "
                f"disclaimer={row['disclaimer_ok']} "
                f"redact={row['redaction_ok']} "
                f"({row['elapsed_s']}s)"
            )
        except Exception as e:
            logging.exception("Question %s raised", qid)
            results.append({
                "id": qid,
                "category": q["category"],
                "error": f"{type(e).__name__}: {e}",
                "routing_correct": False,
                "keyword_recall": 0.0,
                "disclaimer_ok": False,
                "redaction_ok": False,
            })
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    def pct(key: str) -> float:
        if not n:
            return 0.0
        return 100 * sum(1 for r in results if r.get(key)) / n

    avg_recall = (
        sum(r.get("keyword_recall", 0.0) for r in results) / n if n else 0.0
    )

    by_cat: dict[str, dict] = {}
    for r in results:
        c = r.get("category", "unknown")
        d = by_cat.setdefault(c, {"n": 0, "route": 0, "kw": 0.0, "disc": 0, "red": 0})
        d["n"] += 1
        d["route"] += int(bool(r.get("routing_correct")))
        d["kw"] += float(r.get("keyword_recall", 0.0))
        d["disc"] += int(bool(r.get("disclaimer_ok")))
        d["red"] += int(bool(r.get("redaction_ok")))

    return {
        "total": n,
        "routing_accuracy_pct": round(pct("routing_correct"), 1),
        "disclaimer_pass_pct": round(pct("disclaimer_ok"), 1),
        "redaction_pass_pct": round(pct("redaction_ok"), 1),
        "avg_keyword_recall": round(avg_recall, 2),
        "by_category": {
            c: {
                "n": d["n"],
                "routing_pct": round(100 * d["route"] / d["n"], 1),
                "disclaimer_pct": round(100 * d["disc"] / d["n"], 1),
                "redaction_pct": round(100 * d["red"] / d["n"], 1),
                "avg_keyword_recall": round(d["kw"] / d["n"], 2),
            }
            for c, d in by_cat.items()
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--id", type=str, default=None, help="Run only this question id")
    ap.add_argument(
        "--questions",
        type=str,
        default=str(ROOT / "tests" / "eval_questions.json"),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(ROOT / "tests" / "eval_results.json"),
    )
    args = ap.parse_args()

    with open(args.questions) as f:
        questions = json.load(f)
    if args.id:
        questions = [q for q in questions if q["id"] == args.id]
    if args.limit:
        questions = questions[: args.limit]

    # Build the retrieval index once (loads from disk if cached)
    from retrieval import initialize_retrieval
    initialize_retrieval()

    results = run(questions)
    summary = summarize(results)

    out = {"summary": summary, "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()
