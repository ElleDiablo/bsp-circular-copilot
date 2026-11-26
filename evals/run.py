"""Run every question in evals/questions.jsonl through the pipeline and score it.

Usage:
    uv run --env-file .env python -m evals.run     # writes evals/runs/<timestamp>.csv

Scores (True / False, or blank when they don't apply to that question):
- decision_ok:   answered when it should, refused when it should (every question)
- retrieval_hit: an expected circular was among the retrieved passages (answerable questions)
- source_hit:    the answer cited an expected circular (answerable questions it answered)
- fact_hit:      every must_contain item appears in the answer; "a|b" = either (answerable, answered)
Change one thing (prompt, model, top-k, threshold, chunking), run again, compare the CSVs.
"""

import csv  # standard library: write the results as a spreadsheet-friendly file
import json  # standard library: read one JSON object per line from questions.jsonl
import time  # standard library: perf_counter() for latency
from datetime import datetime  # standard library: timestamp for the output file name
from pathlib import Path  # standard library: file paths that work on any OS

# pipeline = the compiled LangGraph graph; we need its full final state (incl. passages), not just answer()
from app.answer import pipeline

HERE = Path(__file__).parent  # the evals/ folder, wherever the project lives
QUESTIONS = HERE / "questions.jsonl"  # one test case per line
RUNS = HERE / "runs"  # one CSV per run, so runs can be compared


def load_cases() -> list[dict]:
    """Read questions.jsonl: one JSON object per line; skip blank lines."""
    return [json.loads(line) for line in QUESTIONS.read_text().splitlines() if line.strip()]


def score(case: dict, state: dict) -> dict:
    """Compare what the pipeline did with what the case expects."""
    expected = set(case.get("expected_circulars", []))  # e.g. {"1242"}; empty for must-refuse cases
    refused = state.get("refused", False)
    result = {"decision_ok": refused == case["must_refuse"],  # right call either way
              "retrieval_hit": None, "source_hit": None, "fact_hit": None}  # None = doesn't apply
    if case["must_refuse"]:
        return result  # refusal cases are scored on the decision only

    retrieved = {p.circular_no for p in state.get("passages", [])}  # what search found
    result["retrieval_hit"] = bool(retrieved & expected)  # & = set intersection: anything in common?
    if refused:
        return result  # it refused, so there's no answer to check sources/facts in

    cited = {c.removeprefix("BSP Circular No. ").split(",")[0] for c in state.get("citations", [])}  # "1242"
    result["source_hit"] = bool(cited & expected)
    answer = state.get("answer", "").lower()  # compare case-insensitively
    must = case.get("must_contain", [])  # e.g. ["36%"] or ["5|five", "banking days"]
    # every item must appear; "|" inside an item = alternatives ("5|five" passes on either)
    result["fact_hit"] = all(any(alt.lower() in answer for alt in m.split("|")) for m in must) if must else None
    return result


def rate(rows: list[dict], key: str) -> str:
    """'7/9 (78%)' over the rows where this score applies."""
    values = [r[key] for r in rows if r[key] is not None]
    if not values:
        return "-"
    return f"{sum(values)}/{len(values)} ({sum(values) / len(values):.0%})"  # True counts as 1


def main() -> None:
    cases = load_cases()
    rows = []
    for case in cases:
        start = time.perf_counter()
        state = pipeline.invoke({"question": case["question"]})  # full final state, incl. passages
        latency_ms = int((time.perf_counter() - start) * 1000)
        tokens = state.get("tokens") or {}  # None when the model wasn't called
        row = {
            "id": case["id"],
            "category": case.get("category", ""),
            **score(case, state),  # ** unpacks the score dict into this one
            "reason": state.get("reason"),
            "cited": "; ".join(state.get("citations", [])),
            "retrieved": "; ".join(f"{p.circular_no} p.{p.page} ({p.score:.2f})" for p in state.get("passages", [])),
            "latency_ms": latency_ms,
            "input_tokens": tokens.get("input"),
            "output_tokens": tokens.get("output"),
            "question": case["question"],
            "answer": state.get("answer", ""),
        }
        rows.append(row)
        print(f"{row['id']:<28} decision_ok={row['decision_ok']!s:<5} reason={row['reason']}")  # progress

    RUNS.mkdir(exist_ok=True)  # create evals/runs/ the first time
    out = RUNS / f"{datetime.now():%Y%m%d-%H%M%S}.csv"  # e.g. 20261126-163000.csv
    with out.open("w", newline="") as f:  # newline="" is what the csv module expects
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())  # column names from the first row
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{len(rows)} questions")
    for key in ("decision_ok", "retrieval_hit", "source_hit", "fact_hit"):
        print(f"  {key:<14} {rate(rows, key)}")
    print(f"  avg latency    {sum(r['latency_ms'] for r in rows) / len(rows) / 1000:.1f} s")
    print(f"  input tokens   {sum(r['input_tokens'] or 0 for r in rows):,}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
