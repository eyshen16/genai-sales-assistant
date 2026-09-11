"""Run a fixed comparison; default is a deterministic-only preview.

Use --live explicitly for three independent model attempts per case. No prompt
tuning or production dispatch occurs here. Review scope mistakes independently
of the lookup's correctness. Missing-region policy differs from the baseline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

from router import route_question
from structured_lookup import PROJECT_ROOT, lookup_compatibility
from tool_calling import INSTRUCTIONS, TOOL, run_compatibility_experiment

CASES_PATH = PROJECT_ROOT / "consulting" / "compatibility_tool_calling_cases.json"
EVALUATION_DATE = date(2026, 9, 10)


def assess(case: dict, actual: dict) -> list[str]:
    failures = []
    if actual["status"] in {"model_failed", "execution_failed", "invalid_tool_call"}:
        failures.append(actual["status"])
    if (actual["proposed_tool"] == "check_compatibility") != case["select_tool"]:
        failures.append("selection_mismatch")
    if (actual["status"] == "executed") != case["execute"]:
        failures.append("execution_mismatch")
    if "arguments" in case and actual["execution_arguments"] != case["arguments"]:
        failures.append("argument_mismatch")
    if "clarification" in case:
        if sorted(actual["missing_information"]) != sorted(case["clarification"]):
            failures.append("clarification_mismatch")
        if case["clarification"] and actual["status"] != "clarification_needed":
            failures.append("clarification_status_mismatch")
    if case["execute"] and actual["status"] == "executed":
        expected = lookup_compatibility(**case["arguments"], as_of_date=EVALUATION_DATE)
        if actual["tool_result"] != expected:
            failures.append("deterministic_result_mismatch")
    return failures


def run_comparison(*, live: bool = False, experiment=run_compatibility_experiment) -> dict:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        baseline = route_question(case["question"])
        row = {"case": case, "deterministic_plan": baseline, "attempts": []}
        if live:
            for repeat in range(1, 4):
                actual = experiment(case["question"], as_of_date=EVALUATION_DATE)
                row["attempts"].append({"repeat": repeat, "actual": actual, "failures": assess(case, actual)})
        signatures = {
            json.dumps({k: a["actual"][k] for k in ("status", "proposed_tool", "execution_arguments", "missing_information", "tool_result")}, sort_keys=True)
            for a in row["attempts"]
        }
        row["variable_behavior"] = len(signatures) > 1 if live else None
        rows.append(row)
    attempts = [a for row in rows for a in row["attempts"]]
    return {
        "mode": "live" if live else "preview_no_model_calls",
        "evaluation_date": EVALUATION_DATE.isoformat(),
        "experiment_fingerprint": hashlib.sha256((INSTRUCTIONS + json.dumps(TOOL, sort_keys=True)).encode()).hexdigest(),
        "dataset_fingerprint": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        "summary": {
            "cases": len(cases), "attempts": len(attempts),
            "passed_attempts": sum(not a["failures"] for a in attempts),
            "failed_attempts": sum(bool(a["failures"]) for a in attempts),
            "variable_cases": sum(row["variable_behavior"] is True for row in rows),
            "input_tokens": sum(a["actual"]["diagnostics"]["input_tokens"] or 0 for a in attempts),
            "output_tokens": sum(a["actual"]["diagnostics"]["output_tokens"] or 0 for a in attempts),
        },
        "results": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make 48 real model requests (16 cases, three repeats).")
    options = parser.parse_args()
    print(json.dumps(run_comparison(live=options.live), indent=2))
