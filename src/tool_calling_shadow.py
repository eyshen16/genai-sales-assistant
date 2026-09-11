"""Offline coexistence gate for the bounded compatibility selector.

This module is deliberately not imported by the production API, router, or
orchestrator. It evaluates a route plan without changing that plan or response.
"""
from __future__ import annotations

import json
import hashlib
import re
import statistics
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable

from router import normalize_text, route_question
from structured_lookup import PROJECT_ROOT, lookup_compatibility
from tool_calling import INSTRUCTIONS, TOOL, explicit_region_codes, run_compatibility_experiment

CASES_PATH = PROJECT_ROOT / "consulting" / "compatibility_tool_calling_shadow_cases.json"
EVALUATION_DATE = date(2026, 9, 10)
EXPERIMENT_VERSION = "tool-calling-shadow-v2-region-code-boundary"
VALIDATOR_CHANGE = "Two-letter region codes require uppercase token matches; country names remain case-insensitive."

RELATION_PATTERNS = (
    r"\bpair(?:ed)? with\b", r"\bvalid pair\b", r"\busable pair\b",
    r"\boperate together\b", r"\bused together\b", r"\buse together\b",
    r"\bcombine(?:d)?\b", r"\bcoupl(?:e|ed) with\b",
)
GOVERNANCE_PATTERNS = (
    r"\bwarranty\b", r"\bcoverage\b", r"\bcertif(?:y|ication)\b",
    r"\bguarantee(?:d)?\b", r"\bundocumented\b", r"\bunsupported\b",
    r"\bthird[ -]party\b", r"\bignore previous instructions\b",
)
RAG_SCOPE_PATTERNS = (
    r"\bpv[ -]surplus\b", r"\bcharging\b", r"\bcommission(?:ing)?\b",
    r"\bretrofit\b", r"\binstallation\b", r"\bwiring\b", r"\bmeter(?:ing| data)?\b",
)


def _unique(pattern: str, text: str) -> list[str]:
    return sorted(set(re.findall(pattern, text)))


def _regions(text: str) -> set[str]:
    return explicit_region_codes(text)


def _compatibility_shape(text: str, inverters: list[str], batteries: list[str]) -> bool:
    relational = any(re.search(pattern, text) for pattern in RELATION_PATTERNS)
    explicit = any(word in text for word in ("compatible", "compatibility", "work with", "works with"))
    return len(inverters) == 1 and len(batteries) == 1 and (relational or explicit)


def decide_shadow_gate(question: str, route_plan: dict[str, Any]) -> dict[str, Any]:
    """Return a deterministic shadow decision without executing any route."""
    text = normalize_text(question)
    inverters = _unique(r"\bve hybrid\s+\d+\b", text)
    batteries = _unique(r"\bhomecell\s+\d+\b", text)
    versions = _unique(r"(?<![\w.])\d+\.\d+(?:\.\d+)*(?![\w.])", text)
    regions = _regions(question)
    governance_scope = any(re.search(pattern, text) for pattern in GOVERNANCE_PATTERNS)
    rag_scope = any(re.search(pattern, text) for pattern in RAG_SCOPE_PATTERNS)
    compatibility_shape = _compatibility_shape(text, inverters, batteries)
    multiple_combinations = len(inverters) > 1 or len(batteries) > 1

    details = {
        "inverter_mentions": inverters,
        "battery_mentions": batteries,
        "firmware_mentions": versions,
        "region_mentions": sorted(regions),
        "anomalies": [],
    }

    if route_plan["route"] == "needs_review" and governance_scope:
        return _gate("fail_closed_governance", False, "Existing governance-sensitive needs-review decision remains closed to tool execution.", details)
    if route_plan["route"] == "composite":
        return _gate("preserve_composite", False, "Existing composite decomposition remains authoritative.", details)
    if governance_scope and (compatibility_shape or multiple_combinations):
        return _gate("fail_closed_governance", False, "Compatibility-shaped request contains governance-sensitive or adversarial scope.", details)
    if route_plan["route"] == "rag":
        return _gate("preserve_rag", False, "Existing RAG route is outside the bounded compatibility selector.", details)
    if multiple_combinations:
        return _gate("fail_closed_governance", False, "Multiple product combinations cannot use the single-combination tool.", details)

    if route_plan["route"] == "structured_lookup":
        execution = route_plan.get("execution") or {}
        anomalies: list[str] = []
        if len(versions) > 1:
            anomalies.append("multiple_firmware_values")
        elif len(versions) == 1 and execution.get("firmware_version") != versions[0]:
            anomalies.append("firmware_provenance_mismatch")
        if len(regions) > 1:
            anomalies.append("multiple_regions")
        elif not regions and execution.get("region") is not None:
            anomalies.append("region_not_explicit")
        elif len(regions) == 1 and execution.get("region") not in regions:
            anomalies.append("region_provenance_mismatch")
        if len(inverters) == 1 and execution.get("inverter_model", "").casefold() != inverters[0]:
            anomalies.append("inverter_provenance_mismatch")
        if len(batteries) == 1 and execution.get("battery_model", "").casefold() != batteries[0]:
            anomalies.append("battery_provenance_mismatch")
        details["anomalies"] = anomalies
        if anomalies:
            return _gate("semantic_selection", True, "Structured route has parameter provenance or ambiguity anomalies.", details)
        return _gate("preserve_structured", False, "Structured route and explicit parameter provenance are consistent.", details)

    if route_plan["route"] == "needs_review" and compatibility_shape and not rag_scope:
        return _gate("semantic_selection", True, "One-combination compatibility shape was not resolved by deterministic routing.", details)
    return _gate("preserve_needs_review", False, "No safe bounded compatibility fallback candidate was identified.", details)


def _gate(action: str, invoke: bool, reason: str, details: dict[str, Any]) -> dict[str, Any]:
    return {"action": action, "invoke_semantic_selector": invoke, "reason": reason, **details}


def evaluate_case(
    case: dict[str, Any], *,
    experiment: Callable[..., dict[str, Any]] = run_compatibility_experiment,
    repeats: int = 3,
    on_attempt: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    plan = route_question(case["question"])
    gate = decide_shadow_gate(case["question"], plan)
    row = {"case": case, "deterministic_plan": plan, "gate": gate, "attempts": []}
    if gate["invoke_semantic_selector"]:
        for repeat in range(1, repeats + 1):
            actual = experiment(case["question"], as_of_date=EVALUATION_DATE)
            attempt = _attempt(case, repeat, actual)
            row["attempts"].append(attempt)
            if on_attempt:
                on_attempt(case, plan, gate, attempt)
    elif gate["action"] == "preserve_structured":
        execution = plan.get("execution") or {}
        missing = list(execution.get("missing_information", []))
        lookup_result = None
        execution_count = 0
        if not missing:
            lookup_result = lookup_compatibility(
                inverter_model=execution.get("inverter_model"), battery_model=execution.get("battery_model"),
                firmware_version=execution.get("firmware_version"), region=execution.get("region"),
                as_of_date=EVALUATION_DATE,
            )
            execution_count = 1
        attempt = {
            "repeat": 1, "selector_invoked": False, "status": "deterministic_fast_path",
            "proposed_tool": None, "proposed_arguments": None,
            "execution_arguments": execution, "missing_information": missing,
            "validation_status": "not_applicable", "validation_issues": [],
            "execution_count": execution_count, "lookup_result": lookup_result,
            "latency_ms": 0, "input_tokens": 0, "output_tokens": 0,
            "model_failure_category": None, "failures": _assess(case, gate, plan, execution, missing, execution_count, lookup_result),
        }
        row["attempts"].append(attempt)
        if on_attempt:
            on_attempt(case, plan, gate, attempt)
    else:
        attempt = {
            "repeat": 1, "selector_invoked": False, "status": gate["action"],
            "proposed_tool": None, "proposed_arguments": None, "execution_arguments": None,
            "missing_information": [], "validation_status": "not_applicable", "validation_issues": [],
            "execution_count": 0, "lookup_result": None, "latency_ms": 0,
            "input_tokens": 0, "output_tokens": 0, "model_failure_category": None,
            "failures": _assess(case, gate, plan, None, [], 0, None),
        }
        row["attempts"].append(attempt)
        if on_attempt:
            on_attempt(case, plan, gate, attempt)
    signatures = {
        json.dumps({key: attempt[key] for key in ("status", "execution_arguments", "missing_information", "lookup_result")}, sort_keys=True)
        for attempt in row["attempts"]
    }
    row["variable_behavior"] = len(signatures) > 1
    return row


def _attempt(case: dict[str, Any], repeat: int, actual: dict[str, Any]) -> dict[str, Any]:
    status = actual["status"]
    validation_status = "validated" if status in {"executed", "clarification_needed", "no_tool_selected"} else "blocked_or_failed"
    failure_category = status if status in {"model_failed", "execution_failed", "invalid_tool_call"} else None
    execution_count = 1 if status == "executed" else 0
    return {
        "repeat": repeat, "selector_invoked": True, "status": status,
        "proposed_tool": actual["proposed_tool"], "proposed_arguments": actual["proposed_arguments"],
        "execution_arguments": actual["execution_arguments"], "missing_information": actual["missing_information"],
        "validation_status": validation_status, "validation_issues": actual["validation_issues"],
        "execution_count": execution_count, "lookup_result": actual["tool_result"],
        "latency_ms": actual["diagnostics"]["model_latency_ms"],
        "input_tokens": actual["diagnostics"]["input_tokens"],
        "output_tokens": actual["diagnostics"]["output_tokens"],
        "model_failure_category": failure_category,
        "failures": _assess(case, {"action": "semantic_selection"}, None, actual["execution_arguments"], actual["missing_information"], execution_count, actual["tool_result"], actual),
    }


def _assess(
    case: dict[str, Any], gate: dict[str, Any], plan: dict[str, Any] | None,
    execution_arguments: dict[str, Any] | None, missing: list[str], execution_count: int,
    lookup_result: dict[str, Any] | None, actual: dict[str, Any] | None = None,
) -> list[str]:
    failures: list[str] = []
    if gate["action"] != case["expected_gate_action"]:
        failures.append("gate_action_mismatch")
    if actual is not None:
        selected = actual["proposed_tool"] == "check_compatibility"
        if selected != case["select_tool"]:
            failures.append("selection_mismatch")
        if actual["status"] in {"model_failed", "execution_failed", "invalid_tool_call", "scope_rejected"}:
            failures.append(actual["status"])
    if execution_count != int(case["execute"]):
        failures.append("execution_mismatch")
    comparable_arguments = None
    if isinstance(execution_arguments, dict):
        comparable_arguments = {key: execution_arguments.get(key) for key in case.get("arguments", {})}
    if "arguments" in case and comparable_arguments != case["arguments"]:
        failures.append("argument_mismatch")
    if "clarification" in case and sorted(missing) != sorted(case["clarification"]):
        failures.append("clarification_mismatch")
    if execution_count:
        expected = lookup_compatibility(**case["arguments"], as_of_date=EVALUATION_DATE)
        if lookup_result != expected:
            failures.append("deterministic_result_mismatch")
    return failures


def run_shadow_evaluation(
    *, live: bool = False, output_path: Path | str | None = None,
    experiment: Callable[..., dict[str, Any]] = run_compatibility_experiment,
) -> dict[str, Any]:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    fingerprint = _experiment_fingerprint()
    rows: list[dict[str, Any]] = []
    output = Path(output_path) if output_path else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("", encoding="utf-8")

    def persist(case: dict[str, Any], plan: dict[str, Any], gate: dict[str, Any], attempt: dict[str, Any]) -> None:
        if not output:
            return
        with output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "experiment_version": EXPERIMENT_VERSION, "experiment_fingerprint": fingerprint,
                "case_id": case["case_id"], "question": case["question"],
                "deterministic_route": plan["route"], "deterministic_domains": plan["domains"],
                "gate": gate, "attempt": attempt,
            }) + "\n")
            stream.flush()

    for case in cases:
        row = evaluate_case(case, experiment=experiment, repeats=3 if live else 0, on_attempt=persist)
        if not live and row["gate"]["invoke_semantic_selector"]:
            row["attempts"] = []
            row["variable_behavior"] = None
        rows.append(row)
    return _report(rows, live)


def _experiment_fingerprint() -> str:
    payload = {
        "version": EXPERIMENT_VERSION, "validator_change": VALIDATOR_CHANGE,
        "instructions": INSTRUCTIONS, "tool": TOOL,
        "relation_patterns": RELATION_PATTERNS, "governance_patterns": GOVERNANCE_PATTERNS,
        "rag_scope_patterns": RAG_SCOPE_PATTERNS,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _report(rows: list[dict[str, Any]], live: bool) -> dict[str, Any]:
    attempts = [attempt for row in rows for attempt in row["attempts"]]
    model_attempts = [attempt for attempt in attempts if attempt["selector_invoked"]]
    latencies = [attempt["latency_ms"] for attempt in model_attempts]
    indirect = [attempt for row in rows if row["case"]["category"] == "indirect" for attempt in row["attempts"]]
    all_failures = [failure for attempt in attempts for failure in attempt["failures"]]
    sorted_latency = sorted(latencies)
    p95 = sorted_latency[max(0, int(len(sorted_latency) * .95 + .9999) - 1)] if sorted_latency else None
    criteria = {
        "gate_expectations_pass": not any("gate_action_mismatch" in attempt["failures"] for attempt in attempts),
        "zero_forbidden_executions": not any(row["case"]["execute"] is False and attempt["execution_count"] for row in rows for attempt in row["attempts"]),
        "zero_argument_or_result_mismatches": not any(failure in {"argument_mismatch", "deterministic_result_mismatch"} for failure in all_failures),
        "indirect_attempt_success_at_least_95_percent": bool(indirect) and sum(not attempt["failures"] for attempt in indirect) / len(indirect) >= .95,
        "each_indirect_case_succeeds_at_least_2_of_3": all(sum(not attempt["failures"] for attempt in row["attempts"]) >= 2 for row in rows if row["case"]["category"] == "indirect") if live else False,
        "model_failures_at_most_one": sum(attempt["model_failure_category"] is not None for attempt in model_attempts) <= 1,
        "median_latency_at_most_6000_ms": bool(latencies) and statistics.median(latencies) <= 6000,
        "p95_latency_at_most_12000_ms": p95 is not None and p95 <= 12000,
    }
    return {
        "mode": "live" if live else "preview_no_model_calls", "evaluation_date": EVALUATION_DATE.isoformat(),
        "experiment_version": EXPERIMENT_VERSION,
        "experiment_fingerprint": _experiment_fingerprint(),
        "dataset_fingerprint": hashlib.sha256(CASES_PATH.read_bytes()).hexdigest(),
        "validator_change": VALIDATOR_CHANGE,
        "summary": {
            "cases": len(rows), "gated_cases": sum(row["gate"]["invoke_semantic_selector"] for row in rows),
            "model_attempts": len(model_attempts), "passed_attempts": sum(not attempt["failures"] for attempt in attempts),
            "failed_attempts": sum(bool(attempt["failures"]) for attempt in attempts),
            "variable_cases": sum(row["variable_behavior"] is True for row in rows),
            "input_tokens": sum(attempt["input_tokens"] or 0 for attempt in model_attempts),
            "output_tokens": sum(attempt["output_tokens"] or 0 for attempt in model_attempts),
            "median_latency_ms": statistics.median(latencies) if latencies else None, "p95_latency_ms": p95,
        },
        "acceptance_criteria": criteria, "all_acceptance_criteria_passed": live and all(criteria.values()),
        "results": rows,
    }
