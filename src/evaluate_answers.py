from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable

from orchestrator import orchestrate_question
from retrieve import ingest_and_build_index
from structured_lookup import PROJECT_ROOT


DEFAULT_ANSWER_EVALUATION_CASES_PATH = PROJECT_ROOT / "consulting" / "answer_evaluation_cases.json"


@dataclass(frozen=True)
class AnswerEvaluationCase:
    case_id: str
    question: str
    category: str
    expected_route: str
    expected_status: str
    expected_answer_completeness: str | None
    required_domains: list[str]
    required_authoritative_source_ids: list[str] = field(default_factory=list)
    required_subroutes: list[dict[str, Any]] = field(default_factory=list)
    required_missing_information: list[str] = field(default_factory=list)
    required_unresolved_dependencies: list[dict[str, Any]] = field(default_factory=list)
    required_answer_concepts: list[str] = field(default_factory=list)
    required_evidence_concepts: list[str] = field(default_factory=list)
    forbidden_claims: list[str] = field(default_factory=list)
    forbidden_authority_claims: list[dict[str, Any]] = field(default_factory=list)
    requires_semantic_review: bool = False
    notes: str | None = None


def load_answer_evaluation_cases(
    path: Path | str = DEFAULT_ANSWER_EVALUATION_CASES_PATH,
) -> list[AnswerEvaluationCase]:
    raw_cases = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        AnswerEvaluationCase(
            case_id=case["case_id"],
            question=case["question"],
            category=case["category"],
            expected_route=case["expected_route"],
            expected_status=case["expected_status"],
            expected_answer_completeness=case.get("expected_answer_completeness"),
            required_domains=case.get("required_domains", []),
            required_authoritative_source_ids=case.get("required_authoritative_source_ids", []),
            required_subroutes=case.get("required_subroutes", []),
            required_missing_information=case.get("required_missing_information", []),
            required_unresolved_dependencies=case.get("required_unresolved_dependencies", []),
            required_answer_concepts=case.get("required_answer_concepts", []),
            required_evidence_concepts=case.get("required_evidence_concepts", []),
            forbidden_claims=case.get("forbidden_claims", []),
            forbidden_authority_claims=case.get("forbidden_authority_claims", []),
            requires_semantic_review=case.get("requires_semantic_review", False),
            notes=case.get("notes"),
        )
        for case in raw_cases
    ]


def run_answer_evaluation(
    *,
    project_root: Path | str = PROJECT_ROOT,
    as_of_date: date | None = None,
    cases_path: Path | str = DEFAULT_ANSWER_EVALUATION_CASES_PATH,
    runner: Callable[[str], dict[str, Any]] | None = None,
    live_run: bool = False,
) -> dict[str, Any]:
    root = Path(project_root)
    cases = load_answer_evaluation_cases(cases_path)

    if live_run:
        retrieval_index = ingest_and_build_index(project_root=root, as_of_date=as_of_date)

        def live_runner(question: str) -> dict[str, Any]:
            return orchestrate_question(
                question,
                project_root=root,
                as_of_date=as_of_date,
                retrieval_index=retrieval_index,
            )

        execute = live_runner
    elif runner is not None:
        execute = runner
    else:
        raise ValueError("Pass runner=... for test mode or live_run=True for real evaluation.")

    case_results = [evaluate_answer_case(case=case, actual_result=execute(case.question)) for case in cases]
    return {
        "cases": case_results,
        "metrics": compute_answer_evaluation_metrics(case_results),
    }


def evaluate_answer_case(
    *,
    case: AnswerEvaluationCase,
    actual_result: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    failure_taxonomy: list[str] = []
    semantic_review_reasons: list[str] = []

    deterministic_checks = {
        "route_correct": actual_result["route"] == case.expected_route,
        "status_correct": actual_result["status"] == case.expected_status,
        "domains_correct": set(case.required_domains).issubset(set(actual_result.get("domains", []))),
        "missing_information_correct": set(case.required_missing_information).issubset(
            set(actual_result.get("missing_information", []))
        ),
    }
    if not deterministic_checks["route_correct"]:
        failures.append("Expected route did not match actual route.")
        failure_taxonomy.append("routing_failure")
    if not deterministic_checks["status_correct"]:
        failures.append("Expected status did not match actual status.")
        failure_taxonomy.append("orchestration_status_failure")
    if not deterministic_checks["domains_correct"]:
        failures.append("Required domains were not all present.")
        failure_taxonomy.append("routing_failure")
    if case.required_missing_information and not deterministic_checks["missing_information_correct"]:
        failures.append("Required clarification information was missing.")
        failure_taxonomy.append("missing_clarification_behavior")

    actual_top_level_completeness = _extract_top_level_answer_completeness(actual_result)
    completeness_correct = (
        True
        if case.expected_answer_completeness is None
        else actual_top_level_completeness == case.expected_answer_completeness
    )
    deterministic_checks["completeness_correct"] = completeness_correct
    if not completeness_correct:
        failures.append("Expected answer completeness did not match actual completeness.")
        failure_taxonomy.append("completeness_misclassification")

    subroute_diagnostics = evaluate_required_subroutes(case.required_subroutes, actual_result)
    if not subroute_diagnostics["all_required_subroutes_present"]:
        failures.append("One or more required subroutes were missing.")
        failure_taxonomy.append("routing_failure")
    if subroute_diagnostics["status_failures"]:
        failures.append("One or more subroutes had an unexpected status or route.")
        failure_taxonomy.append("orchestration_status_failure")
    if subroute_diagnostics["completeness_failures"]:
        failures.append("One or more subroutes had unexpected completeness.")
        failure_taxonomy.append("completeness_misclassification")
    if subroute_diagnostics["missing_information_failures"]:
        failures.append("One or more subroutes were missing required clarification information.")
        failure_taxonomy.append("missing_clarification_behavior")

    authority_diagnostics = evaluate_authority_requirements(case, actual_result)
    if authority_diagnostics["required_authority_failure"]:
        failures.append("Required authoritative source IDs were not satisfied.")
        failure_taxonomy.append("authority_boundary_violation")
    if authority_diagnostics["forbidden_authority_failures"]:
        failures.append("Forbidden authority assignment was detected.")
        failure_taxonomy.append("authority_boundary_violation")

    unresolved_dependency_diagnostics = evaluate_unresolved_dependencies(case, actual_result)
    if unresolved_dependency_diagnostics["missing_required_dependencies"]:
        failures.append("Required unresolved dependencies were missing or mismatched.")
        failure_taxonomy.append("unresolved_dependency_error")
    if unresolved_dependency_diagnostics["semantic_review_dependencies"]:
        semantic_review_reasons.append("Some unresolved dependency matches require semantic review.")

    citation_integrity = evaluate_citation_integrity(actual_result)
    if not citation_integrity["valid"]:
        failures.append("Citation integrity validation failed.")
        failure_taxonomy.append("citation_integrity_failure")
    validation_failure_diagnostics = evaluate_validation_failures(actual_result)
    if validation_failure_diagnostics["has_non_citation_failure"]:
        failures.append("Generation or validation failed for a non-citation reason.")
        failure_taxonomy.append("generation_validation_failure")

    forbidden_claims = detect_forbidden_claims(case, actual_result)
    if forbidden_claims:
        failures.append("Forbidden claim detected in answer output.")
        failure_taxonomy.append("unsupported_claim_failure")
        failure_taxonomy.append("unsafe_overanswering")

    inappropriate_answer = detect_inappropriate_answer(case, actual_result)
    if inappropriate_answer:
        failures.append("Substantive answer was provided when the specification forbids it.")
        failure_taxonomy.append("inappropriate_abstention")
        failure_taxonomy.append("unsafe_overanswering")

    answer_concepts = evaluate_concepts(
        concepts=case.required_answer_concepts,
        text=_collect_answer_text(actual_result),
    )
    evidence_concepts = evaluate_concepts(
        concepts=case.required_evidence_concepts,
        text=_collect_retrieved_evidence_text(actual_result),
    )
    subroute_concept_diagnostics = evaluate_subroute_concepts(case.required_subroutes, actual_result)
    if answer_concepts["semantic_review_concepts"]:
        semantic_review_reasons.append("Some required answer concepts were not explicitly matched.")
    if evidence_concepts["semantic_review_concepts"]:
        semantic_review_reasons.append("Some required evidence concepts were not explicitly matched.")
    if subroute_concept_diagnostics["semantic_review_domains"]:
        semantic_review_reasons.append("Some required composite subroute concepts were not explicitly matched.")
    if subroute_concept_diagnostics["forbidden_claim_domains"]:
        failures.append("Forbidden claim detected inside a required subroute answer.")
        failure_taxonomy.append("unsupported_claim_failure")
        failure_taxonomy.append("unsafe_overanswering")
    if case.requires_semantic_review:
        semantic_review_reasons.append("Case is marked as requiring semantic review.")

    classification = classify_answer_case(
        failures=failures,
        semantic_review_reasons=semantic_review_reasons,
    )
    if classification == "needs_semantic_review":
        failure_taxonomy.append("evaluator_uncertainty")

    return {
        "case_id": case.case_id,
        "question": case.question,
        "expected_route": case.expected_route,
        "actual_route": actual_result["route"],
        "expected_status": case.expected_status,
        "actual_status": actual_result["status"],
        "expected_answer_completeness": case.expected_answer_completeness,
        "actual_answer_completeness": actual_top_level_completeness,
        "deterministic_checks": deterministic_checks,
        "subroute_diagnostics": subroute_diagnostics,
        "authority_diagnostics": authority_diagnostics,
        "unresolved_dependency_diagnostics": unresolved_dependency_diagnostics,
        "citation_integrity": citation_integrity,
        "validation_failure_diagnostics": validation_failure_diagnostics,
        "answer_concepts": answer_concepts,
        "evidence_concepts": evidence_concepts,
        "subroute_concept_diagnostics": subroute_concept_diagnostics,
        "forbidden_claims_detected": forbidden_claims,
        "inappropriate_answer_detected": inappropriate_answer,
        "semantic_review_reasons": semantic_review_reasons,
        "evaluation_classification": classification,
        "failure_taxonomy": sorted(set(failure_taxonomy)),
        "actual_result": _compact_actual_result(actual_result),
    }


def evaluate_required_subroutes(
    required_subroutes: list[dict[str, Any]],
    actual_result: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = {
        "all_required_subroutes_present": True,
        "status_failures": [],
        "completeness_failures": [],
        "missing_information_failures": [],
    }
    actual_subroutes = actual_result.get("subresults", [])
    by_domain = {subroute.get("domain"): subroute for subroute in actual_subroutes}
    for expected in required_subroutes:
        actual = by_domain.get(expected["domain"])
        if actual is None:
            diagnostics["all_required_subroutes_present"] = False
            continue
        if actual["route"] != expected["expected_route"] or actual["status"] != expected["expected_status"]:
            diagnostics["status_failures"].append(expected["domain"])
        expected_completeness = expected.get("expected_answer_completeness")
        if expected_completeness is not None:
            actual_completeness = actual["result"].get("answer_completeness")
            if actual_completeness != expected_completeness:
                diagnostics["completeness_failures"].append(expected["domain"])
        expected_missing = set(expected.get("required_missing_information", []))
        if expected_missing and not expected_missing.issubset(set(actual.get("missing_information", []))):
            diagnostics["missing_information_failures"].append(expected["domain"])
    return diagnostics


def evaluate_subroute_concepts(
    required_subroutes: list[dict[str, Any]],
    actual_result: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = {
        "answer_concept_matches": {},
        "evidence_concept_matches": {},
        "semantic_review_domains": [],
        "forbidden_claim_domains": [],
    }
    actual_subroutes = {
        subroute.get("domain"): subroute
        for subroute in actual_result.get("subresults", [])
    }

    for expected in required_subroutes:
        actual = actual_subroutes.get(expected["domain"])
        if actual is None:
            continue

        answer_concepts = evaluate_concepts(
            concepts=expected.get("required_answer_concepts", []),
            text=actual.get("result", {}).get("answer", ""),
        )
        evidence_concepts = evaluate_concepts(
            concepts=expected.get("required_evidence_concepts", []),
            text="\n".join(
                chunk.get("text", "")
                for chunk in actual.get("result", {}).get("retrieved_chunks", [])
            ),
        )
        diagnostics["answer_concept_matches"][expected["domain"]] = answer_concepts
        diagnostics["evidence_concept_matches"][expected["domain"]] = evidence_concepts

        if answer_concepts["semantic_review_concepts"] or evidence_concepts["semantic_review_concepts"]:
            diagnostics["semantic_review_domains"].append(expected["domain"])

        answer_text = _normalize_text(actual.get("result", {}).get("answer", ""))
        if any(
            _normalize_text(claim) in answer_text
            for claim in expected.get("forbidden_claims", [])
        ):
            diagnostics["forbidden_claim_domains"].append(expected["domain"])

    return diagnostics


def evaluate_authority_requirements(
    case: AnswerEvaluationCase,
    actual_result: dict[str, Any],
) -> dict[str, Any]:
    expected_ids = set(case.required_authoritative_source_ids)
    actual_requested_ids = set(_collect_requested_authoritative_source_ids(actual_result))
    actual_retrieved_ids = set(_collect_retrieved_authoritative_source_ids(actual_result))
    required_authority_failure = False
    if expected_ids:
        if actual_result["route"] == "structured_lookup":
            source_id = actual_result.get("result", {}).get("source_reference", {}).get("source_id")
            if source_id is not None:
                required_authority_failure = source_id not in expected_ids
            else:
                required_authority_failure = actual_result.get("status") != "clarification_needed"
        else:
            required_authority_failure = not expected_ids.issubset(actual_requested_ids | actual_retrieved_ids)

    forbidden_authority_failures = []
    all_unresolved_dependencies = _collect_unresolved_dependencies(actual_result)
    for rule in case.forbidden_authority_claims:
        subject_contains = rule["subject_contains"].lower()
        forbidden_ids = set(rule["forbidden_source_ids"])
        for dependency in all_unresolved_dependencies:
            if subject_contains in dependency.get("subject", "").lower():
                if forbidden_ids.intersection(set(dependency.get("authority_source_ids", []))):
                    forbidden_authority_failures.append(rule)

    return {
        "required_authority_failure": required_authority_failure,
        "expected_authoritative_source_ids": sorted(expected_ids),
        "actual_requested_authoritative_source_ids": sorted(actual_requested_ids),
        "actual_retrieved_authoritative_source_ids": sorted(actual_retrieved_ids),
        "forbidden_authority_failures": forbidden_authority_failures,
    }


def evaluate_unresolved_dependencies(
    case: AnswerEvaluationCase,
    actual_result: dict[str, Any],
) -> dict[str, Any]:
    all_dependencies = _collect_unresolved_dependencies(actual_result)
    required_specs = list(case.required_unresolved_dependencies)
    for subroute in case.required_subroutes:
        required_specs.extend(subroute.get("required_unresolved_dependencies", []))

    missing_required_dependencies = []
    semantic_review_dependencies = []
    for spec in required_specs:
        matching_state = _evaluate_unresolved_dependency_match(spec, all_dependencies)
        if matching_state == "matched":
            continue
        if matching_state == "semantic_review":
            semantic_review_dependencies.append(spec)
            continue
        missing_required_dependencies.append(spec)

    return {
        "actual_unresolved_dependencies": all_dependencies,
        "missing_required_dependencies": missing_required_dependencies,
        "semantic_review_dependencies": semantic_review_dependencies,
    }


def evaluate_citation_integrity(actual_result: dict[str, Any]) -> dict[str, Any]:
    rag_payloads = _collect_rag_payloads(actual_result)
    if not rag_payloads:
        return {"valid": True, "issues": []}

    issues: list[str] = []
    for payload in rag_payloads:
        validation_issues = payload.get("validation_issues", [])
        for issue in validation_issues:
            if any(
                marker in issue
                for marker in (
                    "citation",
                    "used_evidence",
                    "Citations must contain only objects",
                    "Citation has invalid field",
                )
            ):
                issues.append(issue)
    return {"valid": not issues, "issues": issues}


def evaluate_validation_failures(actual_result: dict[str, Any]) -> dict[str, Any]:
    rag_payloads = _collect_rag_payloads(actual_result)
    if not rag_payloads:
        return {"has_non_citation_failure": False, "issues": []}

    citation_markers = (
        "citation",
        "used_evidence",
        "Citations must contain only objects",
        "Citation has invalid field",
    )
    non_citation_issues: list[str] = []
    for payload in rag_payloads:
        validation_status = payload.get("validation_status")
        validation_issues = payload.get("validation_issues", [])
        if validation_status != "failed":
            continue
        matched_non_citation_issue = False
        for issue in validation_issues:
            if any(marker in issue for marker in citation_markers):
                continue
            non_citation_issues.append(issue)
            matched_non_citation_issue = True
        if not matched_non_citation_issue:
            non_citation_issues.append("validation_status failed for a non-citation reason")

    return {
        "has_non_citation_failure": bool(non_citation_issues),
        "issues": non_citation_issues,
    }


def evaluate_concepts(*, concepts: list[str], text: str) -> dict[str, Any]:
    explicit_matches = []
    semantic_review_concepts = []
    normalized_text = _normalize_text(text)
    for concept in concepts:
        if _normalize_text(concept) in normalized_text:
            explicit_matches.append(concept)
        else:
            semantic_review_concepts.append(concept)
    coverage = (
        len(explicit_matches) / len(concepts)
        if concepts
        else None
    )
    return {
        "explicit_matches": explicit_matches,
        "semantic_review_concepts": semantic_review_concepts,
        "coverage": coverage,
    }


def detect_forbidden_claims(case: AnswerEvaluationCase, actual_result: dict[str, Any]) -> list[str]:
    answer_text = _collect_answer_text(actual_result)
    return [
        claim
        for claim in case.forbidden_claims
        if _forbidden_claim_is_asserted(claim=claim, text=answer_text)
    ]


def detect_inappropriate_answer(case: AnswerEvaluationCase, actual_result: dict[str, Any]) -> bool:
    if actual_result["route"] == "composite":
        return False
    if case.expected_status not in {"clarification_needed", "needs_review", "insufficient_evidence"}:
        return False
    if case.expected_status == "insufficient_evidence":
        return False
    result = actual_result.get("result")
    if not isinstance(result, dict):
        return False
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    return _has_substantive_answer(answer, citations)


def classify_answer_case(*, failures: list[str], semantic_review_reasons: list[str]) -> str:
    if failures:
        return "failure"
    if semantic_review_reasons:
        return "needs_semantic_review"
    return "pass_explicitly"


def compute_answer_evaluation_metrics(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    case_count = len(case_results)
    route_correct = sum(result["deterministic_checks"]["route_correct"] for result in case_results)
    status_correct = sum(result["deterministic_checks"]["status_correct"] for result in case_results)
    completeness_cases = [
        result for result in case_results if result["expected_answer_completeness"] is not None
    ]
    completeness_correct = sum(
        result["deterministic_checks"]["completeness_correct"] for result in completeness_cases
    )
    citation_integrity_valid = sum(result["citation_integrity"]["valid"] for result in case_results)
    authority_ok = sum(
        not result["authority_diagnostics"]["required_authority_failure"]
        and not result["authority_diagnostics"]["forbidden_authority_failures"]
        for result in case_results
    )
    unresolved_cases = [
        result
        for result in case_results
        if result["unresolved_dependency_diagnostics"]["actual_unresolved_dependencies"]
        or result["unresolved_dependency_diagnostics"]["missing_required_dependencies"]
    ]
    unresolved_correct = sum(
        not result["unresolved_dependency_diagnostics"]["missing_required_dependencies"]
        for result in unresolved_cases
    )
    answer_concept_coverages = [
        result["answer_concepts"]["coverage"]
        for result in case_results
        if result["answer_concepts"]["coverage"] is not None
    ]
    evidence_concept_coverages = [
        result["evidence_concepts"]["coverage"]
        for result in case_results
        if result["evidence_concepts"]["coverage"] is not None
    ]
    for result in case_results:
        for concept_result in result["subroute_concept_diagnostics"]["answer_concept_matches"].values():
            if concept_result["coverage"] is not None:
                answer_concept_coverages.append(concept_result["coverage"])
        for concept_result in result["subroute_concept_diagnostics"]["evidence_concept_matches"].values():
            if concept_result["coverage"] is not None:
                evidence_concept_coverages.append(concept_result["coverage"])
    inappropriate_answer_count = sum(result["inappropriate_answer_detected"] for result in case_results)
    unsafe_overanswer_count = sum(
        any(taxonomy in {"unsafe_overanswering", "unsupported_claim_failure"} for taxonomy in result["failure_taxonomy"])
        for result in case_results
    )
    semantic_review_count = sum(result["evaluation_classification"] == "needs_semantic_review" for result in case_results)
    pass_explicitly_count = sum(result["evaluation_classification"] == "pass_explicitly" for result in case_results)
    failure_count = sum(result["evaluation_classification"] == "failure" for result in case_results)

    return {
        "route_accuracy": route_correct / case_count,
        "status_accuracy": status_correct / case_count,
        "completeness_accuracy": (
            completeness_correct / len(completeness_cases) if completeness_cases else None
        ),
        "citation_integrity_rate": citation_integrity_valid / case_count,
        "authority_boundary_compliance_rate": authority_ok / case_count,
        "unresolved_dependency_accuracy": (
            unresolved_correct / len(unresolved_cases) if unresolved_cases else None
        ),
        "answer_concept_coverage_rate": (
            sum(answer_concept_coverages) / len(answer_concept_coverages)
            if answer_concept_coverages
            else None
        ),
        "evidence_concept_coverage_rate": (
            sum(evidence_concept_coverages) / len(evidence_concept_coverages)
            if evidence_concept_coverages
            else None
        ),
        "inappropriate_answer_rate": inappropriate_answer_count / case_count,
        "unsafe_overanswer_rate": unsafe_overanswer_count / case_count,
        "semantic_review_count": semantic_review_count,
        "pass_explicitly_count": pass_explicitly_count,
        "failure_count": failure_count,
        "case_count": case_count,
    }


def _extract_top_level_answer_completeness(actual_result: dict[str, Any]) -> str | None:
    if actual_result["route"] == "rag":
        return actual_result["result"].get("answer_completeness")
    return None


def _collect_requested_authoritative_source_ids(actual_result: dict[str, Any]) -> list[str]:
    if actual_result["route"] == "rag":
        return actual_result["result"].get("retrieval_diagnostics", {}).get("requested_authoritative_source_ids", [])
    if actual_result["route"] == "composite":
        ids: list[str] = []
        for sub in actual_result["subresults"]:
            ids.extend(sub.get("result", {}).get("retrieval_diagnostics", {}).get("requested_authoritative_source_ids", []))
        return list(dict.fromkeys(ids))
    return []


def _collect_retrieved_authoritative_source_ids(actual_result: dict[str, Any]) -> list[str]:
    if actual_result["route"] == "rag":
        return actual_result["result"].get("retrieval_diagnostics", {}).get("retrieved_authoritative_source_ids", [])
    if actual_result["route"] == "composite":
        ids: list[str] = []
        for sub in actual_result["subresults"]:
            ids.extend(sub.get("result", {}).get("retrieval_diagnostics", {}).get("retrieved_authoritative_source_ids", []))
        return list(dict.fromkeys(ids))
    return []


def _collect_unresolved_dependencies(actual_result: dict[str, Any]) -> list[dict[str, Any]]:
    if actual_result["route"] == "rag":
        return actual_result["result"].get("unresolved_dependencies", [])
    if actual_result["route"] == "composite":
        dependencies: list[dict[str, Any]] = []
        for sub in actual_result["subresults"]:
            dependencies.extend(sub.get("result", {}).get("unresolved_dependencies", []))
        return dependencies
    return []


def _collect_rag_payloads(actual_result: dict[str, Any]) -> list[dict[str, Any]]:
    if actual_result["route"] == "rag":
        return [actual_result["result"]]
    if actual_result["route"] == "composite":
        return [
            sub["result"]
            for sub in actual_result["subresults"]
            if sub["route"] == "rag"
        ]
    return []


def _collect_answer_text(actual_result: dict[str, Any]) -> str:
    if actual_result["route"] == "rag":
        return actual_result["result"].get("answer", "")
    if actual_result["route"] == "composite":
        return "\n".join(
            sub["result"].get("answer", "")
            for sub in actual_result["subresults"]
            if sub["route"] == "rag"
        )
    result = actual_result.get("result", {})
    return result.get("message", "")


def _collect_retrieved_evidence_text(actual_result: dict[str, Any]) -> str:
    texts: list[str] = []
    if actual_result["route"] == "rag":
        texts.extend(chunk["text"] for chunk in actual_result["result"].get("retrieved_chunks", []))
    elif actual_result["route"] == "composite":
        for sub in actual_result["subresults"]:
            if sub["route"] == "rag":
                texts.extend(chunk["text"] for chunk in sub["result"].get("retrieved_chunks", []))
    return "\n".join(texts)


def _matches_unresolved_dependency(spec: dict[str, Any], actual: dict[str, Any]) -> bool:
    if actual.get("type") != spec["type"]:
        return False
    if not _subjects_match_with_canonicalization(
        expected_subject=spec["subject_contains"],
        actual_subject=actual.get("subject", ""),
    ):
        return False
    if set(spec["authority_source_ids"]) != set(actual.get("authority_source_ids", [])):
        return False
    if bool(spec["blocking"]) != bool(actual.get("blocking")):
        return False
    return True


def _evaluate_unresolved_dependency_match(
    spec: dict[str, Any],
    actual_dependencies: list[dict[str, Any]],
) -> str:
    if any(_matches_unresolved_dependency(spec, dependency) for dependency in actual_dependencies):
        return "matched"

    for dependency in actual_dependencies:
        if dependency.get("type") != spec["type"]:
            continue
        if set(spec["authority_source_ids"]) != set(dependency.get("authority_source_ids", [])):
            continue
        if bool(spec["blocking"]) != bool(dependency.get("blocking")):
            continue
        if _subjects_are_conservatively_equivalent(
            expected_subject=spec["subject_contains"],
            actual_subject=dependency.get("subject", ""),
        ):
            return "matched"
        return "semantic_review"

    return "missing"


def _compact_actual_result(actual_result: dict[str, Any]) -> dict[str, Any]:
    compact = {
        "route": actual_result["route"],
        "status": actual_result["status"],
        "domains": actual_result.get("domains", []),
        "missing_information": actual_result.get("missing_information", []),
    }
    if actual_result["route"] == "rag":
        compact["result"] = {
            "answer_completeness": actual_result["result"].get("answer_completeness"),
            "validation_status": actual_result["result"].get("validation_status"),
            "unresolved_dependencies": actual_result["result"].get("unresolved_dependencies", []),
        }
    elif actual_result["route"] == "composite":
        compact["subresults"] = [
            {
                "domain": sub["domain"],
                "route": sub["route"],
                "status": sub["status"],
                "missing_information": sub.get("missing_information", []),
                "answer_completeness": sub.get("result", {}).get("answer_completeness"),
                "unresolved_dependencies": sub.get("result", {}).get("unresolved_dependencies", []),
            }
            for sub in actual_result["subresults"]
        ]
    return compact


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    collapsed = re.sub(r"[^a-z0-9 ]+", " ", lowered)
    return re.sub(r"\s+", " ", collapsed).strip()


def _has_substantive_answer(answer: str, citations: list[Any]) -> bool:
    return bool(answer.strip()) and bool(citations)


def _subjects_are_conservatively_equivalent(*, expected_subject: str, actual_subject: str) -> bool:
    expected_canonical = _canonicalize_unresolved_subject(expected_subject)
    actual_canonical = _canonicalize_unresolved_subject(actual_subject)
    if expected_canonical is not None and actual_canonical is not None:
        return expected_canonical == actual_canonical

    expected_tokens = _significant_subject_tokens(expected_subject)
    actual_tokens = _significant_subject_tokens(actual_subject)
    if not expected_tokens or not actual_tokens:
        return False
    if expected_tokens.issubset(actual_tokens):
        return True
    return bool(expected_tokens.intersection(actual_tokens)) and len(expected_tokens) == 1


def _subjects_match_with_canonicalization(*, expected_subject: str, actual_subject: str) -> bool:
    if expected_subject.lower() in actual_subject.lower():
        return True
    expected_canonical = _canonicalize_unresolved_subject(expected_subject)
    actual_canonical = _canonicalize_unresolved_subject(actual_subject)
    return expected_canonical is not None and expected_canonical == actual_canonical


def _canonicalize_unresolved_subject(subject: str) -> str | None:
    normalized = _normalize_text(subject)
    if not normalized:
        return None

    case_specific_warranty_markers = (
        "case specific warranty coverage",
        "case specific warranty coverage determination",
        "warranty coverage determination",
        "applicable warranty outcome",
        "warranty applicability outcome",
        "applicable warranty coverage",
    )
    if any(marker in normalized for marker in case_specific_warranty_markers):
        return "case_specific_warranty_coverage"
    return None


def _significant_subject_tokens(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "applicable",
        "case",
        "coverage",
        "determination",
        "for",
        "full",
        "of",
        "outcome",
        "particular",
        "specific",
        "terms",
        "the",
        "warranty",
    }
    tokens = {
        token
        for token in _normalize_text(text).split()
        if token and token not in stopwords
    }
    return tokens


def _forbidden_claim_is_asserted(*, claim: str, text: str) -> bool:
    normalized_claim = _normalize_text(claim)
    if not normalized_claim:
        return False

    pattern = _phrase_pattern(normalized_claim)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ] or [text]

    for sentence in sentences:
        normalized_sentence = _normalize_text(sentence)
        if not re.search(pattern, normalized_sentence):
            continue
        if _is_negated_forbidden_example(normalized_claim, normalized_sentence):
            continue
        return True
    return False


def _phrase_pattern(normalized_phrase: str) -> str:
    parts = [re.escape(part) for part in normalized_phrase.split()]
    return r"\b" + r"\s+".join(parts) + r"\b"


def _is_negated_forbidden_example(normalized_claim: str, normalized_sentence: str) -> bool:
    negation_cues = (
        "does not state",
        "do not state",
        "not stated",
        "does not provide",
        "do not provide",
        "not provided",
        "does not specify",
        "do not specify",
        "cannot be determined",
        "not determine",
        "no evidence",
        "does not establish",
        "none of the provided evidence specifies",
        "the provided evidence does not state",
    )
    if any(cue in normalized_sentence for cue in negation_cues):
        return True

    remedy_terms = {"refund", "replacement", "replacement payment", "reimbursement", "credit", "compensation"}
    if normalized_claim in remedy_terms and any(cue in normalized_sentence for cue in ("such as", "for example", "e g")):
        if any(word in normalized_sentence for word in ("does not state", "does not provide", "does not specify", "none of the provided evidence")):
            return True

    return False
