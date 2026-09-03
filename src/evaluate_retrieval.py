from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from retrieve import DEFAULT_RETRIEVAL_TOP_K, ingest_and_build_index, retrieve_relevant_chunks
from structured_lookup import PROJECT_ROOT, resolve_structured_source


DEFAULT_EVALUATION_CASES_PATH = PROJECT_ROOT / "consulting" / "retrieval_evaluation_cases.json"


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    expected_route: str
    expected_sources: list[str]
    required_authoritative_source: str | None
    expected_evidence: list[str]


@dataclass(frozen=True)
class ExplicitEvidenceMatch:
    concept: str
    matched: bool


def load_evaluation_cases(
    path: Path | str = DEFAULT_EVALUATION_CASES_PATH,
) -> list[EvaluationCase]:
    raw_cases = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            question=case["question"],
            expected_route=case["expected_route"],
            expected_sources=case["expected_sources"],
            required_authoritative_source=case.get("required_authoritative_source"),
            expected_evidence=case["expected_evidence"],
        )
        for case in raw_cases
    ]


def predict_route(question: str) -> str:
    normalized = _normalize_text(question)
    structured_patterns = [
        "kompatibel",
        "compatible",
        "firmware",
        "ve hybrid",
        "homecell",
    ]
    pattern_hits = sum(pattern in normalized for pattern in structured_patterns)
    if ("ve hybrid" in normalized and "homecell" in normalized) and pattern_hits >= 2:
        return "structured_lookup"
    return "semantic_retrieval"


def run_retrieval_evaluation(
    *,
    project_root: Path | str = PROJECT_ROOT,
    as_of_date: date | None = None,
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    cases_path: Path | str = DEFAULT_EVALUATION_CASES_PATH,
) -> dict[str, Any]:
    root = Path(project_root)
    cases = load_evaluation_cases(cases_path)
    retrieval_index = ingest_and_build_index(project_root=root, as_of_date=as_of_date)
    structured_source = resolve_structured_source(root / "consulting" / "knowledge_source_registry.csv")

    case_results: list[dict[str, Any]] = []
    route_correct_count = 0
    source_hit_count = 0
    source_hit_applicable = 0
    authoritative_hit_count = 0
    authoritative_hit_applicable = 0
    explicit_evidence_coverage_total = 0.0
    explicit_evidence_coverage_applicable = 0
    pass_explicitly_count = 0
    needs_semantic_review_count = 0
    retrieval_failure_count = 0

    for case in cases:
        predicted_route = predict_route(case.question)
        route_correct = predicted_route == case.expected_route
        if route_correct:
            route_correct_count += 1

        result: dict[str, Any] = {
            "question": case.question,
            "expected_route": case.expected_route,
            "predicted_route": predicted_route,
            "route_correct": route_correct,
            "expected_sources": case.expected_sources,
            "required_authoritative_source": case.required_authoritative_source,
        }

        if case.expected_route == "semantic_retrieval":
            retrieval_results = retrieve_relevant_chunks(
                query=case.question,
                index=retrieval_index,
                top_k=top_k,
            )
            retrieved_source_ids = [item["source_id"] for item in retrieval_results]
            source_hit = any(source_id in retrieved_source_ids for source_id in case.expected_sources)
            source_hit_applicable += 1
            if source_hit:
                source_hit_count += 1

            authoritative_hit = None
            if case.required_authoritative_source is not None:
                authoritative_hit_applicable += 1
                authoritative_hit = case.required_authoritative_source in retrieved_source_ids
                if authoritative_hit:
                    authoritative_hit_count += 1

            evidence_matches = match_expected_evidence(
                expected_evidence=case.expected_evidence,
                retrieval_results=retrieval_results,
            )
            explicit_evidence_coverage = compute_explicit_evidence_coverage(evidence_matches)
            explicit_evidence_coverage_applicable += 1
            explicit_evidence_coverage_total += explicit_evidence_coverage
            unmatched_evidence = [
                match.concept for match in evidence_matches if not match.matched
            ]
            evaluation_status = classify_retrieval_case(
                route_correct=route_correct,
                source_hit=source_hit,
                authoritative_hit=authoritative_hit,
                explicit_evidence_coverage=explicit_evidence_coverage,
            )
            if evaluation_status == "pass_explicitly":
                pass_explicitly_count += 1
            elif evaluation_status == "needs_semantic_review":
                needs_semantic_review_count += 1
            elif evaluation_status == "retrieval_failure":
                retrieval_failure_count += 1

            result.update(
                {
                    "source_hit": source_hit,
                    "authoritative_source_hit": authoritative_hit,
                    "explicit_evidence_matches": [
                        {"concept": match.concept, "matched": match.matched}
                        for match in evidence_matches
                    ],
                    "explicit_evidence_coverage": explicit_evidence_coverage,
                    "unmatched_evidence": unmatched_evidence,
                    "evaluation_status": evaluation_status,
                    "retrieval_results": retrieval_results,
                    "review_diagnostics": build_review_diagnostics(
                        evaluation_status=evaluation_status,
                        unmatched_evidence=unmatched_evidence,
                        retrieval_results=retrieval_results,
                    ),
                }
            )
        else:
            authoritative_source_hit = (
                case.required_authoritative_source == structured_source.source_id
                if case.required_authoritative_source is not None
                else None
            )
            if case.required_authoritative_source is not None:
                authoritative_hit_applicable += 1
                if authoritative_source_hit:
                    authoritative_hit_count += 1

            result.update(
                {
                    "source_hit": None,
                    "authoritative_source_hit": authoritative_source_hit,
                    "explicit_evidence_matches": [],
                    "explicit_evidence_coverage": None,
                    "unmatched_evidence": [],
                    "evaluation_status": "pass_explicitly" if route_correct else "retrieval_failure",
                    "retrieval_results": [],
                    "review_diagnostics": None,
                }
            )
            if route_correct:
                pass_explicitly_count += 1
            else:
                retrieval_failure_count += 1

        case_results.append(result)

    return {
        "cases": case_results,
        "metrics": {
            "route_accuracy": route_correct_count / len(cases),
            "source_hit_at_k": (
                source_hit_count / source_hit_applicable if source_hit_applicable else None
            ),
            "authoritative_source_hit_at_k": (
                authoritative_hit_count / authoritative_hit_applicable
                if authoritative_hit_applicable
                else None
            ),
            "explicit_evidence_coverage_at_k": (
                explicit_evidence_coverage_total / explicit_evidence_coverage_applicable
                if explicit_evidence_coverage_applicable
                else None
            ),
            "pass_explicitly_count": pass_explicitly_count,
            "pass_explicitly_rate": pass_explicitly_count / len(cases),
            "needs_semantic_review_count": needs_semantic_review_count,
            "retrieval_failure_count": retrieval_failure_count,
            "case_count": len(cases),
            "retrieval_case_count": source_hit_applicable,
        },
    }


def match_expected_evidence(
    *,
    expected_evidence: list[str],
    retrieval_results: list[dict[str, Any]],
) -> list[ExplicitEvidenceMatch]:
    combined_text = " ".join(result["text"] for result in retrieval_results)
    normalized_text = _normalize_text(combined_text)
    return [
        ExplicitEvidenceMatch(
            concept=concept,
            matched=_normalize_text(concept) in normalized_text,
        )
        for concept in expected_evidence
    ]


def compute_explicit_evidence_coverage(matches: list[ExplicitEvidenceMatch]) -> float:
    if not matches:
        return 1.0
    matched_count = sum(match.matched for match in matches)
    return matched_count / len(matches)


def classify_retrieval_case(
    *,
    route_correct: bool,
    source_hit: bool | None,
    authoritative_hit: bool | None,
    explicit_evidence_coverage: float | None,
) -> str:
    if not route_correct:
        return "retrieval_failure"
    if source_hit is False:
        return "retrieval_failure"
    if authoritative_hit is False:
        return "retrieval_failure"
    if explicit_evidence_coverage is not None and explicit_evidence_coverage >= 1.0:
        return "pass_explicitly"
    return "needs_semantic_review"


def build_review_diagnostics(
    *,
    evaluation_status: str,
    unmatched_evidence: list[str],
    retrieval_results: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if evaluation_status != "needs_semantic_review":
        return None
    return {
        "unmatched_evidence": unmatched_evidence,
        "top_k_results": [
            {
                "source_id": item["source_id"],
                "section": item["section"],
                "score": item["score"],
                "text": item["text"],
            }
            for item in retrieval_results
        ],
    }


def summarize_case_statuses(report: dict[str, Any], status: str) -> list[dict[str, Any]]:
    return [case for case in report["cases"] if case["evaluation_status"] == status]


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


if __name__ == "__main__":
    report = run_retrieval_evaluation(as_of_date=date(2026, 8, 23))
    print(json.dumps(report["metrics"], indent=2))
    for status in ("needs_semantic_review", "retrieval_failure"):
        for case in summarize_case_statuses(report, status):
            print(f"[{status}] {case['question']}")
