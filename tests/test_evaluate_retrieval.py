from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate_retrieval import (
    classify_retrieval_case,
    compute_explicit_evidence_coverage,
    load_evaluation_cases,
    match_expected_evidence,
    predict_route,
    run_retrieval_evaluation,
)
from retrieve import DEFAULT_RETRIEVAL_TOP_K


class EvaluateRetrievalTests(unittest.TestCase):
    def test_load_evaluation_cases_reads_expected_fields(self) -> None:
        cases = load_evaluation_cases(PROJECT_ROOT / "consulting" / "retrieval_evaluation_cases.json")

        self.assertGreaterEqual(len(cases), 12)
        self.assertTrue(all(case.question for case in cases))
        self.assertTrue(all(case.expected_route for case in cases))

    def test_predict_route_marks_compatibility_questions_as_structured_lookup(self) -> None:
        route = predict_route("Ist HomeCell 15 mit VE Hybrid 8 kompatibel?")
        self.assertEqual(route, "structured_lookup")

    def test_predict_route_marks_document_questions_as_semantic_retrieval(self) -> None:
        route = predict_route("What data must EnergyHub receive before coordinated charging is activated?")
        self.assertEqual(route, "semantic_retrieval")

    def test_explicit_evidence_coverage_uses_transparent_concept_matching(self) -> None:
        matches = match_expected_evidence(
            expected_evidence=["valid household meter data", "valid PV-generation data"],
            retrieval_results=[
                {
                    "text": (
                        "EnergyHub receives valid household meter data and valid PV-generation data "
                        "before coordinated charging."
                    )
                }
            ],
        )
        coverage = compute_explicit_evidence_coverage(matches)
        self.assertEqual(coverage, 1.0)

    def test_lexical_mismatch_does_not_automatically_become_retrieval_failure(self) -> None:
        status = classify_retrieval_case(
            route_correct=True,
            source_hit=True,
            authoritative_hit=True,
            explicit_evidence_coverage=0.5,
        )
        self.assertEqual(status, "needs_semantic_review")

    def test_true_missing_source_is_still_classified_as_retrieval_failure(self) -> None:
        status = classify_retrieval_case(
            route_correct=True,
            source_hit=False,
            authoritative_hit=None,
            explicit_evidence_coverage=0.0,
        )
        self.assertEqual(status, "retrieval_failure")

    def test_run_retrieval_evaluation_returns_aggregate_metrics(self) -> None:
        report = run_retrieval_evaluation(
            project_root=PROJECT_ROOT,
            as_of_date=date(2026, 8, 23),
        )

        self.assertIn("metrics", report)
        self.assertEqual(report["metrics"]["case_count"], len(report["cases"]))
        self.assertGreater(report["metrics"]["route_accuracy"], 0.0)
        self.assertIn("explicit_evidence_coverage_at_k", report["metrics"])

    def test_semantic_review_cases_expose_diagnostics(self) -> None:
        report = run_retrieval_evaluation(
            project_root=PROJECT_ROOT,
            as_of_date=date(2026, 8, 23),
        )

        communication_case = next(
            case
            for case in report["cases"]
            if case["question"] == "Does loss of communication with EnergyHub stop basic charging?"
        )
        self.assertEqual(communication_case["evaluation_status"], "needs_semantic_review")
        self.assertIsNotNone(communication_case["review_diagnostics"])
        self.assertGreater(len(communication_case["review_diagnostics"]["top_k_results"]), 0)

    def test_evaluation_uses_shared_default_top_k(self) -> None:
        self.assertEqual(DEFAULT_RETRIEVAL_TOP_K, 6)


if __name__ == "__main__":
    unittest.main()
