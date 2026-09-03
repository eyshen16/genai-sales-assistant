from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate_answers import (
    AnswerEvaluationCase,
    classify_answer_case,
    detect_forbidden_claims,
    detect_inappropriate_answer,
    evaluate_answer_case,
    load_answer_evaluation_cases,
    run_answer_evaluation,
)


class EvaluateAnswersTests(unittest.TestCase):
    def test_load_cases_reads_expected_schema(self) -> None:
        cases = load_answer_evaluation_cases(PROJECT_ROOT / "consulting" / "answer_evaluation_cases.json")
        self.assertEqual(len(cases), 20)
        self.assertTrue(all(case.case_id for case in cases))

    def test_exact_route_and_status_match_can_pass_explicitly(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-001",
            question="Q",
            category="structured",
            expected_route="structured_lookup",
            expected_status="completed",
            expected_answer_completeness=None,
            required_domains=["KD-002"],
            required_authoritative_source_ids=["SRC-001"],
        )
        actual = {
            "route": "structured_lookup",
            "status": "completed",
            "domains": ["KD-002"],
            "missing_information": [],
            "result": {"source_reference": {"source_id": "SRC-001"}},
        }

        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "pass_explicitly")

    def test_route_failure_is_classified_as_failure(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-002",
            question="Q",
            category="structured",
            expected_route="structured_lookup",
            expected_status="completed",
            expected_answer_completeness=None,
            required_domains=[],
        )
        actual = {"route": "rag", "status": "completed", "domains": [], "missing_information": [], "result": {}}

        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "failure")
        self.assertIn("routing_failure", result["failure_taxonomy"])

    def test_completeness_failure_is_detected(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-003",
            question="Q",
            category="rag_complete",
            expected_route="rag",
            expected_status="completed",
            expected_answer_completeness="complete",
            required_domains=[],
        )
        actual = {
            "route": "rag",
            "status": "partial",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer_completeness": "partial",
                "validation_status": "validated",
                "validation_issues": [],
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [],
                "retrieved_chunks": [],
            },
        }
        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "failure")
        self.assertIn("completeness_misclassification", result["failure_taxonomy"])

    def test_missing_unresolved_dependency_is_failure(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-004",
            question="Q",
            category="rag_partial",
            expected_route="rag",
            expected_status="partial",
            expected_answer_completeness="partial",
            required_domains=[],
            required_unresolved_dependencies=[
                {
                    "type": "structured_dependency",
                    "subject_contains": "firmware",
                    "authority_source_ids": ["SRC-001"],
                    "blocking": True,
                }
            ],
        )
        actual = {
            "route": "rag",
            "status": "partial",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer_completeness": "partial",
                "validation_status": "validated",
                "validation_issues": [],
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [],
                "retrieved_chunks": [],
            },
        }
        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "failure")
        self.assertIn("unresolved_dependency_error", result["failure_taxonomy"])

    def test_wrong_authority_source_is_failure(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-005",
            question="Q",
            category="rag_partial",
            expected_route="rag",
            expected_status="partial",
            expected_answer_completeness="partial",
            required_domains=[],
            required_unresolved_dependencies=[
                {
                    "type": "structured_dependency",
                    "subject_contains": "compatibility",
                    "authority_source_ids": ["SRC-001"],
                    "blocking": True,
                }
            ],
        )
        actual = {
            "route": "rag",
            "status": "partial",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer_completeness": "partial",
                "validation_status": "validated",
                "validation_issues": [],
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [
                    {
                        "type": "structured_dependency",
                        "subject": "compatibility",
                        "authority_source_ids": ["SRC-002"],
                        "reason": "wrong",
                        "blocking": True,
                    }
                ],
                "retrieved_chunks": [],
            },
        }
        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "failure")

    def test_structured_clarification_does_not_fail_authority_without_lookup_result(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-005A",
            question="Q",
            category="structured",
            expected_route="structured_lookup",
            expected_status="clarification_needed",
            expected_answer_completeness=None,
            required_domains=["KD-002"],
            required_authoritative_source_ids=["SRC-001"],
            required_missing_information=["inverter_model"],
        )
        actual = {
            "route": "structured_lookup",
            "status": "clarification_needed",
            "domains": ["KD-002"],
            "missing_information": ["inverter_model"],
            "result": {"message": "need more info"},
        }

        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "pass_explicitly")
        self.assertFalse(result["authority_diagnostics"]["required_authority_failure"])

    def test_forbidden_claim_detection_is_failure(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-006",
            question="Q",
            category="modal_trap",
            expected_route="rag",
            expected_status="completed",
            expected_answer_completeness="complete",
            required_domains=[],
            forbidden_claims=["warranty is void"],
        )
        actual = {
            "route": "rag",
            "status": "completed",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer": "The warranty is void.",
                "answer_completeness": "complete",
                "validation_status": "validated",
                "validation_issues": [],
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [],
                "retrieved_chunks": [],
            },
        }
        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "failure")
        self.assertIn("unsupported_claim_failure", result["failure_taxonomy"])

    def test_forbidden_claim_matching_uses_word_boundaries(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-006A",
            question="Q",
            category="needs_review",
            expected_route="needs_review",
            expected_status="needs_review",
            expected_answer_completeness=None,
            required_domains=[],
            forbidden_claims=["no", "supported"],
        )
        actual = {
            "route": "needs_review",
            "status": "needs_review",
            "domains": [],
            "missing_information": [],
            "result": {"message": "Automated resolution is not safe for unsupported equipment."},
        }

        detected = detect_forbidden_claims(case, actual)
        self.assertEqual(detected, [])

    def test_negated_forbidden_remedy_example_does_not_count_as_assertion(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-006B",
            question="Q",
            category="abstention",
            expected_route="rag",
            expected_status="insufficient_evidence",
            expected_answer_completeness="none",
            required_domains=[],
            forbidden_claims=["refund", "replacement"],
        )
        actual = {
            "route": "rag",
            "status": "partial",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer": "The provided evidence does not state any monetary remedy such as a refund or replacement payment.",
                "answer_completeness": "partial",
                "validation_status": "validated",
                "validation_issues": [],
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [],
                "retrieved_chunks": [],
            },
        }

        detected = detect_forbidden_claims(case, actual)
        self.assertEqual(detected, [])

    def test_insufficient_evidence_explanation_with_citations_is_not_inappropriate_answer(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-006C",
            question="Q",
            category="abstention",
            expected_route="rag",
            expected_status="insufficient_evidence",
            expected_answer_completeness="none",
            required_domains=[],
        )
        actual = {
            "route": "rag",
            "status": "insufficient_evidence",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer": "The supplied warranty evidence does not state an exact duration in years.",
                "answer_completeness": "none",
                "validation_status": "validated",
                "validation_issues": [],
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [],
                "retrieved_chunks": [],
            },
        }

        self.assertFalse(detect_inappropriate_answer(case, actual))

    def test_paraphrase_gap_falls_back_to_semantic_review(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-007",
            question="Q",
            category="rag_complete",
            expected_route="rag",
            expected_status="completed",
            expected_answer_completeness="complete",
            required_domains=[],
            required_answer_concepts=["household meter data"],
        )
        actual = {
            "route": "rag",
            "status": "completed",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer": "The system needs valid home energy measurement input.",
                "answer_completeness": "complete",
                "validation_status": "validated",
                "validation_issues": [],
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [],
                "retrieved_chunks": [],
            },
        }
        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "needs_semantic_review")

    def test_unresolved_dependency_paraphrase_does_not_hard_fail_when_authority_type_and_blocking_match(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-007A",
            question="Q",
            category="rag_partial",
            expected_route="rag",
            expected_status="partial",
            expected_answer_completeness="partial",
            required_domains=[],
            required_unresolved_dependencies=[
                {
                    "type": "review_dependency",
                    "subject_contains": "coverage",
                    "authority_source_ids": ["SRC-005"],
                    "blocking": True,
                }
            ],
        )
        actual = {
            "route": "rag",
            "status": "partial",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer": "Coverage cannot be confirmed from the supplied facts.",
                "answer_completeness": "partial",
                "validation_status": "validated",
                "validation_issues": [],
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [
                    {
                        "type": "review_dependency",
                        "subject": "case-specific warranty coverage determination",
                        "authority_source_ids": ["SRC-005"],
                        "reason": "review required",
                        "blocking": True,
                    }
                ],
                "retrieved_chunks": [],
            },
        }
        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertNotEqual(result["evaluation_classification"], "failure")

    def test_canonical_unresolved_dependency_subject_matching_accepts_warranty_variants(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-007B",
            question="Q",
            category="rag_partial",
            expected_route="rag",
            expected_status="partial",
            expected_answer_completeness="partial",
            required_domains=[],
            required_unresolved_dependencies=[
                {
                    "type": "review_dependency",
                    "subject_contains": "applicable warranty outcome",
                    "authority_source_ids": ["SRC-005"],
                    "blocking": True,
                }
            ],
        )
        actual = {
            "route": "rag",
            "status": "partial",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer": "A definitive outcome cannot be confirmed from the supplied facts.",
                "answer_completeness": "partial",
                "validation_status": "validated",
                "validation_issues": [],
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [
                    {
                        "type": "review_dependency",
                        "subject": "case-specific warranty coverage determination",
                        "authority_source_ids": ["SRC-005"],
                        "reason": "review required",
                        "blocking": True,
                    }
                ],
                "retrieved_chunks": [],
            },
        }
        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertNotIn("unresolved_dependency_error", result["failure_taxonomy"])

    def test_inappropriate_answer_detection_for_needs_review(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-008",
            question="Q",
            category="escalation",
            expected_route="needs_review",
            expected_status="needs_review",
            expected_answer_completeness=None,
            required_domains=[],
        )
        actual = {
            "route": "needs_review",
            "status": "needs_review",
            "domains": [],
            "missing_information": [],
            "result": {"message": "Automated resolution is not safe under the current governance rules."},
        }
        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "pass_explicitly")

    def test_clarification_behavior_is_checked(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-009",
            question="Q",
            category="structured",
            expected_route="structured_lookup",
            expected_status="clarification_needed",
            expected_answer_completeness=None,
            required_domains=[],
            required_missing_information=["firmware_version"],
        )
        actual = {
            "route": "structured_lookup",
            "status": "clarification_needed",
            "domains": [],
            "missing_information": [],
            "result": {"message": "need more info"},
        }
        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertEqual(result["evaluation_classification"], "failure")
        self.assertIn("missing_clarification_behavior", result["failure_taxonomy"])

    def test_run_answer_evaluation_supports_mock_runner_without_live_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            cases_path = Path(tmpdir) / "cases.json"
            cases_path.write_text(
                '[{"case_id":"TEST-010","question":"Q","category":"structured","expected_route":"structured_lookup","expected_status":"completed","required_domains":[]}]',
                encoding="utf-8",
            )

            report = run_answer_evaluation(
                cases_path=cases_path,
                runner=lambda question: {
                    "route": "structured_lookup",
                    "status": "completed",
                    "domains": [],
                    "missing_information": [],
                    "result": {"source_reference": {"source_id": "SRC-001"}},
                },
            )

        self.assertEqual(report["metrics"]["case_count"], 1)

    def test_non_citation_validation_failure_is_not_labeled_as_citation_integrity_failure(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-010A",
            question="Q",
            category="rag_complete",
            expected_route="rag",
            expected_status="completed",
            expected_answer_completeness="complete",
            required_domains=[],
        )
        actual = {
            "route": "rag",
            "status": "generation_failed",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer": "",
                "answer_completeness": "none",
                "validation_status": "failed",
                "validation_issues": ["Answer must not be empty."],
                "citations": [],
                "used_evidence": [],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [],
                "retrieved_chunks": [],
            },
        }

        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertIn("generation_validation_failure", result["failure_taxonomy"])
        self.assertNotIn("citation_integrity_failure", result["failure_taxonomy"])

    def test_citation_validation_failure_still_uses_citation_integrity_taxonomy(self) -> None:
        case = AnswerEvaluationCase(
            case_id="TEST-010B",
            question="Q",
            category="rag_complete",
            expected_route="rag",
            expected_status="completed",
            expected_answer_completeness="complete",
            required_domains=[],
        )
        actual = {
            "route": "rag",
            "status": "generation_failed",
            "domains": [],
            "missing_information": [],
            "result": {
                "answer": "Some answer",
                "answer_completeness": "none",
                "validation_status": "failed",
                "validation_issues": ["Invalid citation reference: E9."],
                "citations": [],
                "used_evidence": [],
                "retrieval_diagnostics": {},
                "unresolved_dependencies": [],
                "retrieved_chunks": [],
            },
        }

        result = evaluate_answer_case(case=case, actual_result=actual)
        self.assertIn("citation_integrity_failure", result["failure_taxonomy"])

    def test_cleaned_case_spec_reflects_revised_domain_and_route_expectations(self) -> None:
        cases = {
            case.case_id: case
            for case in load_answer_evaluation_cases(PROJECT_ROOT / "consulting" / "answer_evaluation_cases.json")
        }
        self.assertEqual(cases["ANS-007"].required_domains, ["KD-011"])
        self.assertEqual(cases["ANS-008"].required_domains, ["KD-009"])
        self.assertEqual(cases["ANS-017"].expected_route, "rag")
        self.assertEqual(cases["ANS-017"].expected_status, "completed")
        self.assertEqual(cases["ANS-017"].expected_answer_completeness, "complete")


if __name__ == "__main__":
    unittest.main()
