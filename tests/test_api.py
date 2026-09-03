from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import api


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def test_valid_post_query_returns_200_and_delegates_to_orchestrator(self) -> None:
        orchestrator_result = {
            "question": "Is VE Hybrid 8 compatible with HomeCell 15 on firmware 4.2?",
            "route": "structured_lookup",
            "status": "completed",
            "domains": ["KD-002", "KD-003"],
            "result": {"status": "compatible", "message": "ok"},
            "subresults": [],
            "missing_information": [],
            "review_required": False,
            "router_reason": "structured route",
        }

        with patch.object(api, "orchestrate_question", return_value=orchestrator_result) as orchestrate_mock:
            response = self.client.post(
                "/query",
                json={"question": "Is VE Hybrid 8 compatible with HomeCell 15 on firmware 4.2?"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), orchestrator_result)
        orchestrate_mock.assert_called_once_with("Is VE Hybrid 8 compatible with HomeCell 15 on firmware 4.2?")

    def test_mocked_structured_response_is_returned_through_api_contract(self) -> None:
        payload = {
            "question": "Is VE Hybrid 8 compatible with HomeCell 15 on firmware 4.2?",
            "route": "structured_lookup",
            "status": "completed",
            "domains": ["KD-002", "KD-003"],
            "result": {
                "status": "compatible",
                "message": "The combination is compatible under the matched deterministic rule.",
                "matched_rule": {"inverter_model": "VE Hybrid 8"},
                "required_firmware": "4.2",
                "source_reference": {"source_id": "SRC-001"},
            },
            "subresults": [],
            "missing_information": [],
            "review_required": False,
            "router_reason": "structured route",
        }

        with patch.object(api, "orchestrate_question", return_value=payload):
            response = self.client.post("/query", json={"question": payload["question"]})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["route"], "structured_lookup")
        self.assertEqual(body["result"]["source_reference"]["source_id"], "SRC-001")

    def test_mocked_rag_response_is_returned_through_api_contract(self) -> None:
        payload = {
            "question": "What is required for PV-surplus charging with ChargeOne 11?",
            "route": "rag",
            "status": "completed",
            "domains": ["KD-009", "KD-010"],
            "result": {
                "answer": "PV-surplus charging requires EnergyHub integration.",
                "citations": [{"evidence_id": "E1"}],
                "used_evidence": ["E1"],
                "insufficient_evidence": False,
                "answer_completeness": "complete",
                "unresolved_dependencies": [],
                "validation_status": "validated",
                "validation_issues": [],
                "retrieved_chunks": [{"source_id": "SRC-004", "text": "evidence"}],
                "retrieval_diagnostics": {
                    "requested_authoritative_source_ids": ["SRC-004"],
                    "retrieved_authoritative_source_ids": ["SRC-004"],
                    "authority_gap": False,
                },
            },
            "subresults": [],
            "missing_information": [],
            "review_required": False,
            "router_reason": "rag route",
        }

        with patch.object(api, "orchestrate_question", return_value=payload):
            response = self.client.post("/query", json={"question": payload["question"]})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["route"], "rag")
        self.assertEqual(body["result"]["answer_completeness"], "complete")

    def test_governed_non_success_status_still_returns_http_200(self) -> None:
        payload = {
            "question": "Is HomeCell 15 compatible with my inverter?",
            "route": "structured_lookup",
            "status": "clarification_needed",
            "domains": ["KD-002"],
            "result": {
                "message": "Structured lookup requires additional information before execution.",
                "required_parameters": ["inverter_model", "firmware_version"],
            },
            "subresults": [],
            "missing_information": ["inverter_model", "firmware_version"],
            "review_required": False,
            "router_reason": "structured route",
        }

        with patch.object(api, "orchestrate_question", return_value=payload):
            response = self.client.post("/query", json={"question": payload["question"]})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "clarification_needed")

    def test_missing_question_returns_422(self) -> None:
        with patch.object(api, "orchestrate_question") as orchestrate_mock:
            response = self.client.post("/query", json={})

        self.assertEqual(response.status_code, 422)
        orchestrate_mock.assert_not_called()

    def test_non_string_question_returns_422(self) -> None:
        with patch.object(api, "orchestrate_question") as orchestrate_mock:
            response = self.client.post("/query", json={"question": 42})

        self.assertEqual(response.status_code, 422)
        orchestrate_mock.assert_not_called()

    def test_empty_question_returns_422(self) -> None:
        with patch.object(api, "orchestrate_question") as orchestrate_mock:
            response = self.client.post("/query", json={"question": ""})

        self.assertEqual(response.status_code, 422)
        orchestrate_mock.assert_not_called()

    def test_whitespace_only_question_returns_422(self) -> None:
        with patch.object(api, "orchestrate_question") as orchestrate_mock:
            response = self.client.post("/query", json={"question": "   \n\t  "})

        self.assertEqual(response.status_code, 422)
        orchestrate_mock.assert_not_called()

    def test_unexpected_orchestrator_exception_returns_predictable_5xx_json(self) -> None:
        with patch.object(api, "orchestrate_question", side_effect=RuntimeError("boom")):
            response = self.client.post("/query", json={"question": "What is required for PV-surplus charging?"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"detail": "Internal server error"})

    def test_health_returns_200_with_expected_payload(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
