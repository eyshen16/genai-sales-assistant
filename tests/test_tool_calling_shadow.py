from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import tool_calling_shadow as shadow
from router import route_question


class ShadowGateTests(unittest.TestCase):
    def gate(self, question):
        return shadow.decide_shadow_gate(question, route_question(question))

    def test_clean_structured_fast_path(self):
        result = self.gate("Is HomeCell 10 compatible with VE Hybrid 12 on firmware 4.0 in Germany?")
        self.assertEqual(result["action"], "preserve_structured")
        self.assertFalse(result["invoke_semantic_selector"])

    def test_indirect_unresolved_pair_invokes_selector(self):
        result = self.gate("Would HomeCell 10 pair with VE Hybrid 12 in Deutschland if it runs software release 4.0?")
        self.assertEqual(result["action"], "semantic_selection")
        self.assertTrue(result["invoke_semantic_selector"])

    def test_structured_provenance_anomalies_invoke_selector(self):
        cases = [
            ("Is HomeCell 10 compatible with VE Hybrid 8 on firmware 4.0?", "region_not_explicit"),
            ("Does HomeCell 10 work with VE Hybrid 12 on firmware 4.0 in Austria?", "region_provenance_mismatch"),
            ("Can HomeCell 15 work with VE Hybrid 12 in Germany if its firmware is either 4.0 or 4.1?", "multiple_firmware_values"),
        ]
        for question, anomaly in cases:
            with self.subTest(question=question):
                result = self.gate(question)
                self.assertTrue(result["invoke_semantic_selector"])
                self.assertIn(anomaly, result["anomalies"])

    def test_shadow_region_detection_uses_code_boundaries(self):
        self.assertEqual(shadow._regions("in DE at version 4.1"), {"DE"})
        self.assertEqual(shadow._regions("in AT on version 4.1"), {"AT"})
        self.assertEqual(shadow._regions("in Austria at version 4.1"), {"AT"})
        self.assertEqual(shadow._regions("in Germany at version 4.1"), {"DE"})
        self.assertEqual(shadow._regions("a valid pair at version 4.1"), set())
        self.assertEqual(shadow._regions("in Germany or Austria"), {"DE", "AT"})

    def test_rag_and_composite_are_preserved(self):
        rag = self.gate("What meter data is needed before coordinated charging begins?")
        composite = self.gate("Is HomeCell 10 compatible with VE Hybrid 12 on firmware 4.0 in Germany, and what warranty exclusions apply?")
        self.assertEqual(rag["action"], "preserve_rag")
        self.assertEqual(composite["action"], "preserve_composite")

    def test_governance_sensitive_scope_fails_closed(self):
        result = self.gate("Can HomeCell 15 and VE Hybrid 8 operate together on release 4.2 in Germany, and will coverage remain valid?")
        self.assertEqual(result["action"], "fail_closed_governance")
        self.assertFalse(result["invoke_semantic_selector"])

    def test_preview_never_calls_model_and_dataset_is_frozen_shape(self):
        experiment = Mock(side_effect=AssertionError("must not call"))
        report = shadow.run_shadow_evaluation(experiment=experiment)
        self.assertEqual(report["summary"]["cases"], 24)
        self.assertEqual(report["summary"]["model_attempts"], 0)
        experiment.assert_not_called()

    def test_live_runner_calls_only_gated_cases_three_times_and_persists(self):
        def fake(question, **kwargs):
            case = next(case for case in json.loads(shadow.CASES_PATH.read_text()) if case["question"] == question)
            args = case.get("arguments")
            missing = case.get("clarification", [])
            return {
                "status": "clarification_needed" if missing else "executed",
                "proposed_tool": "check_compatibility", "proposed_arguments": args,
                "execution_arguments": args or {
                    "inverter_model": "VE Hybrid 8", "battery_model": "HomeCell 10",
                    "firmware_version": "4.0", "region": None,
                },
                "missing_information": missing, "validation_issues": [],
                "tool_result": None if missing else shadow.lookup_compatibility(**args, as_of_date=shadow.EVALUATION_DATE),
                "diagnostics": {"model_latency_ms": 100, "input_tokens": 10, "output_tokens": 5},
            }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results.jsonl"
            report = shadow.run_shadow_evaluation(live=True, output_path=output, experiment=fake)
            lines = output.read_text().splitlines()
        self.assertEqual(len(lines), 44)
        self.assertEqual(report["summary"]["gated_cases"], 10)
        self.assertEqual(report["summary"]["model_attempts"], 30)
        self.assertTrue(all(json.loads(line)["case_id"] for line in lines))


if __name__ == "__main__":
    unittest.main()
