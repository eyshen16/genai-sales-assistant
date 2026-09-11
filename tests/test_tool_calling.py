from __future__ import annotations

import json
import sys
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import tool_calling as tc
import compare_compatibility_selection as comparison

QUESTION = "Is VE Hybrid 8 compatible with HomeCell 15 on firmware 4.2 in Germany?"
ARGS = dict(inverter_model="VE Hybrid 8", battery_model="HomeCell 15", firmware_version="4.2", region="Germany")


def model(args=None, *, name="check_compatibility", count=1, raw=None, status="completed"):
    client = Mock()
    client.with_options.return_value = client
    call = NS(type="function_call", name=name, arguments=raw if raw is not None else json.dumps(ARGS if args is None else args))
    client.responses.create.return_value = NS(status=status, output=[call] * count, id="response-test", usage=NS(input_tokens=10, output_tokens=20))
    return client


class ToolCallingTests(unittest.TestCase):
    def test_valid_call_once_and_result_unchanged(self):
        client = model()
        expected = {"status": "compatible", "message": "unchanged"}
        with patch.object(tc, "lookup_compatibility", return_value=expected) as lookup:
            actual = tc.run_compatibility_experiment(QUESTION, client=client)
        self.assertIs(actual["tool_result"], expected)
        lookup.assert_called_once()
        self.assertEqual(lookup.call_args.kwargs["region"], "DE")
        client.responses.create.assert_called_once()
        client.with_options.assert_called_once_with(timeout=60.0, max_retries=0)
        self.assertEqual(client.responses.create.call_args.kwargs["tool_choice"], "auto")
        self.assertFalse(client.responses.create.call_args.kwargs["parallel_tool_calls"])

    def test_real_lookup_results_preserved(self):
        for battery, firmware, region, expected in [
            ("HomeCell 15", "4.2", "Germany", "compatible"),
            ("HomeCell 15", "4.1", "Germany", "firmware_too_low"),
            ("HomeCell 20", "4.2", "Germany", "no_applicable_rule"),
            ("HomeCell 15", "4.2", "Austria", "no_applicable_rule"),
        ]:
            with self.subTest(expected=expected, region=region):
                args = dict(ARGS, battery_model=battery, firmware_version=firmware, region=region)
                q = f"Is {battery} compatible with VE Hybrid 8 on firmware {firmware} in {region}?"
                actual = tc.run_compatibility_experiment(q, client=model(args), as_of_date=date(2026, 9, 10))
                self.assertEqual(actual["status"], "executed")
                self.assertEqual(actual["tool_result"]["status"], expected)
                self.assertEqual(actual["tool_result"]["source_reference"]["source_id"], "SRC-001")

    def test_invalid_input_lookup_result_not_overridden(self):
        with patch.object(tc, "lookup_compatibility", return_value={"status": "invalid_input"}) as lookup:
            actual = tc.run_compatibility_experiment(QUESTION, client=model())
        lookup.assert_called_once()
        self.assertEqual(actual["tool_result"], {"status": "invalid_input"})

    def test_missing_ambiguous_and_unrecognized_region(self):
        for q, args, field in [
            (QUESTION, dict(ARGS, inverter_model=None), "inverter_model"),
            (QUESTION.replace(" in Germany", ""), dict(ARGS, region=None), "region"),
            (QUESTION.replace("4.2", "4.1 or 4.2"), ARGS, "firmware_version"),
            (QUESTION.replace("Germany", "Germany or Austria"), ARGS, "region"),
            (QUESTION.replace("Germany", "France"), dict(ARGS, region="France"), "region"),
        ]:
            with self.subTest(field=field, q=q), patch.object(tc, "lookup_compatibility") as lookup:
                actual = tc.run_compatibility_experiment(q, client=model(args))
                self.assertEqual(actual["status"], "clarification_needed")
                self.assertIn(field, actual["missing_information"])
                lookup.assert_not_called()

    def test_invalid_arguments_never_execute(self):
        clients = [model(name="shell"), model(count=2), model(raw="{oops"),
                   model(dict(ARGS, source_id="SRC-001")), model({"region": "Germany"}),
                   model(dict(ARGS, firmware_version=4.2)), model(dict(ARGS, battery_model="HomeCell 10")),
                   model(dict(ARGS, region="DE")), model(dict(ARGS, firmware_version=""))]
        for client in clients:
            with self.subTest(client=client), patch.object(tc, "lookup_compatibility") as lookup:
                self.assertEqual(tc.run_compatibility_experiment(QUESTION, client=client)["status"], "invalid_tool_call")
                lookup.assert_not_called()

    def test_no_tool_or_prose_is_not_answer(self):
        client = model(count=0)
        client.responses.create.return_value.output = [NS(type="message", content=[NS(type="output_text", text="Definitely compatible!")])]
        with patch.object(tc, "lookup_compatibility") as lookup:
            actual = tc.run_compatibility_experiment(QUESTION, client=client)
        self.assertEqual(actual["status"], "no_tool_selected")
        self.assertIsNone(actual["tool_result"])
        self.assertNotIn("Definitely", json.dumps(actual))
        lookup.assert_not_called()

    def test_model_failures_redact_exception_and_never_execute(self):
        for client in [model(status="incomplete"), model()]:
            if client.responses.create.return_value.status == "completed":
                client.responses.create.side_effect = RuntimeError("secret-value must not be exposed")
            with patch.object(tc, "lookup_compatibility") as lookup:
                actual = tc.run_compatibility_experiment(QUESTION, client=client)
            self.assertEqual(actual["status"], "model_failed")
            self.assertNotIn("secret-value", json.dumps(actual))
            lookup.assert_not_called()

    def test_lookup_error(self):
        with patch.object(tc, "lookup_compatibility", side_effect=FileNotFoundError("private path")):
            actual = tc.run_compatibility_experiment(QUESTION, client=model())
        self.assertEqual(actual["status"], "execution_failed")
        self.assertNotIn("private path", json.dumps(actual))

    def test_negative_scope_veto_even_when_model_selects(self):
        for suffix in [" What are the warranty exclusions?", " Explain PV-surplus charging.",
                       " Certify my third-party controller.", " Also check VE Hybrid 12 with HomeCell 10."]:
            with self.subTest(suffix=suffix), patch.object(tc, "lookup_compatibility") as lookup:
                actual = tc.run_compatibility_experiment(QUESTION + suffix, client=model())
                self.assertEqual(actual["status"], "scope_rejected")
                lookup.assert_not_called()

    def test_blank_input_does_not_call_model(self):
        client = model()
        self.assertEqual(tc.run_compatibility_experiment(" ", client=client)["status"], "invalid_tool_call")
        client.responses.create.assert_not_called()

    def test_version_prefix_and_bad_syntax_rejected(self):
        for question, args in [(QUESTION.replace("4.2", "4.2.0"), ARGS),
                               (QUESTION.replace("4.2", "v4.2"), dict(ARGS, firmware_version="v4.2"))]:
            with patch.object(tc, "lookup_compatibility") as lookup:
                self.assertEqual(tc.run_compatibility_experiment(question, client=model(args))["status"], "invalid_tool_call")
                lookup.assert_not_called()

    def test_refusal_is_model_failure(self):
        client = model(count=0)
        client.responses.create.return_value.output = [NS(type="message", content=[NS(type="refusal")])]
        self.assertEqual(tc.run_compatibility_experiment(QUESTION, client=client)["status"], "model_failed")

    def test_aliases_and_explicit_region_cannot_be_replaced(self):
        for alias, code in {
            "DE": "DE", "Germany": "DE", "germany": "DE", "Deutschland": "DE",
            "AT": "AT", "Austria": "AT", "austria": "AT", "Österreich": "AT",
        }.items():
            question = QUESTION.replace("Germany", alias)
            execution, missing, issues = tc.validate_arguments(dict(ARGS, region=alias), question)
            self.assertEqual(execution["region"], code)
            self.assertEqual(missing, [])
            self.assertEqual(issues, [])
        actual = tc.run_compatibility_experiment(QUESTION.replace("Germany", "Austria"), client=model())
        self.assertEqual(actual["status"], "invalid_tool_call")

    def test_region_code_boundaries_do_not_treat_lowercase_at_as_austria(self):
        cases = [
            ("in DE at version 4.1", {"DE"}),
            ("in AT on version 4.1", {"AT"}),
            ("in Austria at version 4.1", {"AT"}),
            ("in Germany at version 4.1", {"DE"}),
            ("a valid pair at version 4.1", set()),
            ("in Germany or Austria at version 4.1", {"DE", "AT"}),
        ]
        for suffix, expected in cases:
            with self.subTest(suffix=suffix):
                self.assertEqual(tc.explicit_region_codes(suffix), expected)

        question = "Are HomeCell 15 and VE Hybrid 12 a valid pair in DE at version 4.1?"
        execution, missing, issues = tc.validate_arguments(
            {"inverter_model": "VE Hybrid 12", "battery_model": "HomeCell 15",
             "firmware_version": "4.1", "region": "DE"}, question
        )
        self.assertEqual(execution["region"], "DE")
        self.assertEqual(missing, [])
        self.assertEqual(issues, [])

        conflict = question.replace("DE", "Germany or Austria")
        execution, missing, issues = tc.validate_arguments(
            {"inverter_model": "VE Hybrid 12", "battery_model": "HomeCell 15",
             "firmware_version": "4.1", "region": "Germany"}, conflict
        )
        self.assertIsNone(execution["region"])
        self.assertEqual(missing, ["region"])
        self.assertEqual(issues, [])

    def test_live_comparison_three_repeats_and_variability(self):
        calls = []
        def fake(question, **kwargs):
            calls.append(question)
            return {"status": "no_tool_selected", "proposed_tool": None,
                    "execution_arguments": None, "missing_information": [], "tool_result": None,
                    "diagnostics": {"input_tokens": 1, "output_tokens": 1}}
        report = comparison.run_comparison(live=True, experiment=fake)
        self.assertEqual(len(calls), 48)
        self.assertEqual(report["summary"]["variable_cases"], 0)

    def test_preview_never_calls_experiment(self):
        experiment = Mock(side_effect=AssertionError("Paid request"))
        report = comparison.run_comparison(experiment=experiment)
        experiment.assert_not_called()
        self.assertEqual(report["summary"]["cases"], 16)
        self.assertEqual(report["summary"]["attempts"], 0)

    def test_comparison_flags_unsafe_selection_even_if_vetoed(self):
        case = {"select_tool": False, "execute": False}
        actual = {"status": "scope_rejected", "proposed_tool": "check_compatibility"}
        self.assertEqual(comparison.assess(case, actual), ["selection_mismatch"])


if __name__ == "__main__":
    unittest.main()
