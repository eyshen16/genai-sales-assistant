from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from router import build_executable_subquestion, route_question


class RouterTests(unittest.TestCase):
    def test_explicit_authority_resolution_uses_source_ids(self) -> None:
        from router import load_source_of_truth_mappings, resolve_authority_entries_for_mapping
        from ingest import load_source_registry

        mappings = load_source_of_truth_mappings(PROJECT_ROOT / "consulting" / "source_of_truth_mapping.csv")
        registry_entries = load_source_registry(PROJECT_ROOT / "consulting" / "knowledge_source_registry.csv")

        authority_entries = resolve_authority_entries_for_mapping(mappings["KD-007"], registry_entries)

        self.assertEqual([entry.source_id for entry in authority_entries], ["SRC-002", "SRC-003"])

    def test_routes_compatibility_question_to_structured_lookup(self) -> None:
        result = route_question("Is VE Hybrid 8 compatible with HomeCell 15 on firmware 4.2?")

        self.assertEqual(result["route"], "structured_lookup")
        self.assertIn("KD-002", result["domains"])
        self.assertIn("KD-003", result["domains"])
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["subroutes"], [])
        self.assertEqual(
            result["execution"],
            {
                "inverter_model": "VE Hybrid 8",
                "battery_model": "HomeCell 15",
                "firmware_version": "4.2",
                "region": "DE",
                "missing_information": [],
            },
        )

    def test_routes_pv_surplus_question_to_rag(self) -> None:
        result = route_question("What is required for PV-surplus charging with ChargeOne 11?")

        self.assertEqual(result["route"], "rag")
        self.assertIn("KD-009", result["domains"])
        self.assertIn("KD-010", result["domains"])

    def test_routes_warranty_question_to_rag(self) -> None:
        result = route_question("What are the warranty conditions for an unauthorized third-party modification?")

        self.assertEqual(result["route"], "rag")
        self.assertIn("KD-017", result["domains"])
        self.assertIn("KD-020", result["domains"])

    def test_routes_multi_domain_question_to_composite(self) -> None:
        result = route_question(
            "Can I add HomeCell 15 to my VE Hybrid 8 system, and will that affect my warranty?"
        )

        self.assertEqual(result["route"], "composite")
        self.assertGreaterEqual(len(result["subroutes"]), 3)
        structured_subroute = next(
            subroute for subroute in result["subroutes"] if subroute["route"] == "structured_lookup"
        )
        self.assertEqual(structured_subroute["intent"], "Is HomeCell 15 compatible with VE Hybrid 8?")
        self.assertEqual(structured_subroute["execution"]["inverter_model"], "VE Hybrid 8")
        self.assertEqual(structured_subroute["execution"]["battery_model"], "HomeCell 15")
        self.assertIsNone(structured_subroute["execution"]["firmware_version"])
        self.assertEqual(structured_subroute["missing_information"], ["firmware_version"])
        retrofit_subroute = next(subroute for subroute in result["subroutes"] if subroute["domain"] == "KD-007")
        warranty_subroute = next(subroute for subroute in result["subroutes"] if subroute["domain"] == "KD-017")
        self.assertEqual(
            retrofit_subroute["intent"],
            "What retrofit requirements apply when adding HomeCell 15 to an existing VE Hybrid 8 installation?",
        )
        self.assertEqual(
            warranty_subroute["intent"],
            "What warranty conditions apply when modifying an existing VE Hybrid 8 system to add HomeCell 15?",
        )
        self.assertNotIn("warranty", retrofit_subroute["intent"].lower())
        self.assertNotIn("compatible", warranty_subroute["intent"].lower())
        self.assertTrue(any(subroute["domain"] == "KD-017" for subroute in result["subroutes"]))
        self.assertTrue(any(subroute["domain"] == "KD-007" for subroute in result["subroutes"]))
        self.assertIn("firmware_version", result["missing_information"])

    def test_routes_unsupported_definitive_question_to_needs_review(self) -> None:
        result = route_question(
            "Will an undocumented third-party controller definitely work with the system and remain under warranty?"
        )

        self.assertEqual(result["route"], "needs_review")
        self.assertEqual(result["confidence"], "high")
        self.assertIn("unsupported third-party configuration", result["reason"])

    def test_reports_missing_information_for_partial_compatibility_question(self) -> None:
        result = route_question("Is HomeCell 15 compatible with my inverter?")

        self.assertEqual(result["route"], "structured_lookup")
        self.assertIn("KD-002", result["domains"])
        self.assertIn("inverter_model", result["missing_information"])
        self.assertEqual(result["execution"]["battery_model"], "HomeCell 15")
        self.assertIsNone(result["execution"]["inverter_model"])
        self.assertIn("inverter_model", result["execution"]["missing_information"])

    def test_paraphrased_compatibility_question_still_routes_to_structured_lookup(self) -> None:
        result = route_question("Does HomeCell 15 work with VE Hybrid 8 if the firmware version is 4.2.0?")

        self.assertEqual(result["route"], "structured_lookup")
        self.assertIn("KD-002", result["domains"])
        self.assertIn("KD-003", result["domains"])
        self.assertEqual(result["status"] if "status" in result else None, None)
        self.assertEqual(result["execution"]["firmware_version"], "4.2.0")

    def test_standard_charging_without_energyhub_routes_to_rag(self) -> None:
        result = route_question("Can ChargeOne 11 provide standard charging without EnergyHub?")

        self.assertEqual(result["route"], "rag")
        self.assertIn("KD-012", result["domains"])
        self.assertIn("KD-011", result["domains"])

    def test_basic_charging_without_energyhub_paraphrase_routes_to_rag(self) -> None:
        result = route_question("Will basic charging still work if EnergyHub is not connected?")

        self.assertEqual(result["route"], "rag")
        self.assertIn("KD-012", result["domains"])
        self.assertIn("KD-011", result["domains"])

    def test_document_premise_about_firmware_does_not_create_retrofit_subroute(self) -> None:
        result = route_question(
            "Since the retrofit guide says to check firmware, that means VE Hybrid 8 requires firmware 4.2, right?"
        )

        self.assertEqual(result["route"], "structured_lookup")
        self.assertEqual(result["domains"], ["KD-003"])
        self.assertEqual(result["subroutes"], [])
        self.assertIn("battery_model", result["missing_information"])
        self.assertEqual(result["execution"]["inverter_model"], "VE Hybrid 8")
        self.assertEqual(result["execution"]["firmware_version"], "4.2")

    def test_document_premise_about_compatibility_does_not_create_rag_subroute(self) -> None:
        result = route_question(
            "The installation guide mentions HomeCell 15, so it must be compatible with VE Hybrid 8, right?"
        )

        self.assertEqual(result["route"], "structured_lookup")
        self.assertEqual(result["domains"], ["KD-002"])
        self.assertEqual(result["subroutes"], [])
        self.assertEqual(result["execution"]["inverter_model"], "VE Hybrid 8")
        self.assertEqual(result["execution"]["battery_model"], "HomeCell 15")
        self.assertIn("firmware_version", result["missing_information"])

    def test_paraphrased_warranty_question_still_routes_to_rag(self) -> None:
        result = route_question("Is coverage affected by unsupported third-party equipment changes?")

        self.assertEqual(result["route"], "rag")
        self.assertIn("KD-017", result["domains"])

    def test_warranty_domain_specific_subquestion_for_modification_is_narrowed(self) -> None:
        intent = build_executable_subquestion(
            question="What are the warranty conditions for an unauthorized third-party modification?",
            domain_id="KD-017",
        )

        self.assertEqual(
            intent,
            "What warranty conditions apply to an unauthorized third-party modification?",
        )

    def test_composite_subquestions_preserve_relevant_context_without_leaking_other_domains(self) -> None:
        result = route_question(
            "Can I add HomeCell 15 to my VE Hybrid 8 system, and will that affect my warranty?"
        )
        intents = {subroute["domain"]: subroute["intent"] for subroute in result["subroutes"]}

        self.assertIn("HomeCell 15", intents["KD-002"])
        self.assertIn("VE Hybrid 8", intents["KD-002"])
        self.assertIn("existing VE Hybrid 8 installation", intents["KD-007"])
        self.assertIn("modifying an existing VE Hybrid 8 system", intents["KD-017"])
        self.assertNotIn("warranty", intents["KD-007"].lower())
        self.assertNotIn("compatible", intents["KD-017"].lower())

    def test_ambiguous_question_routes_to_needs_review(self) -> None:
        result = route_question("Can you advise me on this setup?")

        self.assertEqual(result["route"], "needs_review")
        self.assertEqual(result["confidence"], "low")


if __name__ == "__main__":
    unittest.main()
