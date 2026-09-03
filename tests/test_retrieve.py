from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from retrieve import (
    DEFAULT_EMBEDDING_MODEL,
    build_retrieval_index,
    ingest_and_build_index,
    retrieve_relevant_chunks,
)


class RetrieveTests(unittest.TestCase):
    def test_chargeone_pv_surplus_query_returns_chargeone_and_energyhub_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = self._write_retrieval_project(Path(tmpdir))
            index = ingest_and_build_index(
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
                model_name=DEFAULT_EMBEDDING_MODEL,
            )

            results = retrieve_relevant_chunks(
                query="What is required for PV-surplus charging with ChargeOne 11?",
                index=index,
                top_k=4,
            )

        top_sources = {result["source_id"] for result in results}
        top_sections = {(result["source_id"], result["section"]) for result in results}
        self.assertIn("SRC-003", top_sources)
        self.assertIn("SRC-004", top_sources)
        self.assertIn(("SRC-003", "PV-Surplus Charging"), top_sections)
        self.assertTrue(any(result["score"] > 0 for result in results))

    def test_homecell_retrofit_query_returns_battery_retrofit_and_commissioning_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = self._write_retrieval_project(Path(tmpdir))
            index = ingest_and_build_index(
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
                model_name=DEFAULT_EMBEDDING_MODEL,
            )

            results = retrieve_relevant_chunks(
                query="What must be checked when adding a HomeCell battery to an existing installation?",
                index=index,
                top_k=3,
            )

        self.assertEqual(results[0]["source_id"], "SRC-002")
        self.assertIn(results[0]["section"], {"Retrofit Installations", "Commissioning Sequence"})
        self.assertIn("HomeCell", results[0]["title"])

    def test_warranty_query_prioritizes_warranty_policy_when_relevant_text_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = self._write_retrieval_project(Path(tmpdir))
            index = ingest_and_build_index(
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
                model_name=DEFAULT_EMBEDDING_MODEL,
            )

            results = retrieve_relevant_chunks(
                query="What are the warranty conditions for an unauthorized third-party modification?",
                index=index,
                top_k=3,
            )

        self.assertEqual(results[0]["source_id"], "SRC-005")
        self.assertIn("Unauthorized Modifications", results[0]["section"])

    def test_build_retrieval_index_preserves_chunk_metadata(self) -> None:
        chunks = [
            {
                "source_id": "SRC-999",
                "source_name": "Example Guide",
                "source_path": "data/documents/Example.md",
                "title": "Example Title",
                "section": "Example Section",
                "region": "DE",
                "approval_status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-01-01",
                "text": "Example retrieval text.",
            }
        ]

        index = build_retrieval_index(chunks=chunks, model_name=DEFAULT_EMBEDDING_MODEL)
        results = retrieve_relevant_chunks(query="example retrieval", index=index, top_k=1)

        self.assertEqual(results[0]["source_id"], "SRC-999")
        self.assertIn("score", results[0])

    def test_authoritative_warranty_source_is_represented_in_top_k_when_requested(self) -> None:
        index = ingest_and_build_index(
            project_root=PROJECT_ROOT,
            as_of_date=date(2026, 8, 24),
            model_name=DEFAULT_EMBEDDING_MODEL,
        )

        result = retrieve_relevant_chunks(
            query="What warranty conditions apply when modifying an existing VE Hybrid 8 system to add HomeCell 15?",
            index=index,
            top_k=6,
            authoritative_source_ids=["SRC-005"],
            include_diagnostics=True,
        )

        source_ids = [chunk["source_id"] for chunk in result["chunks"]]
        self.assertIn("SRC-005", source_ids)
        self.assertIn("SRC-002", source_ids)
        self.assertFalse(result["authority_gap"])
        self.assertEqual(result["requested_authoritative_source_ids"], ["SRC-005"])
        self.assertEqual(result["retrieved_authoritative_source_ids"], ["SRC-005"])

    def test_supporting_source_does_not_fully_displace_requested_authoritative_source(self) -> None:
        chunks = [
            {
                "source_id": "SRC-002",
                "source_name": "Battery Guide",
                "source_path": "data/documents/Battery.md",
                "title": "Battery Guide",
                "section": "Retrofit",
                "region": "DE",
                "approval_status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-01-01",
                "text": "Warranty conditions for battery retrofit modifications and installation compliance.",
            },
            {
                "source_id": "SRC-005",
                "source_name": "Warranty Policy",
                "source_path": "data/documents/Warranty.md",
                "title": "Warranty Policy",
                "section": "Warranty Eligibility",
                "region": "DE",
                "approval_status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-01-01",
                "text": "Warranty eligibility depends on approved documentation and review conditions.",
            },
        ]

        index = build_retrieval_index(chunks=chunks, model_name=DEFAULT_EMBEDDING_MODEL)
        result = retrieve_relevant_chunks(
            query="What warranty conditions apply to this modification?",
            index=index,
            top_k=1,
            authoritative_source_ids=["SRC-005"],
            include_diagnostics=True,
        )

        self.assertEqual(result["chunks"][0]["source_id"], "SRC-005")
        self.assertFalse(result["authority_gap"])

    def test_no_authoritative_ids_preserves_existing_behavior(self) -> None:
        chunks = [
            {
                "source_id": "SRC-AAA",
                "source_name": "Example A",
                "source_path": "data/documents/A.md",
                "title": "Example A",
                "section": "A",
                "region": "DE",
                "approval_status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-01-01",
                "text": "solar charging requirement",
            },
            {
                "source_id": "SRC-BBB",
                "source_name": "Example B",
                "source_path": "data/documents/B.md",
                "title": "Example B",
                "section": "B",
                "region": "DE",
                "approval_status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-01-01",
                "text": "battery retrofit requirement",
            },
        ]
        index = build_retrieval_index(chunks=chunks, model_name=DEFAULT_EMBEDDING_MODEL)

        plain = retrieve_relevant_chunks(query="battery retrofit requirement", index=index, top_k=1)
        diagnosed = retrieve_relevant_chunks(
            query="battery retrofit requirement",
            index=index,
            top_k=1,
            include_diagnostics=True,
        )

        self.assertEqual([chunk["source_id"] for chunk in plain], [chunk["source_id"] for chunk in diagnosed["chunks"]])
        self.assertFalse(diagnosed["authority_gap"])
        self.assertEqual(diagnosed["requested_authoritative_source_ids"], [])

    def test_authority_gap_is_true_when_requested_authoritative_source_has_no_eligible_chunks(self) -> None:
        chunks = [
            {
                "source_id": "SRC-002",
                "source_name": "Battery Guide",
                "source_path": "data/documents/Battery.md",
                "title": "Battery Guide",
                "section": "Warranty",
                "region": "DE",
                "approval_status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-01-01",
                "text": "Warranty conditions are governed elsewhere.",
            }
        ]
        index = build_retrieval_index(chunks=chunks, model_name=DEFAULT_EMBEDDING_MODEL)
        result = retrieve_relevant_chunks(
            query="What warranty conditions apply?",
            index=index,
            top_k=3,
            authoritative_source_ids=["SRC-005"],
            include_diagnostics=True,
        )

        self.assertTrue(result["authority_gap"])
        self.assertEqual(result["retrieved_authoritative_source_ids"], [])
        self.assertEqual(result["chunks"][0]["source_id"], "SRC-002")

    def test_deterministic_source_ids_do_not_enter_rag_results(self) -> None:
        index = ingest_and_build_index(
            project_root=PROJECT_ROOT,
            as_of_date=date(2026, 8, 24),
            model_name=DEFAULT_EMBEDDING_MODEL,
        )

        result = retrieve_relevant_chunks(
            query="Is VE Hybrid 8 compatible with HomeCell 15?",
            index=index,
            top_k=3,
            authoritative_source_ids=["SRC-001"],
            include_diagnostics=True,
        )

        self.assertTrue(result["authority_gap"])
        self.assertNotIn("SRC-001", [chunk["source_id"] for chunk in result["chunks"]])

    def test_multiple_authoritative_sources_are_treated_as_one_pool_without_source_quotas(self) -> None:
        index = ingest_and_build_index(
            project_root=PROJECT_ROOT,
            as_of_date=date(2026, 8, 24),
            model_name=DEFAULT_EMBEDDING_MODEL,
        )

        result = retrieve_relevant_chunks(
            query="What warranty conditions apply when modifying an existing VE Hybrid 8 system to add HomeCell 15?",
            index=index,
            top_k=4,
            authoritative_source_ids=["SRC-004", "SRC-005"],
            include_diagnostics=True,
        )

        retrieved_authoritative = set(result["retrieved_authoritative_source_ids"])
        self.assertTrue(retrieved_authoritative.issubset({"SRC-004", "SRC-005"}))
        self.assertGreaterEqual(len(retrieved_authoritative), 1)
        self.assertFalse(result["authority_gap"])

    def _write_retrieval_project(self, project_root: Path) -> Path:
        self._write_registry(project_root)
        self._write_documents(project_root)
        return project_root

    def _write_registry(self, project_root: Path) -> None:
        consulting_dir = project_root / "consulting"
        consulting_dir.mkdir(parents=True, exist_ok=True)
        registry_path = consulting_dir / "knowledge_source_registry.csv"

        fieldnames = [
            "source_id",
            "source_name",
            "source_path",
            "source_type",
            "owner",
            "authoritative_for",
            "not_authoritative_for",
            "region",
            "required_approval_status",
            "required_lifecycle_status",
            "usage_mode",
            "risk_level",
            "notes",
        ]
        rows = [
            {
                "source_id": "SRC-002",
                "source_name": "Battery Installation Guide",
                "source_path": "data/documents/Battery_Installation_Guide.md",
                "source_type": "document",
                "owner": "Docs",
                "authoritative_for": "Battery installation prerequisites; commissioning; retrofit requirements",
                "not_authoritative_for": "Product compatibility",
                "region": "DE",
                "required_approval_status": "approved",
                "required_lifecycle_status": "current",
                "usage_mode": "rag",
                "risk_level": "medium",
                "notes": "",
            },
            {
                "source_id": "SRC-003",
                "source_name": "ChargeOne EV Charger Integration Guide",
                "source_path": "data/documents/ChargeOne_EV_Charger_Integration_Guide.md",
                "source_type": "document",
                "owner": "Docs",
                "authoritative_for": "ChargeOne integration",
                "not_authoritative_for": "Product compatibility",
                "region": "DE",
                "required_approval_status": "approved",
                "required_lifecycle_status": "current",
                "usage_mode": "rag",
                "risk_level": "medium",
                "notes": "",
            },
            {
                "source_id": "SRC-004",
                "source_name": "EnergyHub System Integration Guide",
                "source_path": "data/documents/EnergyHub_System_Integration_Guide.md",
                "source_type": "document",
                "owner": "Docs",
                "authoritative_for": "System integration",
                "not_authoritative_for": "Product compatibility",
                "region": "DE",
                "required_approval_status": "approved",
                "required_lifecycle_status": "current",
                "usage_mode": "rag",
                "risk_level": "medium",
                "notes": "",
            },
            {
                "source_id": "SRC-005",
                "source_name": "Warranty Policy",
                "source_path": "data/documents/Warranty_Policy.md",
                "source_type": "document",
                "owner": "Docs",
                "authoritative_for": "Warranty conditions",
                "not_authoritative_for": "Product compatibility",
                "region": "DE",
                "required_approval_status": "approved",
                "required_lifecycle_status": "current",
                "usage_mode": "rag_strict_citation",
                "risk_level": "high",
                "notes": "",
            },
        ]

        with registry_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _write_documents(self, project_root: Path) -> None:
        documents_dir = project_root / "data" / "documents"
        documents_dir.mkdir(parents=True, exist_ok=True)

        (documents_dir / "Battery_Installation_Guide.md").write_text(
            self._doc(
                title="HomeCell Battery Installation and Commissioning Guide",
                body=(
                    "## Retrofit Installations\n"
                    "When a HomeCell battery is added to an existing VoltEdge hybrid inverter installation, "
                    "the installer must verify the inverter firmware version before battery commissioning. "
                    "If a firmware update is required, the inverter must be updated before the HomeCell battery is commissioned. "
                    "The existing installation should also be checked for compliance with current wiring and metering requirements.\n\n"
                    "## Commissioning Sequence\n"
                    "1. Verify product compatibility.\n"
                    "2. Verify and, where required, update inverter firmware.\n"
                    "3. Install and connect the HomeCell battery.\n"
                ),
            ),
            encoding="utf-8",
        )
        (documents_dir / "ChargeOne_EV_Charger_Integration_Guide.md").write_text(
            self._doc(
                title="ChargeOne EV Charger Integration Guide",
                body=(
                    "## PV-Surplus Charging\n"
                    "PV-surplus charging is available only when ChargeOne 11 is integrated with EnergyHub and the required metering data is available. "
                    "If a HomeCell battery is installed, battery and EV charging are coordinated according to the active EnergyHub operating strategy.\n\n"
                    "## Coordinated Energy Management\n"
                    "Functions such as PV-surplus charging and dynamic charging optimization require integration with EnergyHub. "
                    "EnergyHub must have access to current household consumption and PV-generation data.\n"
                ),
            ),
            encoding="utf-8",
        )
        (documents_dir / "EnergyHub_System_Integration_Guide.md").write_text(
            self._doc(
                title="EnergyHub System Integration Guide",
                body=(
                    "## Required Metering Data\n"
                    "Coordinated functions require reliable household grid import/export data. "
                    "PV-surplus charging additionally requires current PV-generation information. "
                    "If required data is unavailable, EnergyHub must not activate dependent functions.\n\n"
                    "## Integration with ChargeOne 11\n"
                    "Before coordinated charging is activated, the installer must verify that EnergyHub receives valid household meter data and valid PV-generation data.\n"
                ),
            ),
            encoding="utf-8",
        )
        (documents_dir / "Warranty_Policy.md").write_text(
            self._doc(
                title="VoltEdge Residential Energy Warranty Policy - Germany",
                body=(
                    "## Unauthorized Modifications\n"
                    "Warranty coverage is void for failures caused by unauthorized third-party modifications, "
                    "unsupported control equipment changes, or unapproved alterations to the installation.\n\n"
                    "## Warranty Conditions\n"
                    "Warranty claims require installation and commissioning according to current approved VoltEdge instructions.\n"
                ),
            ),
            encoding="utf-8",
        )

    def _doc(self, *, title: str, body: str) -> str:
        return (
            "---\n"
            f"title: {title}\n"
            "status: approved\n"
            "lifecycle_status: current\n"
            "effective_date: 2026-01-01\n"
            "region: DE\n"
            "---\n\n"
            f"# {title}\n\n"
            f"{body}\n"
        )


if __name__ == "__main__":
    unittest.main()
