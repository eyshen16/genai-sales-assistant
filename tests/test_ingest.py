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

from ingest import (
    chunk_document,
    ingest_documents,
    is_source_eligible,
    load_markdown_document,
    load_source_registry,
    parse_frontmatter,
)


class IngestTests(unittest.TestCase):
    def test_parse_frontmatter_returns_metadata_and_body(self) -> None:
        metadata, body = parse_frontmatter(
            "---\n"
            "title: Example Guide\n"
            "status: approved\n"
            "lifecycle_status: current\n"
            "effective_date: 2026-06-01\n"
            "region: DE\n"
            "---\n\n"
            "# Example Guide\n\n"
            "## Purpose\n"
            "Text.\n"
        )

        self.assertEqual(metadata["title"], "Example Guide")
        self.assertEqual(metadata["status"], "approved")
        self.assertEqual(metadata["lifecycle_status"], "current")
        self.assertIn("## Purpose", body)

    def test_load_markdown_document_reads_frontmatter_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            document_path = Path(tmpdir) / "Guide.md"
            document_path.write_text(
                "---\n"
                "title: Disk Guide\n"
                "status: approved\n"
                "lifecycle_status: current\n"
                "effective_date: 2026-06-01\n"
                "region: DE\n"
                "---\n\n"
                "# Disk Guide\n\n"
                "Body text.\n",
                encoding="utf-8",
            )

            document = load_markdown_document(document_path)

        self.assertEqual(document["metadata"]["title"], "Disk Guide")
        self.assertIn("Body text.", document["body"])

    def test_source_eligibility_uses_registry_requirements(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            entry = self._write_registry(project_root)[0]

            eligible_metadata = {
                "status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-06-01",
                "region": "DE",
            }
            ineligible_metadata = {
                "status": "draft",
                "lifecycle_status": "current",
                "effective_date": "2026-06-01",
                "region": "DE",
            }

            self.assertTrue(
                is_source_eligible(entry, eligible_metadata, as_of_date=date(2026, 8, 23))
            )
            self.assertFalse(
                is_source_eligible(entry, ineligible_metadata, as_of_date=date(2026, 8, 23))
            )

    def test_missing_lifecycle_status_fails_current_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            entry = self._write_registry(project_root)[0]

            metadata = {
                "status": "approved",
                "effective_date": "2026-06-01",
                "region": "DE",
            }

            self.assertFalse(is_source_eligible(entry, metadata, as_of_date=date(2026, 8, 23)))

    def test_future_effective_dates_fail_eligibility_even_when_lifecycle_is_current(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            entry = self._write_registry(project_root)[0]

            metadata = {
                "status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-12-01",
                "region": "DE",
            }

            self.assertFalse(is_source_eligible(entry, metadata, as_of_date=date(2026, 8, 23)))

    def test_ingest_documents_filters_ineligible_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)
            documents_dir = project_root / "data" / "documents"
            documents_dir.mkdir(parents=True)

            (documents_dir / "Eligible.md").write_text(
                self._document_text(
                    title="Eligible Guide",
                    status="approved",
                    lifecycle_status="current",
                    effective_date="2026-06-01",
                    body_sections=[("Purpose", "Eligible text.")],
                ),
                encoding="utf-8",
            )
            (documents_dir / "Future.md").write_text(
                self._document_text(
                    title="Future Guide",
                    status="approved",
                    lifecycle_status="current",
                    effective_date="2026-12-01",
                    body_sections=[("Purpose", "Future text.")],
                ),
                encoding="utf-8",
            )

            chunks = ingest_documents(project_root=project_root, as_of_date=date(2026, 8, 23))

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["title"], "Eligible Guide")

    def test_ingest_documents_raises_for_registered_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)

            with self.assertRaises(FileNotFoundError):
                ingest_documents(project_root=project_root, as_of_date=date(2026, 8, 23))

    def test_chunk_document_groups_content_by_markdown_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            entry = self._write_registry(project_root)[0]
            metadata = {
                "title": "Procedure Guide",
                "status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-06-01",
                "region": "DE",
            }
            body = (
                "# Procedure Guide\n\n"
                "## Installation\n"
                "Prepare the site.\n\n"
                "### Wiring\n"
                "Connect the battery.\n\n"
                "1. Verify isolation.\n"
                "2. Tighten terminals.\n\n"
                "## Commissioning\n"
                "1. Power on the inverter.\n"
                "2. Verify communication.\n"
            )

            chunks = chunk_document(
                entry=entry,
                metadata=metadata,
                body=body,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual([chunk["section"] for chunk in chunks], ["Installation", "Commissioning"])
        self.assertIn("### Wiring", chunks[0]["text"])
        self.assertIn("1. Verify isolation.", chunks[0]["text"])
        self.assertIn("2. Verify communication.", chunks[1]["text"])
        self.assertEqual(chunks[0]["lifecycle_status"], "current")

    def test_chunk_document_safely_splits_oversized_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            entry = self._write_registry(project_root)[0]
            metadata = {
                "title": "Large Guide",
                "status": "approved",
                "lifecycle_status": "current",
                "effective_date": "2026-06-01",
                "region": "DE",
            }
            body = (
                "# Large Guide\n\n"
                "## Procedure\n"
                "Paragraph one with several words.\n\n"
                "Paragraph two with several words.\n\n"
                "Paragraph three with several words.\n"
            )

            chunks = chunk_document(
                entry=entry,
                metadata=metadata,
                body=body,
                as_of_date=date(2026, 8, 23),
                max_chunk_chars=60,
            )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk["section"] == "Procedure" for chunk in chunks))

    def test_load_source_registry_reads_document_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            registry_entries = self._write_registry(project_root)

            loaded_entries = load_source_registry(
                project_root / "consulting" / "knowledge_source_registry.csv"
            )

        self.assertEqual(len(loaded_entries), 2)
        self.assertEqual(loaded_entries[0].source_id, registry_entries[0].source_id)

    def _write_registry(self, project_root: Path):
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
                "source_id": "SRC-100",
                "source_name": "Eligible Guide",
                "source_path": "data/documents/Eligible.md",
                "source_type": "document",
                "owner": "Docs",
                "authoritative_for": "Installation",
                "not_authoritative_for": "Compatibility",
                "region": "DE",
                "required_approval_status": "approved",
                "required_lifecycle_status": "current",
                "usage_mode": "rag",
                "risk_level": "medium",
                "notes": "",
            },
            {
                "source_id": "SRC-101",
                "source_name": "Future Guide",
                "source_path": "data/documents/Future.md",
                "source_type": "document",
                "owner": "Docs",
                "authoritative_for": "Installation",
                "not_authoritative_for": "Compatibility",
                "region": "DE",
                "required_approval_status": "approved",
                "required_lifecycle_status": "current",
                "usage_mode": "rag",
                "risk_level": "medium",
                "notes": "",
            },
        ]

        with registry_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        return load_source_registry(registry_path)

    def _document_text(
        self,
        *,
        title: str,
        status: str,
        lifecycle_status: str,
        effective_date: str,
        body_sections: list[tuple[str, str]],
    ) -> str:
        body_parts = [f"# {title}", ""]
        for heading, content in body_sections:
            body_parts.append(f"## {heading}")
            body_parts.append(content)
            body_parts.append("")

        body = "\n".join(body_parts).strip()
        return (
            "---\n"
            f"title: {title}\n"
            f"status: {status}\n"
            f"lifecycle_status: {lifecycle_status}\n"
            f"effective_date: {effective_date}\n"
            "region: DE\n"
            "---\n\n"
            f"{body}\n"
        )


if __name__ == "__main__":
    unittest.main()
