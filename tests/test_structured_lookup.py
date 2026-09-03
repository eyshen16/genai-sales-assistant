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

from structured_lookup import lookup_compatibility, resolve_structured_source


class StructuredLookupTests(unittest.TestCase):
    def test_returns_compatible_for_matching_rule_with_sufficient_firmware(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)
            self._write_matrix(
                project_root,
                rows=[
                    self._matrix_row(
                        inverter_model="VE Hybrid 8",
                        battery_model="HomeCell 15",
                        min_firmware="4.2",
                        region="DE",
                        status="approved",
                        effective_from="2026-07-01",
                    )
                ],
            )

            result = lookup_compatibility(
                inverter_model="VE Hybrid 8",
                battery_model="HomeCell 15",
                firmware_version="4.2",
                region="DE",
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual(result["status"], "compatible")
        self.assertEqual(result["required_firmware"], "4.2")
        self.assertEqual(result["matched_rule"]["battery_model"], "HomeCell 15")
        self.assertEqual(result["source_reference"]["source_id"], "SRC-001")

    def test_returns_firmware_too_low_when_rule_exists_but_version_is_below_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)
            self._write_matrix(
                project_root,
                rows=[self._matrix_row(min_firmware="4.2", effective_from="2026-07-01")],
            )

            result = lookup_compatibility(
                inverter_model="VE Hybrid 8",
                battery_model="HomeCell 15",
                firmware_version="4.1",
                region="DE",
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual(result["status"], "firmware_too_low")
        self.assertEqual(result["required_firmware"], "4.2")
        self.assertEqual(result["matched_rule"]["inverter_model"], "VE Hybrid 8")

    def test_returns_no_applicable_rule_for_unknown_combination(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)
            self._write_matrix(project_root, rows=[self._matrix_row()])

            result = lookup_compatibility(
                inverter_model="VE Hybrid 99",
                battery_model="HomeCell 15",
                firmware_version="4.2",
                region="DE",
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual(result["status"], "no_applicable_rule")
        self.assertIsNone(result["matched_rule"])
        self.assertEqual(
            result["source_reference"]["source_path"],
            "data/structured/compatibility_matrix.csv",
        )

    def test_returns_no_applicable_rule_for_region_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)
            self._write_matrix(project_root, rows=[self._matrix_row(region="DE")])

            result = lookup_compatibility(
                inverter_model="VE Hybrid 8",
                battery_model="HomeCell 15",
                firmware_version="4.2",
                region="FR",
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual(result["status"], "no_applicable_rule")
        self.assertIsNone(result["matched_rule"])

    def test_returns_invalid_input_for_missing_required_fields(self) -> None:
        result = lookup_compatibility(
            inverter_model="",
            battery_model="HomeCell 15",
            firmware_version="4.2",
            region="DE",
        )

        self.assertEqual(result["status"], "invalid_input")
        self.assertIn("inverter_model", result["message"])
        self.assertIsNone(result["source_reference"])

    def test_returns_invalid_input_for_malformed_firmware_version(self) -> None:
        result = lookup_compatibility(
            inverter_model="VE Hybrid 8",
            battery_model="HomeCell 15",
            firmware_version="4.x",
            region="DE",
        )

        self.assertEqual(result["status"], "invalid_input")
        self.assertEqual(result["matched_rule"], None)

    def test_treats_equal_firmware_as_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)
            self._write_matrix(project_root, rows=[self._matrix_row(min_firmware="4.1")])

            result = lookup_compatibility(
                inverter_model="VE Hybrid 8",
                battery_model="HomeCell 15",
                firmware_version="4.1",
                region="DE",
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual(result["status"], "compatible")
        self.assertEqual(result["required_firmware"], "4.1")

    def test_treats_semantically_equivalent_firmware_versions_as_equal(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)
            self._write_matrix(project_root, rows=[self._matrix_row(min_firmware="4.2")])

            result = lookup_compatibility(
                inverter_model="VE Hybrid 8",
                battery_model="HomeCell 15",
                firmware_version="4.2.0",
                region="DE",
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual(result["status"], "compatible")

    def test_registry_drives_source_resolution_at_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(
                project_root,
                source_path="data/structured/custom_matrix.csv",
            )
            self._write_matrix(
                project_root,
                path="data/structured/custom_matrix.csv",
                rows=[self._matrix_row()],
            )

            source = resolve_structured_source(
                project_root / "consulting" / "knowledge_source_registry.csv"
            )
            result = lookup_compatibility(
                inverter_model="VE Hybrid 8",
                battery_model="HomeCell 15",
                firmware_version="4.2",
                region="DE",
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual(source.source_path, "data/structured/custom_matrix.csv")
        self.assertEqual(result["status"], "compatible")
        self.assertEqual(
            result["source_reference"]["source_path"],
            "data/structured/custom_matrix.csv",
        )

    def test_raises_for_non_deterministic_registry_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root, usage_mode="rag")
            self._write_matrix(project_root, rows=[self._matrix_row()])

            with self.assertRaises(ValueError):
                lookup_compatibility(
                    inverter_model="VE Hybrid 8",
                    battery_model="HomeCell 15",
                    firmware_version="4.2",
                    region="DE",
                    project_root=project_root,
                    as_of_date=date(2026, 8, 23),
                )

    def test_future_effective_dates_are_not_yet_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)
            self._write_matrix(
                project_root,
                rows=[self._matrix_row(effective_from="2026-12-01")],
            )

            result = lookup_compatibility(
                inverter_model="VE Hybrid 8",
                battery_model="HomeCell 15",
                firmware_version="4.2",
                region="DE",
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual(result["status"], "no_applicable_rule")
        self.assertEqual(result["matched_rule"]["effective_from"], "2026-12-01")

    def test_lifecycle_status_must_match_registry_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = Path(tmpdir)
            self._write_registry(project_root)
            self._write_matrix(
                project_root,
                rows=[self._matrix_row(lifecycle_status="superseded")],
            )

            result = lookup_compatibility(
                inverter_model="VE Hybrid 8",
                battery_model="HomeCell 15",
                firmware_version="4.2",
                region="DE",
                project_root=project_root,
                as_of_date=date(2026, 8, 23),
            )

        self.assertEqual(result["status"], "no_applicable_rule")
        self.assertEqual(result["matched_rule"]["lifecycle_status"], "superseded")

    def _write_registry(
        self,
        project_root: Path,
        *,
        source_path: str = "data/structured/compatibility_matrix.csv",
        usage_mode: str = "deterministic_lookup",
    ) -> None:
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
        row = {
            "source_id": "SRC-001",
            "source_name": "Compatibility Matrix",
            "source_path": source_path,
            "source_type": "structured",
            "owner": "Product Engineering / System Engineering",
            "authoritative_for": "Product compatibility",
            "not_authoritative_for": "Installation procedures",
            "region": "DE",
            "required_approval_status": "approved",
            "required_lifecycle_status": "current",
            "usage_mode": usage_mode,
            "risk_level": "high",
            "notes": "",
        }

        with registry_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    def _write_matrix(
        self,
        project_root: Path,
        *,
        rows: list[dict[str, str]],
        path: str = "data/structured/compatibility_matrix.csv",
    ) -> None:
        matrix_path = project_root / path
        matrix_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "inverter_model",
            "battery_model",
            "min_firmware",
            "max_battery_capacity_kwh",
            "region",
            "status",
            "lifecycle_status",
            "effective_from",
        ]

        with matrix_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _matrix_row(
        self,
        *,
        inverter_model: str = "VE Hybrid 8",
        battery_model: str = "HomeCell 15",
        min_firmware: str = "4.2",
        region: str = "DE",
        status: str = "approved",
        lifecycle_status: str = "current",
        effective_from: str = "2026-07-01",
    ) -> dict[str, str]:
        return {
            "inverter_model": inverter_model,
            "battery_model": battery_model,
            "min_firmware": min_firmware,
            "max_battery_capacity_kwh": "15",
            "region": region,
            "status": status,
            "lifecycle_status": lifecycle_status,
            "effective_from": effective_from,
        }


if __name__ == "__main__":
    unittest.main()
