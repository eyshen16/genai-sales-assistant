from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

from ingest import PROJECT_ROOT, SourceRegistryEntry, load_source_registry


DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "consulting" / "knowledge_source_registry.csv"
STRUCTURED_SOURCE_ID = "SRC-001"


@dataclass(frozen=True)
class CompatibilityRule:
    inverter_model: str
    battery_model: str
    min_firmware: str
    max_battery_capacity_kwh: str
    region: str
    status: str
    lifecycle_status: str
    effective_from: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def lookup_compatibility(
    inverter_model: str | None,
    battery_model: str | None,
    firmware_version: str | None,
    region: str | None,
    *,
    project_root: Path | str = PROJECT_ROOT,
    as_of_date: date | None = None,
) -> dict[str, Any]:
    validation_error = _validate_inputs(
        inverter_model=inverter_model,
        battery_model=battery_model,
        firmware_version=firmware_version,
        region=region,
    )
    if validation_error is not None:
        return {
            "status": "invalid_input",
            "message": validation_error,
            "matched_rule": None,
            "required_firmware": None,
            "source_reference": None,
        }

    root = Path(project_root)
    source_entry = resolve_structured_source(root / "consulting" / "knowledge_source_registry.csv")
    source_reference = build_source_reference(source_entry)
    rules = _load_rules(root / source_entry.source_path)

    matched_rule = _find_matching_rule(
        rules=rules,
        inverter_model=inverter_model.strip(),
        battery_model=battery_model.strip(),
        region=region.strip(),
    )

    if matched_rule is None:
        return {
            "status": "no_applicable_rule",
            "message": "No applicable compatibility rule exists for the provided combination and region.",
            "matched_rule": None,
            "required_firmware": None,
            "source_reference": source_reference,
        }

    if not rule_is_eligible(matched_rule, source_entry, as_of_date=as_of_date):
        return {
            "status": "no_applicable_rule",
            "message": "A matching compatibility row exists, but it is not currently eligible for use.",
            "matched_rule": matched_rule.to_dict(),
            "required_firmware": matched_rule.min_firmware,
            "source_reference": source_reference,
        }

    requested_firmware = _parse_version(firmware_version.strip())
    required_firmware = _parse_version(matched_rule.min_firmware)

    if requested_firmware < required_firmware:
        return {
            "status": "firmware_too_low",
            "message": "The combination exists, but the firmware version is below the minimum required version.",
            "matched_rule": matched_rule.to_dict(),
            "required_firmware": matched_rule.min_firmware,
            "source_reference": source_reference,
        }

    return {
        "status": "compatible",
        "message": "The combination is compatible under the matched deterministic rule.",
        "matched_rule": matched_rule.to_dict(),
        "required_firmware": matched_rule.min_firmware,
        "source_reference": source_reference,
    }


def resolve_structured_source(registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> SourceRegistryEntry:
    registry_entries = load_source_registry(registry_path)
    for entry in registry_entries:
        if entry.source_id == STRUCTURED_SOURCE_ID:
            if entry.usage_mode != "deterministic_lookup":
                raise ValueError(
                    f"{STRUCTURED_SOURCE_ID} must use deterministic_lookup, found {entry.usage_mode}."
                )
            return entry
    raise LookupError(f"Could not resolve {STRUCTURED_SOURCE_ID} from the source registry.")


def build_source_reference(entry: SourceRegistryEntry) -> dict[str, str]:
    return {
        "source_id": entry.source_id,
        "source_name": entry.source_name,
        "source_path": entry.source_path,
        "usage_mode": entry.usage_mode,
        "required_approval_status": entry.required_approval_status,
        "required_lifecycle_status": entry.required_lifecycle_status,
    }


def rule_is_eligible(
    rule: CompatibilityRule,
    source_entry: SourceRegistryEntry,
    *,
    as_of_date: date | None = None,
) -> bool:
    if rule.status.strip().lower() != source_entry.required_approval_status.strip().lower():
        return False

    if rule.lifecycle_status.strip().lower() != source_entry.required_lifecycle_status.strip().lower():
        return False

    if rule.region != source_entry.region:
        return False

    if not rule_is_effective_on(rule, as_of_date=as_of_date):
        return False

    return True


def rule_is_effective_on(
    rule: CompatibilityRule,
    *,
    as_of_date: date | None = None,
) -> bool:
    effective_date = date.fromisoformat(rule.effective_from)
    comparison_date = as_of_date or date.today()
    return effective_date <= comparison_date


def _validate_inputs(
    *,
    inverter_model: str | None,
    battery_model: str | None,
    firmware_version: str | None,
    region: str | None,
) -> str | None:
    required_fields = {
        "inverter_model": inverter_model,
        "battery_model": battery_model,
        "firmware_version": firmware_version,
        "region": region,
    }

    missing_fields = [
        field_name
        for field_name, value in required_fields.items()
        if value is None or not str(value).strip()
    ]
    if missing_fields:
        return f"Missing required input: {', '.join(missing_fields)}."

    try:
        _parse_version(str(firmware_version).strip())
    except ValueError:
        return "Invalid firmware_version format."

    return None


def _load_rules(matrix_path: Path | str) -> list[CompatibilityRule]:
    path = Path(matrix_path)
    if not path.exists():
        raise FileNotFoundError(f"Structured compatibility source is missing: {path}")

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            CompatibilityRule(
                inverter_model=row["inverter_model"],
                battery_model=row["battery_model"],
                min_firmware=row["min_firmware"],
                max_battery_capacity_kwh=row["max_battery_capacity_kwh"],
                region=row["region"],
                status=row["status"],
                lifecycle_status=row["lifecycle_status"],
                effective_from=row["effective_from"],
            )
            for row in reader
        ]


def _find_matching_rule(
    *,
    rules: list[CompatibilityRule],
    inverter_model: str,
    battery_model: str,
    region: str,
) -> CompatibilityRule | None:
    for rule in rules:
        if (
            rule.inverter_model == inverter_model
            and rule.battery_model == battery_model
            and rule.region == region
        ):
            return rule
    return None


def _parse_version(version: str) -> tuple[int, ...]:
    parts = version.split(".")
    if not parts or any(part == "" for part in parts):
        raise ValueError("Version string is empty or malformed.")

    parsed_parts: list[int] = []
    for part in parts:
        if not part.isdigit():
            raise ValueError("Version string contains non-numeric parts.")
        parsed_parts.append(int(part))

    while len(parsed_parts) > 1 and parsed_parts[-1] == 0:
        parsed_parts.pop()
    return tuple(parsed_parts)
