from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

from ingest import PROJECT_ROOT, SourceRegistryEntry, load_source_registry


DEFAULT_TAXONOMY_PATH = PROJECT_ROOT / "consulting" / "knowledge_domain_taxonomy.csv"
DEFAULT_SOURCE_MAPPING_PATH = PROJECT_ROOT / "consulting" / "source_of_truth_mapping.csv"
ALLOWED_ROUTES = {"structured_lookup", "rag", "composite", "needs_review"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}


@dataclass(frozen=True)
class KnowledgeDomain:
    domain_id: str
    knowledge_domain: str
    description: str
    typical_owner: str
    routing_hints: tuple[str, ...]


@dataclass(frozen=True)
class SourceOfTruthMapping:
    domain_id: str
    knowledge_domain: str
    primary_authoritative_source: str
    primary_authoritative_source_ids: tuple[str, ...]
    supporting_sources: str
    supporting_source_ids: tuple[str, ...]
    owner: str
    resolution_rule: str


def route_question(
    question: str,
    *,
    project_root: Path | str = PROJECT_ROOT,
) -> dict[str, object]:
    normalized_question = question.strip()
    if not normalized_question:
        return {
            "route": "needs_review",
            "domains": [],
            "confidence": "low",
            "reason": "The question is empty, so no safe domain routing decision can be made.",
            "subroutes": [],
            "missing_information": [],
        }

    root = Path(project_root)
    domains = load_domain_taxonomy(root / "consulting" / "knowledge_domain_taxonomy.csv")
    mappings = load_source_of_truth_mappings(root / "consulting" / "source_of_truth_mapping.csv")
    registry_entries = load_source_registry(root / "consulting" / "knowledge_source_registry.csv")

    detected_domains = detect_domains(question=normalized_question, domains=domains)
    if not detected_domains:
        return {
            "route": "needs_review",
            "domains": [],
            "confidence": "low",
            "reason": "No configured knowledge domain could be identified with enough confidence.",
            "subroutes": [],
            "missing_information": [],
        }

    definitive_unsupported = _requests_definitive_unsupported_conclusion(normalized_question)
    if definitive_unsupported:
        return {
            "route": "needs_review",
            "domains": [domain["domain_id"] for domain in detected_domains],
            "confidence": "high",
            "reason": "The question asks for a definitive conclusion about an undocumented or unsupported third-party configuration, which the current governance model reserves for review.",
            "subroutes": [],
            "missing_information": [],
        }

    subroutes = []
    unresolved_domains: list[str] = []
    aggregated_missing_information: set[str] = set()
    for domain_match in detected_domains:
        mapping = mappings.get(domain_match["domain_id"])
        if mapping is None:
            unresolved_domains.append(domain_match["domain_id"])
            continue

        authority_entries = resolve_authority_entries_for_mapping(mapping, registry_entries)
        if not authority_entries:
            unresolved_domains.append(domain_match["domain_id"])
            continue

        resolved_route = resolve_route_for_mapping(authority_entries)
        if resolved_route is None:
            unresolved_domains.append(domain_match["domain_id"])
            continue

        subroute = build_subroute(
            question=normalized_question,
            domain_match=domain_match,
            mapping=mapping,
            authority_entries=authority_entries,
            route=resolved_route,
        )
        aggregated_missing_information.update(subroute.get("missing_information", []))
        subroutes.append(
            subroute
        )

    if unresolved_domains:
        return {
            "route": "needs_review",
            "domains": [domain["domain_id"] for domain in detected_domains],
            "confidence": "low",
            "reason": (
                "One or more identified domains could not be resolved to an authoritative configured route: "
                + ", ".join(unresolved_domains)
                + "."
            ),
            "subroutes": subroutes,
            "missing_information": sorted(aggregated_missing_information),
        }

    unique_routes = {subroute["route"] for subroute in subroutes}
    if len(subroutes) == 1:
        top_route = subroutes[0]["route"]
    elif len(unique_routes) == 1:
        top_route = unique_routes.pop()
    else:
        top_route = "composite"

    reason = _build_top_level_reason(subroutes=subroutes, top_route=top_route)
    confidence = derive_top_level_confidence(subroutes)
    return {
        "route": top_route,
        "domains": [domain["domain_id"] for domain in detected_domains],
        "confidence": confidence,
        "reason": reason,
        "subroutes": [] if top_route != "composite" else subroutes,
        "missing_information": sorted(aggregated_missing_information),
        "execution": _top_level_execution(top_route=top_route, subroutes=subroutes),
        "authoritative_source_ids": _aggregate_authoritative_source_ids(top_route=top_route, subroutes=subroutes),
    }


def load_domain_taxonomy(taxonomy_path: Path | str = DEFAULT_TAXONOMY_PATH) -> list[KnowledgeDomain]:
    path = Path(taxonomy_path)
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            KnowledgeDomain(
                domain_id=row["domain_id"],
                knowledge_domain=row["knowledge_domain"],
                description=row["description"],
                typical_owner=row["typical_owner"],
                routing_hints=tuple(
                    hint.strip().lower()
                    for hint in row.get("routing_hints", "").split(";")
                    if hint.strip()
                ),
            )
            for row in reader
        ]


def load_source_of_truth_mappings(
    mapping_path: Path | str = DEFAULT_SOURCE_MAPPING_PATH,
) -> dict[str, SourceOfTruthMapping]:
    path = Path(mapping_path)
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return {
            row["domain_id"]: SourceOfTruthMapping(
                domain_id=row["domain_id"],
                knowledge_domain=row["knowledge_domain"],
                primary_authoritative_source=row["primary_authoritative_source"],
                primary_authoritative_source_ids=_split_ids(row.get("primary_authoritative_source_ids", "")),
                supporting_sources=row["supporting_sources"],
                supporting_source_ids=_split_ids(row.get("supporting_source_ids", "")),
                owner=row["owner"],
                resolution_rule=row["resolution_rule"],
            )
            for row in reader
        }


def detect_domains(*, question: str, domains: list[KnowledgeDomain]) -> list[dict[str, str]]:
    normalized_question = normalize_text(question)
    matches: list[dict[str, str | int]] = []
    for domain in domains:
        if not domain.routing_hints:
            continue

        matched_hints = [
            hint for hint in domain.routing_hints if hint and hint in normalized_question
        ]
        if not matched_hints:
            continue

        confidence = "high" if len(matched_hints) > 1 else "medium"
        matches.append(
            {
                "domain_id": domain.domain_id,
                "knowledge_domain": domain.knowledge_domain,
                "confidence": confidence,
                "match_count": len(matched_hints),
            }
        )

    if _looks_like_compatibility_question(normalized_question):
        _upsert_domain_match(matches, "KD-002", "Product compatibility", confidence="high", boost=10)
    if _looks_like_firmware_question(normalized_question):
        _upsert_domain_match(matches, "KD-003", "Firmware requirements", confidence="high", boost=10)
    if _looks_like_warranty_question(normalized_question):
        _upsert_domain_match(matches, "KD-017", "Warranty conditions", confidence="high", boost=10)
    if _looks_like_third_party_question(normalized_question):
        _upsert_domain_match(matches, "KD-020", "Third-party equipment limitations", confidence="high", boost=10)
    if _looks_like_retrofit_question(normalized_question):
        _upsert_domain_match(matches, "KD-007", "Retrofit requirements", confidence="medium", boost=5)
    if _looks_like_metering_question(normalized_question):
        _upsert_domain_match(matches, "KD-010", "Metering requirements", confidence="medium", boost=5)

    prioritized_domain_ids = {
        "KD-002",
        "KD-003",
        "KD-007",
        "KD-008",
        "KD-009",
        "KD-010",
        "KD-011",
        "KD-012",
        "KD-017",
        "KD-020",
        "KD-022",
    }
    filtered = [match for match in matches if match["domain_id"] in prioritized_domain_ids]
    filtered = _apply_premise_vs_intent_precedence(normalized_question, filtered)
    filtered.sort(key=lambda item: (item["match_count"], item["confidence"] == "high"), reverse=True)
    return [
        {
            "domain_id": str(match["domain_id"]),
            "knowledge_domain": str(match["knowledge_domain"]),
            "confidence": str(match["confidence"]),
        }
        for match in _dedupe_domain_matches(filtered)
    ]


def usage_mode_to_route(usage_mode: str) -> str | None:
    normalized = usage_mode.strip().lower()
    if normalized == "deterministic_lookup":
        return "structured_lookup"
    if normalized in {"rag", "rag_strict_citation"}:
        return "rag"
    return None


def resolve_authority_entries_for_mapping(
    mapping: SourceOfTruthMapping,
    registry_entries: list[SourceRegistryEntry],
) -> list[SourceRegistryEntry]:
    if not mapping.primary_authoritative_source_ids:
        return []
    registry_by_id = {entry.source_id: entry for entry in registry_entries}
    resolved_entries: list[SourceRegistryEntry] = []
    for source_id in mapping.primary_authoritative_source_ids:
        entry = registry_by_id.get(source_id)
        if entry is None:
            return []
        resolved_entries.append(entry)
    return resolved_entries


def resolve_route_for_mapping(
    authority_entries: list[SourceRegistryEntry],
) -> str | None:
    candidate_routes = {
        usage_mode_to_route(entry.usage_mode)
        for entry in authority_entries
        if usage_mode_to_route(entry.usage_mode) is not None
    }
    if len(candidate_routes) == 1:
        return candidate_routes.pop()
    return None


def derive_top_level_confidence(subroutes: list[dict[str, str]]) -> str:
    confidences = {subroute["confidence"] for subroute in subroutes}
    if confidences == {"high"}:
        return "high"
    if "low" in confidences:
        return "low"
    return "medium"


def normalize_text(text: str) -> str:
    collapsed = re.sub(r"[^a-z0-9+.\- ]+", " ", text.lower())
    return re.sub(r"\s+", " ", collapsed).strip()


def _split_ids(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(";") if part.strip())


def _upsert_domain_match(
    matches: list[dict[str, str | int]],
    domain_id: str,
    knowledge_domain: str,
    *,
    confidence: str,
    boost: int,
) -> None:
    for match in matches:
        if match["domain_id"] == domain_id:
            match["match_count"] = max(int(match["match_count"]), boost)
            if confidence == "high":
                match["confidence"] = "high"
            return
    matches.append(
        {
            "domain_id": domain_id,
            "knowledge_domain": knowledge_domain,
            "confidence": confidence,
            "match_count": boost,
        }
    )


def _dedupe_domain_matches(matches: list[dict[str, str | int]]) -> list[dict[str, str | int]]:
    deduped: list[dict[str, str | int]] = []
    seen: set[str] = set()
    for match in matches:
        domain_id = str(match["domain_id"])
        if domain_id in seen:
            continue
        deduped.append(match)
        seen.add(domain_id)
    return deduped


def _looks_like_compatibility_question(normalized_question: str) -> bool:
    explicit_compatibility_language = (
        "compatible" in normalized_question
        or "compatibility" in normalized_question
        or "work with" in normalized_question
        or "works with" in normalized_question
    )
    product_pair_context = (
        "homecell" in normalized_question
        and "ve hybrid" in normalized_question
        and ("add " in normalized_question or "system" in normalized_question)
    )
    return (explicit_compatibility_language or product_pair_context) and (
        "homecell" in normalized_question or "ve hybrid" in normalized_question or "battery" in normalized_question
    )


def _looks_like_firmware_question(normalized_question: str) -> bool:
    return "firmware" in normalized_question or bool(re.search(r"\bfirmware\s+\d", normalized_question))


def _looks_like_warranty_question(normalized_question: str) -> bool:
    return "warranty" in normalized_question or "covered" in normalized_question


def _looks_like_third_party_question(normalized_question: str) -> bool:
    return "third-party" in normalized_question or "third party" in normalized_question or "unsupported" in normalized_question


def _looks_like_retrofit_question(normalized_question: str) -> bool:
    return "existing installation" in normalized_question or "add " in normalized_question


def _looks_like_metering_question(normalized_question: str) -> bool:
    return "pv-surplus charging" in normalized_question or "metering" in normalized_question or "meter data" in normalized_question


def _requests_definitive_unsupported_conclusion(question: str) -> bool:
    normalized = normalize_text(question)
    asks_for_definitive = any(
        phrase in normalized
        for phrase in ("definitely", "guaranteed", "remain under warranty", "will it work", "will definitely work")
    )
    unsupported_context = any(
        phrase in normalized
        for phrase in ("undocumented", "unsupported", "third-party controller", "third party controller")
    )
    return asks_for_definitive and unsupported_context


def _identify_missing_information(question: str, detected_domains: list[dict[str, str]]) -> set[str]:
    return set()


def _build_top_level_reason(*, subroutes: list[dict[str, str]], top_route: str) -> str:
    if top_route == "composite":
        involved_domains = ", ".join(subroute["knowledge_domain"] for subroute in subroutes)
        return (
            "The question spans multiple governed knowledge domains that resolve independently: "
            f"{involved_domains}."
        )
    return subroutes[0]["reason"]


def _top_level_execution(*, top_route: str, subroutes: list[dict[str, object]]) -> dict[str, object] | None:
    if top_route != "structured_lookup":
        return None
    for subroute in subroutes:
        execution = subroute.get("execution")
        if isinstance(execution, dict):
            return execution
    return None


def _aggregate_authoritative_source_ids(*, top_route: str, subroutes: list[dict[str, object]]) -> list[str]:
    if top_route not in {"rag", "composite"}:
        return []
    seen: set[str] = set()
    ordered_ids: list[str] = []
    for subroute in subroutes:
        for source_id in subroute.get("authoritative_source_ids", []):
            if source_id not in seen:
                seen.add(source_id)
                ordered_ids.append(source_id)
    return ordered_ids


def build_subroute(
    *,
    question: str,
    domain_match: dict[str, str],
    mapping: SourceOfTruthMapping,
    authority_entries: list[SourceRegistryEntry],
    route: str,
) -> dict[str, object]:
    subquestion = build_executable_subquestion(question=question, domain_id=domain_match["domain_id"])
    subroute: dict[str, object] = {
        "intent": subquestion,
        "domain": domain_match["domain_id"],
        "knowledge_domain": domain_match["knowledge_domain"],
        "route": route,
        "confidence": domain_match["confidence"],
        "reason": (
            f"{domain_match['knowledge_domain']} resolves to {','.join(mapping.primary_authoritative_source_ids)}, "
            f"which maps to route {route} under the configured source registry."
        ),
        "authoritative_source_ids": list(mapping.primary_authoritative_source_ids),
    }
    if route == "structured_lookup":
        execution = build_structured_execution_payload(question=question, authority_entries=authority_entries)
        subroute["execution"] = execution
        subroute["missing_information"] = list(execution["missing_information"])
    else:
        subroute["missing_information"] = []
    return subroute


def build_executable_subquestion(*, question: str, domain_id: str) -> str:
    normalized = normalize_text(question)
    inverter_model = parse_inverter_model(normalized)
    battery_model = parse_battery_model(normalized)
    firmware_version = parse_firmware_version(normalized)
    existing_installation = _looks_like_retrofit_question(normalized)
    third_party_context = _extract_third_party_context(normalized)

    if domain_id == "KD-002":
        if inverter_model and battery_model:
            return f"Is {battery_model} compatible with {inverter_model}?"
        if battery_model:
            return f"Is {battery_model} compatible with the inverter?"
        if inverter_model:
            return f"Which batteries are compatible with {inverter_model}?"
        return "Is this product combination compatible?"

    if domain_id == "KD-003":
        if inverter_model and battery_model:
            return f"What firmware version is required for {battery_model} with {inverter_model}?"
        if battery_model:
            return f"What firmware version is required to use {battery_model} with the inverter?"
        return "What firmware version is required for this configuration?"

    if domain_id == "KD-007":
        if inverter_model and battery_model and existing_installation:
            return f"What retrofit requirements apply when adding {battery_model} to an existing {inverter_model} installation?"
        if battery_model and existing_installation:
            return f"What retrofit requirements apply when adding {battery_model} to an existing installation?"
        if battery_model:
            return f"What retrofit requirements apply when adding {battery_model} to a system?"
        return "What retrofit requirements apply to this installation?"

    if domain_id == "KD-017":
        if third_party_context:
            return f"What warranty conditions apply to {third_party_context}?"
        if inverter_model and battery_model and existing_installation:
            return f"What warranty conditions apply when modifying an existing {inverter_model} system to add {battery_model}?"
        if battery_model and existing_installation:
            return f"What warranty conditions apply when adding {battery_model} to an existing installation?"
        return "What warranty conditions apply to this configuration?"

    if domain_id == "KD-020":
        if third_party_context:
            return f"What limitations apply to {third_party_context}?"
        return "What limitations apply to third-party equipment in this configuration?"

    if domain_id == "KD-009":
        if "pv-surplus charging" in normalized:
            return "What is required for PV-surplus charging with ChargeOne 11?"
        return "How does the system behave in this configuration?"

    if domain_id == "KD-010":
        if "pv-surplus charging" in normalized:
            return "What metering data is required for PV-surplus charging with ChargeOne 11?"
        return "What metering requirements apply to this configuration?"

    return question


def build_structured_execution_payload(
    *,
    question: str,
    authority_entries: list[SourceRegistryEntry],
) -> dict[str, object]:
    normalized = normalize_text(question)
    inverter_model = parse_inverter_model(normalized)
    battery_model = parse_battery_model(normalized)
    firmware_version = parse_firmware_version(normalized)
    region = authority_entries[0].region if len({entry.region for entry in authority_entries}) == 1 else None

    missing_information: list[str] = []
    if inverter_model is None:
        missing_information.append("inverter_model")
    if battery_model is None:
        missing_information.append("battery_model")
    if firmware_version is None:
        missing_information.append("firmware_version")
    if region is None:
        missing_information.append("region")

    return {
        "inverter_model": inverter_model,
        "battery_model": battery_model,
        "firmware_version": firmware_version,
        "region": region,
        "missing_information": missing_information,
    }


def parse_inverter_model(normalized_question: str) -> str | None:
    match = re.search(r"\bve hybrid\s+\d+\b", normalized_question)
    if not match:
        return None
    return " ".join(part.capitalize() if part != "ve" else "VE" for part in match.group(0).split())


def parse_battery_model(normalized_question: str) -> str | None:
    match = re.search(r"\bhomecell\s+\d+\b", normalized_question)
    if not match:
        return None
    return " ".join(part.capitalize() if part != "homecell" else "HomeCell" for part in match.group(0).split())


def parse_firmware_version(normalized_question: str) -> str | None:
    match = re.search(
        r"\bfirmware(?: version)?(?:\s+(?:is|of))?\s+(\d+(?:\.\d+){1,2}|\d+)\b",
        normalized_question,
    )
    if not match:
        return None
    return match.group(1)


def _apply_premise_vs_intent_precedence(
    normalized_question: str,
    matches: list[dict[str, str | int]],
) -> list[dict[str, str | int]]:
    if not _looks_like_document_premise_for_deterministic_conclusion(normalized_question):
        return matches

    deterministic_domains = {"KD-002", "KD-003"}
    supporting_premise_domains = {
        "KD-004",
        "KD-005",
        "KD-006",
        "KD-007",
        "KD-008",
        "KD-009",
        "KD-010",
        "KD-011",
        "KD-012",
    }
    matched_domain_ids = {str(match["domain_id"]) for match in matches}
    if not matched_domain_ids.intersection(deterministic_domains):
        return matches

    return [
        match
        for match in matches
        if str(match["domain_id"]) not in supporting_premise_domains
    ]


def _looks_like_document_premise_for_deterministic_conclusion(normalized_question: str) -> bool:
    mentions_narrative_source = any(
        phrase in normalized_question
        for phrase in (
            "guide says",
            "guide mentions",
            "document says",
            "document mentions",
            "installation guide",
            "retrofit guide",
        )
    )
    inferential_ask = any(
        phrase in normalized_question
        for phrase in (
            "that means",
            "must be",
            "so it must be",
            "so firmware",
            "right",
            "correct",
        )
    )
    asks_deterministic_conclusion = _looks_like_compatibility_question(normalized_question) or _looks_like_firmware_question(
        normalized_question
    )
    return mentions_narrative_source and inferential_ask and asks_deterministic_conclusion


def _extract_third_party_context(normalized_question: str) -> str | None:
    if "unauthorized third-party modification" in normalized_question:
        return "an unauthorized third-party modification"
    if "unauthorized modification" in normalized_question:
        return "an unauthorized modification"
    if "third-party controller" in normalized_question or "third party controller" in normalized_question:
        return "an undocumented or unsupported third-party controller"
    if "third-party equipment" in normalized_question or "third party equipment" in normalized_question:
        return "unsupported third-party equipment"
    if "third-party" in normalized_question or "third party" in normalized_question:
        return "an unsupported third-party configuration"
    return None
