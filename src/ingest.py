from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "consulting" / "knowledge_source_registry.csv"
DEFAULT_SECTION_MAX_CHARS = 2000


@dataclass(frozen=True)
class SourceRegistryEntry:
    source_id: str
    source_name: str
    source_path: str
    source_type: str
    owner: str
    authoritative_for: str
    not_authoritative_for: str
    region: str
    required_approval_status: str
    required_lifecycle_status: str
    usage_mode: str
    risk_level: str
    notes: str


def ingest_documents(
    *,
    project_root: Path | str = PROJECT_ROOT,
    as_of_date: date | None = None,
    max_chunk_chars: int = DEFAULT_SECTION_MAX_CHARS,
) -> list[dict[str, str]]:
    root = Path(project_root)
    registry_entries = load_source_registry(root / "consulting" / "knowledge_source_registry.csv")
    chunks: list[dict[str, str]] = []

    for entry in registry_entries:
        if entry.source_type != "document":
            continue
        if not entry.source_path.startswith("data/documents/"):
            continue

        document_path = root / entry.source_path
        if not document_path.exists():
            raise FileNotFoundError(f"Registered document source is missing: {document_path}")

        document = load_markdown_document(document_path)
        if not is_source_eligible(entry, document["metadata"], as_of_date=as_of_date):
            continue

        chunks.extend(
            chunk_document(
                entry=entry,
                metadata=document["metadata"],
                body=document["body"],
                as_of_date=as_of_date,
                max_chunk_chars=max_chunk_chars,
            )
        )

    return chunks


def load_source_registry(registry_path: Path | str = DEFAULT_REGISTRY_PATH) -> list[SourceRegistryEntry]:
    path = Path(registry_path)
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        return [
            SourceRegistryEntry(
                source_id=row["source_id"],
                source_name=row["source_name"],
                source_path=row["source_path"],
                source_type=row["source_type"],
                owner=row["owner"],
                authoritative_for=row["authoritative_for"],
                not_authoritative_for=row["not_authoritative_for"],
                region=row["region"],
                required_approval_status=row["required_approval_status"],
                required_lifecycle_status=row["required_lifecycle_status"],
                usage_mode=row["usage_mode"],
                risk_level=row["risk_level"],
                notes=row["notes"],
            )
            for row in reader
        ]


def load_markdown_document(document_path: Path | str) -> dict[str, dict[str, str] | str]:
    raw_text = Path(document_path).read_text(encoding="utf-8")
    metadata, body = parse_frontmatter(raw_text)
    return {
        "metadata": metadata,
        "body": body,
    }


def parse_frontmatter(markdown_text: str) -> tuple[dict[str, str], str]:
    if not markdown_text.startswith("---"):
        return {}, markdown_text.strip()

    lines = markdown_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, markdown_text.strip()

    closing_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        return {}, markdown_text.strip()

    frontmatter_text = "\n".join(lines[1:closing_index])
    parsed = yaml.safe_load(frontmatter_text) or {}
    metadata = {str(key): _normalize_metadata_value(value) for key, value in parsed.items()}
    body = "\n".join(lines[closing_index + 1 :]).strip()
    return metadata, body


def _normalize_metadata_value(value: object) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def is_source_eligible(
    entry: SourceRegistryEntry,
    metadata: dict[str, str],
    *,
    as_of_date: date | None = None,
) -> bool:
    approval_status = metadata.get("status", "").strip().lower()
    lifecycle_status = derive_lifecycle_status(metadata).lower()
    region = metadata.get("region", "").strip()

    if approval_status != entry.required_approval_status.strip().lower():
        return False
    if lifecycle_status != entry.required_lifecycle_status.strip().lower():
        return False
    if region != entry.region:
        return False
    if not is_effective_on(metadata, as_of_date=as_of_date):
        return False
    return True


def derive_lifecycle_status(metadata: dict[str, str]) -> str:
    explicit_status = metadata.get("lifecycle_status")
    if explicit_status:
        return explicit_status.strip()
    return "unknown"


def is_effective_on(metadata: dict[str, str], *, as_of_date: date | None = None) -> bool:
    effective_date_text = metadata.get("effective_date", "").strip()
    if not effective_date_text:
        return True

    effective_date = date.fromisoformat(effective_date_text)
    comparison_date = as_of_date or date.today()
    return effective_date <= comparison_date


def chunk_document(
    *,
    entry: SourceRegistryEntry,
    metadata: dict[str, str],
    body: str,
    as_of_date: date | None = None,
    max_chunk_chars: int = DEFAULT_SECTION_MAX_CHARS,
) -> list[dict[str, str]]:
    title = metadata.get("title") or extract_title_from_body(body) or entry.source_name
    lifecycle_status = derive_lifecycle_status(metadata)
    section_chunks = split_into_sections(body, title=title)

    chunks: list[dict[str, str]] = []
    for section_name, section_text in section_chunks:
        for chunk_text in split_oversized_section(section_text, max_chunk_chars=max_chunk_chars):
            chunks.append(
                {
                    "source_id": entry.source_id,
                    "source_name": entry.source_name,
                    "source_path": entry.source_path,
                    "title": title,
                    "section": section_name,
                    "region": metadata.get("region", entry.region),
                    "approval_status": metadata.get("status", ""),
                    "lifecycle_status": lifecycle_status,
                    "effective_date": metadata.get("effective_date", ""),
                    "text": chunk_text.strip(),
                }
            )
    return chunks


def extract_title_from_body(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def split_into_sections(body: str, *, title: str) -> list[tuple[str, str]]:
    lines = body.splitlines()
    sections: list[tuple[str, str]] = []
    current_section = title
    current_lines: list[str] = []

    for raw_line in lines:
        stripped = raw_line.strip()

        if stripped.startswith("# "):
            continue

        if stripped.startswith("## "):
            if current_lines:
                sections.append((current_section, "\n".join(current_lines).strip()))
            current_section = stripped[3:].strip()
            current_lines = [raw_line]
            continue

        current_lines.append(raw_line)

    if current_lines:
        sections.append((current_section, "\n".join(current_lines).strip()))

    return [section for section in sections if section[1].strip()]


def split_oversized_section(section_text: str, *, max_chunk_chars: int) -> list[str]:
    normalized = section_text.strip()
    if len(normalized) <= max_chunk_chars:
        return [normalized]

    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []

    for paragraph in paragraphs:
        candidate_parts = current_parts + [paragraph]
        candidate_text = "\n\n".join(candidate_parts)
        if current_parts and len(candidate_text) > max_chunk_chars:
            chunks.append("\n\n".join(current_parts).strip())
            current_parts = [paragraph]
            continue

        if len(paragraph) > max_chunk_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
            chunks.extend(split_long_paragraph(paragraph, max_chunk_chars=max_chunk_chars))
            continue

        current_parts = candidate_parts

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return chunks


def split_long_paragraph(paragraph: str, *, max_chunk_chars: int) -> list[str]:
    lines = paragraph.splitlines()
    chunks: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        candidate_lines = current_lines + [line]
        candidate_text = "\n".join(candidate_lines)
        if current_lines and len(candidate_text) > max_chunk_chars:
            chunks.append("\n".join(current_lines).strip())
            current_lines = [line]
            continue
        current_lines = candidate_lines

    if current_lines:
        chunks.append("\n".join(current_lines).strip())

    return chunks
