"""Identity and integrity helpers shared by the RAG indexing pipeline.

Chunk identities deliberately depend on the exact embedded payload rather
than section-local counters.  This prevents repeated section headings from
colliding and makes any title/section/text change invalidate the old dense
embedding instead of silently reusing it.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re
from typing import Iterable, Mapping, Sequence


CHUNK_ID_VERSION = "chunk_content_sha256_v1"
CHUNK_ID_PREFIX = "cidv1_"
INDEX_ID_SCHEMA_VERSION = "ordered_chunk_ids_v1"
BM25_DOCUMENT_VERSION = "title_section_text_v1"

_CHUNK_ID_RE = re.compile(r"^cidv1_[0-9a-f]{32}$")
_DROP_PATTERNS = (
    re.compile(r"Downloaded From:|Terms of Use:.*asme", re.IGNORECASE),
    re.compile(r"Contributed by the .+ Division for presentation"),
)


def _identity_key(record: Mapping[str, object]) -> tuple[str, str, str, str]:
    """Return the exact content key used to count identical duplicates."""
    return (
        str(record.get("source_file") or ""),
        str(record.get("paper_title") or ""),
        str(record.get("section") or "body"),
        str(record.get("text") or ""),
    )


def make_chunk_id(
    record: Mapping[str, object], *, duplicate_ordinal: int = 0
) -> str:
    """Create a versioned, content-aware chunk ID.

    ``duplicate_ordinal`` only disambiguates byte-identical chunks from the
    same paper and section.  It does not depend on the old ``chunk_idx`` and
    therefore cannot repeat merely because a section-local counter resets.
    """
    if duplicate_ordinal < 0:
        raise ValueError("duplicate_ordinal must be non-negative")
    source_file, paper_title, section, text = _identity_key(record)
    payload = json.dumps(
        {
            "version": CHUNK_ID_VERSION,
            "source_file": source_file,
            "paper_title": paper_title,
            "section": section,
            "text": text,
            "duplicate_ordinal": int(duplicate_ordinal),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return CHUNK_ID_PREFIX + hashlib.sha256(payload).hexdigest()[:32]


def assign_chunk_ids(
    records: Sequence[Mapping[str, object]], *, validate_existing: bool = True
) -> list[dict]:
    """Return record copies with deterministic unique current-version IDs."""
    seen_content: dict[tuple[str, str, str, str], int] = defaultdict(int)
    output: list[dict] = []
    for row, record in enumerate(records):
        key = _identity_key(record)
        ordinal = seen_content[key]
        seen_content[key] += 1
        expected = make_chunk_id(record, duplicate_ordinal=ordinal)
        existing = record.get("chunk_id")
        if validate_existing and existing is not None and existing != expected:
            raise RuntimeError(
                f"row {row} has stale/invalid chunk_id {existing!r}; "
                f"expected {expected!r} ({CHUNK_ID_VERSION})"
            )
        updated = dict(record)
        updated["chunk_id"] = expected
        output.append(updated)
    ids = [str(record["chunk_id"]) for record in output]
    if len(ids) != len(set(ids)):
        duplicates = len(ids) - len(set(ids))
        raise RuntimeError(f"generated {duplicates} duplicate chunk_id rows")
    return output


def is_current_chunk_id(value: object) -> bool:
    return isinstance(value, str) and bool(_CHUNK_ID_RE.fullmatch(value))


def validate_current_chunk_records(
    records: Sequence[Mapping[str, object]], *, context: str = "chunk records"
) -> list[str]:
    """Validate version, content binding, and uniqueness of ordered records."""
    expected = assign_chunk_ids(records, validate_existing=False) if records else []
    ids: list[str] = []
    for row, (record, canonical) in enumerate(zip(records, expected)):
        chunk_id = record.get("chunk_id")
        if not is_current_chunk_id(chunk_id):
            raise RuntimeError(
                f"{context} row {row} uses a legacy/missing chunk_id: {chunk_id!r}; "
                f"expected {CHUNK_ID_VERSION}"
            )
        if chunk_id != canonical["chunk_id"]:
            raise RuntimeError(
                f"{context} row {row} chunk_id is not bound to its current "
                "title/section/text payload"
            )
        ids.append(str(chunk_id))
    if len(ids) != len(set(ids)):
        duplicates = len(ids) - len(set(ids))
        raise RuntimeError(f"{context} contains {duplicates} duplicate chunk_id rows")
    return ids


def validate_current_id_list(ids: Sequence[object], *, context: str) -> list[str]:
    """Validate an ID-only list when record contents are unavailable."""
    normalized: list[str] = []
    for row, chunk_id in enumerate(ids):
        if not is_current_chunk_id(chunk_id):
            raise RuntimeError(
                f"{context} row {row} uses a legacy/missing chunk_id: {chunk_id!r}"
            )
        normalized.append(str(chunk_id))
    if len(normalized) != len(set(normalized)):
        raise RuntimeError(f"{context} contains duplicate chunk IDs")
    return normalized


def ordered_id_fingerprint(ids: Iterable[str]) -> str:
    """Hash both ID values and row boundaries/order for cross-artifact checks."""
    digest = hashlib.sha256()
    digest.update((INDEX_ID_SCHEMA_VERSION + "\0").encode("ascii"))
    for chunk_id in ids:
        encoded = str(chunk_id).encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def embedding_filter_reason(record: Mapping[str, object]) -> str | None:
    """Return the stable pre-embedding filter reason, or ``None`` if eligible."""
    text = str(record.get("text") or "")
    word_count = int(record.get("word_count") or 0)
    if word_count < 60:
        return "too_short"
    for pattern in _DROP_PATTERNS:
        if pattern.search(text):
            return "publisher_boilerplate"
    junk_runs = len(re.findall(r"[\^*]{3,}", text)) + len(
        re.findall(r"\s{20,}", text)
    )
    if junk_runs > 5 and word_count < 200:
        return "layout_garbage"
    return None


def eligible_chunk_records(
    records: Sequence[Mapping[str, object]], *, validate_existing: bool = True
) -> tuple[list[dict], dict[str, int]]:
    """Assign IDs and return only records eligible for dense embedding."""
    identified = assign_chunk_ids(records, validate_existing=validate_existing)
    eligible: list[dict] = []
    skipped: dict[str, int] = {}
    for record in identified:
        reason = embedding_filter_reason(record)
        if reason:
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            eligible.append(record)
    return eligible, skipped
