"""Shared BM25 tokenization for literature indexing and online queries."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re


BM25_TOKENIZATION_VERSION = "process_numeric_tokens_v4_unicode_minus"

_UNICODE_MINUS_TRANSLATION = str.maketrans(
    {
        "\N{MINUS SIGN}": "-",
        "\N{FULLWIDTH HYPHEN-MINUS}": "-",
        "\N{SMALL HYPHEN-MINUS}": "-",
    }
)

# Split letters and numbers so strings such as ``573K``, ``T6`` and
# ``Mg-3.5Zn`` produce compatible lexical tokens on both the document and query
# sides.  Scientific notation is retained as one numeric value.
_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?|(?:\d+(?:\.\d*)?|\.\d+)"
    r"(?:[eE][+-]?\d+)?|[A-Za-z]+"
)
_KEEP_SINGLE_UPPER = set("BCFHKNOPSTUVWY")
_KEEP_SINGLE_LOWER = {"h", "m", "s"}


def _canonical_number(raw: str) -> str | None:
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == 0:
        return "0"
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def tokenize_bm25(text: str) -> list[str]:
    """Tokenize words plus canonical numeric values for BM25.

    Numeric tokens are prefixed with ``num_`` so ``573`` and ``573.0`` match
    without colliding with alphanumeric identifiers.  Digits attached to
    letters are also retained, which lets BM25 distinguish process labels such
    as ``T6`` and ``T4``.  Retaining selected single-letter uppercase tokens
    preserves common element symbols, units and temper labels that the
    previous tokenizer silently discarded.
    """
    tokens: list[str] = []
    normalized_text = str(text or "").translate(_UNICODE_MINUS_TRANSLATION)
    for match in _TOKEN_PATTERN.finditer(normalized_text):
        raw = match.group(0)
        if raw[0].isdigit() or raw[0] in {".", "+", "-"}:
            number = _canonical_number(raw)
            if number is not None:
                tokens.append(f"num_{number}")
            continue

        if len(raw) == 1:
            if raw.isupper() and raw in _KEEP_SINGLE_UPPER:
                tokens.append(raw.lower())
            elif raw in _KEEP_SINGLE_LOWER:
                tokens.append(raw)
            continue
        tokens.append(raw.lower())
    return tokens
