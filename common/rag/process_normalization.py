"""Deterministic process-field normalization for RAG queries.

The source datasets intentionally keep their original column names and values.
This module adds search-only aliases and physically equivalent unit forms; it
never replaces the raw value used by the surrogate model or the LLM reviewer.
"""

from __future__ import annotations

import math
import re


PROCESS_NORMALIZATION_VERSION = (
    "process_alias_units_v3_binary_state_condition_codes"
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        value = " ".join(str(value).split()).strip()
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            result.append(value)
    return tuple(result)


def humanize_process_field(field_name: str) -> str:
    """Split snake/kebab/CamelCase names while retaining the raw abbreviations."""
    text = _CAMEL_BOUNDARY.sub(" ", str(field_name or "").strip())
    text = re.sub(r"[_\-]+", " ", text)
    return " ".join(text.split())


def _field_key(field_name: str) -> str:
    return _NON_ALNUM.sub("", str(field_name or "").casefold())


def _condition_code_aliases(text: str) -> tuple[str, ...]:
    """Expand common, unambiguous processing-condition abbreviations."""
    raw = str(text or "").strip()
    key = _field_key(raw)
    aliases: list[str] = []

    exact = {
        "st": ("solution treatment", "solution treated"),
        "ac": ("air cooling", "air cooled"),
        "fc": ("furnace cooling", "furnace cooled"),
        "wq": ("water quenching", "water quenched"),
        "pm": ("powder metallurgy",),
        "pmfc": (
            "powder metallurgy followed by furnace cooling",
            "powder metallurgy processed and furnace cooled",
        ),
        "ascast": ("as cast",),
        "stwq": (
            "solution treatment followed by water quenching",
            "solution treated and water quenched",
        ),
        "stac": (
            "solution treatment followed by air cooling",
            "solution treated and air cooled",
        ),
        "stfc": (
            "solution treatment followed by furnace cooling",
            "solution treated and furnace cooled",
        ),
    }
    aliases.extend(exact.get(key, ()))
    return _dedupe(aliases)


def process_field_aliases(field_name: str) -> tuple[str, ...]:
    """Return deterministic semantic aliases for a process/test-condition field.

    Aliases are deliberately conservative.  In particular, a temperature field
    without a unit receives semantic aliases but no Celsius/Kelvin conversion.
    """
    raw = str(field_name or "").strip()
    readable = humanize_process_field(raw)
    key = _field_key(raw)
    aliases: list[str] = [readable]

    if "satemp" in key:
        aliases.extend(
            ["solution annealing temperature", "solution treatment temperature"]
        )
    elif key.startswith("tss"):
        aliases.append(
            "solution treatment time" if _has_hour_or_minute_unit(raw) else
            "solution treatment temperature"
        )
    elif key.startswith("tag"):
        if _has_hour_or_minute_unit(raw):
            aliases.extend(["aging time", "ageing time"])
        else:
            aliases.extend(["aging temperature", "ageing temperature"])

    if "ageingtemp" in key or "agingtemp" in key:
        aliases.extend(["aging temperature", "ageing temperature"])
    if "ageingtime" in key or "agingtime" in key:
        aliases.extend(["aging time", "ageing time"])
    if key in {"aging", "ageing"}:
        aliases.extend(["aging", "ageing"])
    if "ageingtype" in key or "agingtype" in key:
        aliases.extend(["aging type", "ageing type"])
    if "manufacturingmethod" in key:
        aliases.append("manufacturing method")
    if "heattreatmentmedium" in key:
        aliases.append("heat treatment medium")
    if "secondarythermomechanicalprocess" in key:
        aliases.extend(
            ["secondary thermo-mechanical process", "secondary thermomechanical process"]
        )
    if key.startswith("crreduction") or "coldreduction" in key:
        aliases.extend(["cold reduction", "cold rolling reduction"])
    if "strainrate1pers" in key:
        aliases.extend(["strain rate per second", "strain rate s^-1"])

    # Generic Temp -> temperature expansion, including compounds such as
    # ExtrusionTemp and PreheatTemp.
    expanded = re.sub(r"\btemp\b", "temperature", readable, flags=re.IGNORECASE)
    if expanded.casefold() != readable.casefold():
        aliases.append(expanded)

    if "mmin" in key:
        aliases.extend(["m/min", "metre per minute", "meter per minute"])
    if key.endswith("pct") or "percent" in key:
        aliases.extend(["percent", "percentage"])
    if "1pers" in key:
        aliases.extend(["per second", "s^-1"])

    if key.startswith("cond"):
        condition = re.sub(r"^cond[_\- ]*", "", raw, flags=re.IGNORECASE)
        aliases.extend(_condition_code_aliases(condition))

    return _dedupe(aliases)


def _has_kelvin_unit(field_name: str) -> bool:
    return bool(re.search(r"(?:^|[_\- ])k$", str(field_name), flags=re.IGNORECASE))


def _has_celsius_unit(field_name: str) -> bool:
    return bool(
        re.search(
            r"(?:^|[_\- ])(?:c|degc|celsius)$",
            str(field_name),
            flags=re.IGNORECASE,
        )
    )


def _has_hour_unit(field_name: str) -> bool:
    return bool(re.search(r"(?:^|[_\- ])(?:h|hr|hour)$", str(field_name), re.I))


def _has_minute_unit(field_name: str) -> bool:
    return bool(re.search(r"(?:^|[_\- ])(?:min|minute)$", str(field_name), re.I))


def _has_hour_or_minute_unit(field_name: str) -> bool:
    return _has_hour_unit(field_name) or _has_minute_unit(field_name)


def _numeric(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _number(value: float, digits: int = 6) -> str:
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    if float(rounded).is_integer():
        return str(int(rounded))
    return f"{rounded:.{digits}f}".rstrip("0").rstrip(".")


def _converted_temperature(value: float, *, kelvin_input: bool) -> list[str]:
    if kelvin_input:
        converted = value - 273.15
        exact = f"{_number(converted, 2)} °C"
        integer = f"approximately {round(converted):d} °C"
    else:
        converted = value + 273.15
        exact = f"{_number(converted, 2)} K"
        integer = f"approximately {round(converted):d} K"
    values = [exact]
    if abs(converted - round(converted)) > 0.01:
        values.append(integer)
    return values


def process_value_equivalents(field_name: str, value: object) -> tuple[str, ...]:
    """Return unit-labelled and equivalent search forms for a raw value."""
    raw_text = str(value).strip()
    number = _numeric(value)
    key = _field_key(field_name)
    values: list[str] = [raw_text]

    if number is not None:
        if _has_kelvin_unit(field_name):
            values.append(f"{_number(number)} K")
            values.extend(_converted_temperature(number, kelvin_input=True))
        elif _has_celsius_unit(field_name):
            values.append(f"{_number(number)} °C")
            values.extend(_converted_temperature(number, kelvin_input=False))
        elif _has_hour_unit(field_name):
            values.extend(
                [f"{_number(number)} h", f"{_number(number * 60.0)} min"]
            )
        elif _has_minute_unit(field_name):
            values.extend(
                [f"{_number(number)} min", f"{_number(number / 60.0)} h"]
            )
        elif "mmin" in key:
            values.extend(
                [
                    f"{_number(number)} m/min",
                    f"{_number(number)} metre per minute",
                ]
            )
        elif "1pers" in key:
            values.extend(
                [f"{_number(number)} s^-1", f"{_number(number)} per second"]
            )
        elif key.endswith("pct") or "percent" in key:
            values.extend([f"{_number(number)} %", f"{_number(number)} percent"])
    else:
        values.extend(_condition_code_aliases(raw_text))

    return _dedupe(values)


def render_process_condition(
    field_name: str,
    value: object,
    *,
    indicator_state: str | None = None,
) -> str:
    """Render raw provenance plus deterministic aliases/equivalent values.

    Inactive one-hot conditions deliberately return an empty string so their
    names cannot contaminate either dense or sparse retrieval.
    """
    if indicator_state == "inactive":
        return ""

    raw_value = str(value).strip()
    readable = humanize_process_field(field_name)
    if indicator_state == "active":
        raw_clause = f"{readable} active {raw_value}"
    elif indicator_state == "applied":
        raw_clause = f"{readable} applied enabled yes {raw_value}"
    elif indicator_state == "not_applied":
        raw_clause = f"{readable} not applied disabled no {raw_value}"
    else:
        raw_clause = f"{readable} {raw_value}"

    aliases = process_field_aliases(field_name)
    equivalents = process_value_equivalents(field_name, value)
    parts = [raw_clause, f"raw {field_name}={raw_value}"]
    parts.extend(aliases)
    parts.extend(equivalents)
    return " | ".join(_dedupe(parts))


__all__ = [
    "PROCESS_NORMALIZATION_VERSION",
    "humanize_process_field",
    "process_field_aliases",
    "process_value_equivalents",
    "render_process_condition",
]
