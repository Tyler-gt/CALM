"""Utilities for OpenAI-compatible chat completion responses."""

from __future__ import annotations

import json
import math
from typing import Any


def parse_json_object(text: str) -> dict:
    """Read exactly one JSON object, rejecting non-finite numeric values."""
    def reject_constant(value: str):
        raise ValueError(f"Non-finite JSON value: {value}")

    def finite_float(value: str) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON number is not finite")
        return number

    result = json.loads(
        text,
        parse_constant=reject_constant,
        parse_float=finite_float,
    )
    if not isinstance(result, dict):
        raise ValueError("Response must be a JSON object")
    return result


def extract_message_text(message: Any) -> str:
    """Return text from standard content or USTC reasoning_content fallback."""
    content = getattr(message, "content", None)
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        joined = "\n".join(parts).strip()
        if joined:
            return joined

    reasoning = getattr(message, "reasoning_content", None)
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning

    extra = getattr(message, "model_extra", None)
    if isinstance(extra, dict):
        reasoning = extra.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning

    if hasattr(message, "model_dump"):
        try:
            dumped = message.model_dump()
        except Exception:
            dumped = {}
        if isinstance(dumped, dict):
            for key in ("content", "reasoning_content"):
                value = dumped.get(key)
                if isinstance(value, str) and value.strip():
                    return value

    return "" if content is None else str(content)
