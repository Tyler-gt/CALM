"""Utilities for OpenAI-compatible chat completion responses."""

from __future__ import annotations

from typing import Any


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
