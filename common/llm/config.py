"""Shared settings for OpenAI-compatible chat-completions services."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv


DEFAULT_API_BASE_URL = "https://api.example.com/v1"
DEFAULT_LLM_MODEL = "deepseek-v4-flash-ascend"

# Read only this checkout's file. Existing shell variables always take priority.
load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def validate_api_base_url(value: str) -> str:
    """Keep the endpoint safe to include in run and cache identities."""
    value = str(value).strip().rstrip("/")
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme in {"http", "https"}
            and bool(parts.hostname)
            and parts.username is None
            and parts.password is None
            and not parts.query
            and not parts.fragment
            and "?" not in value
            and "#" not in value
            and not any(char.isspace() for char in value)
        )
        parts.port  # Reject malformed port numbers without printing the URL.
    except ValueError:
        valid = False
    if not valid:
        raise ValueError(
            "CALM_API_BASE_URL must be an http(s) URL without credentials, "
            "query parameters, or fragments"
        )
    return value


def get_api_base_url() -> str:
    value = (
        os.environ.get("CALM_API_BASE_URL")
        or os.environ.get("USTC_DEEPSEEK_API_BASE_URL")
        or os.environ.get("USTC_DEEPSEEK_BASE_URL")
        or DEFAULT_API_BASE_URL
    )
    return validate_api_base_url(value)


def get_llm_model() -> str:
    value = (
        os.environ.get("CALM_LLM_MODEL")
        or os.environ.get("USTC_DEEPSEEK_MODEL")
        or DEFAULT_LLM_MODEL
    )
    model = str(value).strip()
    if not model:
        raise ValueError("CALM_LLM_MODEL must be a non-empty model identifier")
    return model
