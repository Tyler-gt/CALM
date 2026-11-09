"""Independent-quota API key loading and asynchronous scheduling.

Secrets are read only from environment variables. Logs and public state expose
stable aliases (``key_1``, ``key_2``), never key values or fingerprints.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)
from common.llm.config import (
    DEFAULT_API_BASE_URL,
    get_api_base_url,
    validate_api_base_url,
)


USTC_BASE_URL = DEFAULT_API_BASE_URL  # Compatibility for existing callers.


class RetryableAPIKeyError(RuntimeError):
    """One key failed transiently or was disabled while alternatives remain."""


class NoAvailableAPIKeyError(RuntimeError):
    """Every configured key is permanently disabled."""


def load_ustc_api_keys(
    environ: Mapping[str, str] | None = None,
) -> list[str]:
    """Load CALM keys, falling back to the legacy USTC variable family."""
    env = os.environ if environ is None else environ
    keys = _load_api_key_family(env, "CALM")
    return keys or _load_api_key_family(env, "USTC_DEEPSEEK")


def _load_api_key_family(env: Mapping[str, str], prefix: str) -> list[str]:
    candidates: list[str] = []

    for index in range(1, 33):
        value = str(env.get(f"{prefix}_API_KEY_{index}", "")).strip()
        if value:
            candidates.append(value)

    packed = str(env.get(f"{prefix}_API_KEYS", "")).strip()
    if packed:
        if packed.startswith("["):
            try:
                decoded = json.loads(packed)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{prefix}_API_KEYS must be valid JSON or CSV") from exc
            if not isinstance(decoded, list):
                raise ValueError(f"{prefix}_API_KEYS JSON value must be a list")
            candidates.extend(str(value).strip() for value in decoded)
        else:
            candidates.extend(part.strip() for part in packed.replace(";", ",").split(","))

    legacy = str(env.get(f"{prefix}_API_KEY", "")).strip()
    # The legacy variable is a fallback only. When numbered/list variables are
    # present, an unrelated old legacy key must not silently become key_3.
    if legacy and not candidates:
        candidates.append(legacy)

    keys: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        if value and value not in seen:
            keys.append(value)
            seen.add(value)
    return keys


def require_ustc_api_keys(min_count: int = 1) -> list[str]:
    keys = load_ustc_api_keys()
    if len(keys) < int(min_count):
        raise RuntimeError(
            f"Expected at least {int(min_count)} independent API keys; "
            "configure CALM_API_KEY or numbered CALM_API_KEY_1..N entries in "
            "common/llm/.env"
        )
    return keys


@dataclass
class _KeyState:
    alias: str
    api_key: str = field(repr=False)
    in_flight: int = 0
    cooldown_until: float = 0.0
    consecutive_failures: int = 0
    disabled: bool = False
    last_selected: int = 0


class AsyncUSTCKeyPool:
    """Least-loaded scheduler for API keys with independent quotas."""

    def __init__(
        self,
        api_keys: list[str],
        *,
        key_aliases: list[str] | None = None,
        per_key_concurrency: int = 4,
        base_url: str | None = None,
        cooldown_max_seconds: float = 60.0,
        client_factory: Callable[[str], object] | None = None,
    ) -> None:
        unique_keys = list(dict.fromkeys(key for key in api_keys if key))
        if not unique_keys:
            raise ValueError("AsyncUSTCKeyPool requires at least one API key")
        if key_aliases is None:
            aliases = [f"key_{index + 1}" for index in range(len(unique_keys))]
        else:
            aliases = [str(alias).strip() for alias in key_aliases]
            if (
                len(aliases) != len(unique_keys)
                or any(not alias for alias in aliases)
                or len(set(aliases)) != len(aliases)
            ):
                raise ValueError(
                    "key_aliases must be unique non-empty labels matching api_keys"
                )
        self._states = [
            _KeyState(alias=alias, api_key=key)
            for alias, key in zip(aliases, unique_keys)
        ]
        self.per_key_concurrency = max(1, int(per_key_concurrency))
        self.base_url = validate_api_base_url(
            get_api_base_url() if base_url is None else base_url
        )
        self.cooldown_max_seconds = max(1.0, float(cooldown_max_seconds))
        self._client_factory = client_factory or (
            lambda key: AsyncOpenAI(
                api_key=key,
                base_url=self.base_url,
                max_retries=0,
            )
        )
        self._lock = threading.Lock()
        self._selection_counter = 0
        self._sessions: dict[object, tuple[list[object], int]] = {}

    @property
    def key_count(self) -> int:
        return len(self._states)

    def public_state(self) -> list[dict]:
        """Return audit state without exposing any secret material."""
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "alias": state.alias,
                    "in_flight": state.in_flight,
                    "cooldown_remaining_seconds": max(
                        0.0, state.cooldown_until - now
                    ),
                    "disabled": state.disabled,
                    "consecutive_failures": state.consecutive_failures,
                }
                for state in self._states
            ]

    async def open_session(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            existing = self._sessions.get(loop)
            if existing is not None:
                clients, references = existing
                self._sessions[loop] = (clients, references + 1)
                return
            clients = [self._client_factory(state.api_key) for state in self._states]
            self._sessions[loop] = (clients, 1)

    async def close_session(self) -> None:
        loop = asyncio.get_running_loop()
        clients: list[object] = []
        with self._lock:
            existing = self._sessions.get(loop)
            if existing is None:
                return
            clients, references = existing
            if references > 1:
                self._sessions[loop] = (clients, references - 1)
                return
            self._sessions.pop(loop, None)
        for client in clients:
            close = getattr(client, "close", None)
            if close is not None:
                try:
                    result = close()
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    # Transport cleanup must not mask the scoring result/error.
                    pass

    async def _clients_for_current_loop(self) -> list[object]:
        loop = asyncio.get_running_loop()
        with self._lock:
            existing = self._sessions.get(loop)
        if existing is None:
            await self.open_session()
            with self._lock:
                existing = self._sessions[loop]
        return existing[0]

    async def _acquire_state(self) -> tuple[int, _KeyState]:
        while True:
            now = time.monotonic()
            with self._lock:
                enabled = [state for state in self._states if not state.disabled]
                if not enabled:
                    raise NoAvailableAPIKeyError(
                        "All configured USTC API keys are disabled"
                    )
                candidates = [
                    (index, state)
                    for index, state in enumerate(self._states)
                    if not state.disabled
                    and state.cooldown_until <= now
                    and state.in_flight < self.per_key_concurrency
                ]
                if candidates:
                    index, state = min(
                        candidates,
                        key=lambda pair: (
                            pair[1].in_flight,
                            pair[1].last_selected,
                        ),
                    )
                    self._selection_counter += 1
                    state.last_selected = self._selection_counter
                    state.in_flight += 1
                    return index, state

                cooldown_waits = [
                    max(0.0, state.cooldown_until - now)
                    for state in enabled
                    if state.in_flight < self.per_key_concurrency
                ]
            wait_seconds = min(cooldown_waits) if cooldown_waits else 0.05
            await asyncio.sleep(max(0.01, min(wait_seconds, 0.25)))

    def _release_state(self, state: _KeyState) -> None:
        with self._lock:
            state.in_flight = max(0, state.in_flight - 1)

    @staticmethod
    def _retry_after_seconds(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            return None

    def _mark_success(self, state: _KeyState) -> None:
        with self._lock:
            state.consecutive_failures = 0
            state.cooldown_until = 0.0

    def _mark_transient_failure(self, state: _KeyState, exc: Exception) -> float:
        with self._lock:
            state.consecutive_failures += 1
            retry_after = self._retry_after_seconds(exc)
            if retry_after is None:
                retry_after = min(
                    self.cooldown_max_seconds,
                    2.0 ** min(state.consecutive_failures, 6),
                )
            cooldown = min(self.cooldown_max_seconds, max(1.0, retry_after))
            state.cooldown_until = max(
                state.cooldown_until,
                time.monotonic() + cooldown,
            )
            return cooldown

    def _disable_key(self, state: _KeyState) -> bool:
        with self._lock:
            state.disabled = True
            return any(not item.disabled for item in self._states)

    @staticmethod
    def _is_transient_model_route_error(exc: APIStatusError) -> bool:
        """Recognize LiteLLM's intermittent registered-model routing miss."""
        status = int(getattr(exc, "status_code", 0) or 0)
        message = str(exc).lower()
        return (
            status == 404
            and "no fallback model group found" in message
            and "litellm.notfounderror" in message
        )

    @classmethod
    def is_retryable_status_error(cls, exc: APIStatusError) -> bool:
        """Share HTTP retry classification with callers of the pool."""
        status = int(getattr(exc, "status_code", 0) or 0)
        return (
            status in {408, 409, 425, 429}
            or status >= 500
            or cls._is_transient_model_route_error(exc)
        )

    async def create_chat_completion(self, **kwargs):
        """Execute one request and return ``(response, key_alias)``."""
        clients = await self._clients_for_current_loop()
        index, state = await self._acquire_state()
        try:
            response = await clients[index].chat.completions.create(**kwargs)
        except (AuthenticationError, PermissionDeniedError) as exc:
            alternatives_remain = self._disable_key(state)
            if alternatives_remain:
                raise RetryableAPIKeyError(
                    f"{state.alias} was disabled; retrying with another key"
                ) from exc
            raise NoAvailableAPIKeyError(
                "All configured USTC API keys failed authentication/permission checks"
            ) from exc
        except (
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
            InternalServerError,
        ) as exc:
            cooldown = self._mark_transient_failure(state, exc)
            raise RetryableAPIKeyError(
                f"{state.alias} cooling down for {cooldown:.1f}s after "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        except APIStatusError as exc:
            status = int(getattr(exc, "status_code", 0) or 0)
            if self.is_retryable_status_error(exc):
                cooldown = self._mark_transient_failure(state, exc)
                raise RetryableAPIKeyError(
                    f"{state.alias} cooling down for {cooldown:.1f}s after "
                    f"HTTP {status}: {exc}"
                ) from exc
            raise
        else:
            self._mark_success(state)
            return response, state.alias
        finally:
            self._release_state(state)
