from __future__ import annotations

import time
from typing import Any

from llm_eval_harness.contracts import StageTrace


def build_stage_trace(
    stage: str,
    model: str | None = None,
    latency_seconds: float = 0.0,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    output_length_chars: int | None = None,
    validation_flags: dict[str, bool] | None = None,
    error: str | None = None,
) -> StageTrace:
    return StageTrace(
        stage=stage,
        model=model,
        latency_seconds=latency_seconds,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        output_length_chars=output_length_chars,
        validation_flags=validation_flags or {},
        error=error,
    )


class TraceTimer:
    def __init__(self) -> None:
        self._start: float | None = None

    def start(self) -> None:
        self._start = time.monotonic()

    def elapsed(self) -> float:
        if self._start is None:
            return 0.0
        return time.monotonic() - self._start


def extract_usage_tokens(provider_metadata: dict[str, Any] | None) -> dict[str, int | None]:
    if not isinstance(provider_metadata, dict):
        return {"prompt": None, "completion": None, "total": None}
    usage = provider_metadata.get("usage") or {}
    if not isinstance(usage, dict):
        return {"prompt": None, "completion": None, "total": None}
    return {
        "prompt": usage.get("prompt_tokens"),
        "completion": usage.get("completion_tokens"),
        "total": usage.get("total_tokens"),
    }


__all__ = [
    "TraceTimer",
    "build_stage_trace",
    "extract_usage_tokens",
]
