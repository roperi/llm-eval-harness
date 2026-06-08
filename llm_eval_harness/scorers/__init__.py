from __future__ import annotations

from llm_eval_harness.contracts import Scorer
from llm_eval_harness.scorers.speaker_id import (
    build_speaker_id_scorers,
)
from llm_eval_harness.scorers.summary import build_summary_scorers


def get_deterministic_scorers() -> list[Scorer]:
    return build_summary_scorers() + build_speaker_id_scorers()


__all__ = ["get_deterministic_scorers"]
