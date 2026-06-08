from __future__ import annotations

from llm_eval_harness.contracts import Scorer
from llm_eval_harness.scorers.speaker_id import (
    build_speaker_id_scorers,
)
from llm_eval_harness.scorers.summary import build_summary_scorers

_SUMMARY_SCORER_NAMES: set[str] = set()
_SPEAKER_ID_SCORER_NAMES: set[str] = set()


def _register_scorers() -> None:
    global _SUMMARY_SCORER_NAMES, _SPEAKER_ID_SCORER_NAMES
    if not _SUMMARY_SCORER_NAMES:
        for s in build_summary_scorers():
            _SUMMARY_SCORER_NAMES.add(s.__name__)
    if not _SPEAKER_ID_SCORER_NAMES:
        for s in build_speaker_id_scorers():
            _SPEAKER_ID_SCORER_NAMES.add(s.__name__)


def get_deterministic_scorers() -> list[Scorer]:
    _register_scorers()
    return build_summary_scorers() + build_speaker_id_scorers()


def is_summary_scorer(scorer: Scorer) -> bool:
    _register_scorers()
    return scorer.__name__ in _SUMMARY_SCORER_NAMES


def is_speaker_id_scorer(scorer: Scorer) -> bool:
    _register_scorers()
    return scorer.__name__ in _SPEAKER_ID_SCORER_NAMES


__all__ = [
    "get_deterministic_scorers",
    "is_summary_scorer",
    "is_speaker_id_scorer",
]
