# LLM Eval Harness

A generic, standalone eval-as-CI harness for LLM enrichment pipelines.

Extracted from a production transcription + enrichment pipeline that processes
audio through Whisper transcription, diarization, LLM-powered show summaries,
and speaker identity resolution. The eval framework was built to catch
regressions in enrichment quality before they reach production.

## Features

- **Deterministic scorers** — structural checks (required fields, hallucinated names,
  quote attribution, content coverage, speaker contract validation)
- **Pluggable LLM-as-judge** — optional LLM-based coherence scoring via a simple
  `JudgeModel` protocol (any callable `(rubric: str) -> float`)
- **Golden set** — versioned JSON fixtures with expected ground truth
- **CI-ready** — GitHub Actions workflow runs lint + test + eval on push/PR
- **Zero external dependencies** — pure Python standard library
- **Stage traces** — per-stage latency/token telemetry helpers

## Quick Start

```bash
# Install
pip install -e .

# Run tests
make test

# Run deterministic evals (creates sample metrics if none exist)
make eval

# Lint
make lint
```

## Usage

### Basic

```python
from llm_eval_harness import run_evals

report = run_evals("data/golden_set_v1.json", episode_metrics)
print(f"Pass rate: {report.aggregate_pass_rate:.1%}")
```

### With LLM Judge

```python
from llm_eval_harness import run_evals

def my_judge_model(rubric: str) -> float:
    # Call your LLM here
    return 4.5

report = run_evals("data/golden_set_v1.json", episodes, use_judge=True, judge_model=my_judge_model)
```

## Architecture

```
┌──────────────────┐     ┌──────────────────────┐
│ Episode Metrics   │────▶│   Eval Harness       │
│ (JSON with        │     │                      │
│  summary_text,    │     │  ┌───────────────┐   │
│  speaker_identity │     │  │ Golden Set    │   │
│  metadata)        │     │  │ Loader        │   │
└──────────────────┘     │  └───────┬───────┘   │
                         │          │           │
                         │  ┌───────▼───────┐   │
                         │  │ Deterministic │   │
                         │  │ Scorers       │   │
                         │  │ (7 built-in)  │   │
                         │  └───────┬───────┘   │
                         │          │           │
                         │  ┌───────▼───────┐   │
                         │  │ LLM-as-Judge │   │
                         │  │ (optional)   │   │
                         │  └───────┬───────┘   │
                         │          │           │
                         │  ┌───────▼───────┐   │
                         │  │ Eval Report   │   │
                         │  │ (JSON)        │   │
                         │  └───────────────┘   │
                         └──────────────────────┘
```

## Adding a New Scorer

1. Create a function matching the `Scorer` protocol:
   ```python
   from llm_eval_harness.contracts import ScorerResult

   def my_custom_scorer(output: dict, golden) -> ScorerResult:
       if "target_phrase" in output.get("summary_text", ""):
           return ScorerResult(name="my_scorer", passed=True)
       return ScorerResult(
           name="my_scorer", passed=False,
           failures=["missing target_phrase"],
           failure_category="custom_failure"
       )
   ```
2. Add it to the scorer list in `llm_eval_harness/scorers/__init__.py`
3. Update your golden set fixture with the expected data the new scorer needs

## Golden Set Format

The golden set is a versioned JSON file with two episode types:

- `summary_episodes` — episodes with expected summary fields, spellings, coverage terms
- `speaker_identity_episodes` — episodes with expected host labels, speaker mappings

See `data/golden_set_v1.json` for the full schema.

## Why this exists

This harness was extracted from a private production pipeline that transcribes
and enriches talk-radio episodes. It was designed to catch enrichment regressions
(eg, hallucinated names, missing fields, poor quote attribution) before they
reach production, using deterministic checks that are fast, cheap, and CI-friendly.

The LLM-as-judge scorer is reserved for manual or nightly runs to avoid CI
flakiness, while the deterministic scorers form the per-commit quality gate.
