#!/usr/bin/env bash
# run_eval.sh — Run deterministic evals against the golden set.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
GOLDEN_SET="${GOLDEN_SET:-data/golden_set_v1.json}"
MIN_PASS_RATE="${EVAL_MIN_PASS_RATE:-0.9}"
LATEST_METRICS="${LATEST_METRICS:-data/latest_run_metrics.json}"
EVAL_OUTPUT="${EVAL_OUTPUT:-data/latest_eval_report.json}"

echo "==> Evals (golden set: ${GOLDEN_SET}, min pass rate: ${MIN_PASS_RATE})"

if [[ ! -f "$LATEST_METRICS" ]]; then
  cat > "$LATEST_METRICS" << 'JSONEOF'
{
  "episodes": [
    {
      "date": "2026-01-01",
      "summary_text": "John Smith and Jane Doe interviewed a guest about technology trends. John Smith asked about AI developments. Jane Doe discussed the future of computing.",
      "speaker_identity_metadata": {
        "mappings": [
          {"channel": "SPEAKER_00", "resolved_label": "John Smith", "status": "resolved"},
          {"channel": "SPEAKER_01", "resolved_label": "Jane Doe", "status": "resolved"}
        ]
      }
    },
    {
      "date": "2026-01-02",
      "summary_text": "Jane Doe and John Smith took calls from listeners. Smith discussed current events. Doe moderated the discussion.",
      "speaker_identity_metadata": {
        "mappings": [
          {"channel": "SPEAKER_00", "resolved_label": "Jane Doe", "status": "resolved"},
          {"channel": "SPEAKER_01", "resolved_label": "John Smith", "status": "resolved"},
          {"channel": "SPEAKER_02", "resolved_label": "Unknown Speaker 1", "status": "unresolved"}
        ]
      }
    }
  ]
}
JSONEOF
  echo "  Created sample metrics file at ${LATEST_METRICS}"
fi

"$PYTHON_BIN" -c "
import json, sys
from llm_eval_harness import run_evals

with open('${LATEST_METRICS}') as f:
    run_metrics = json.load(f)

episodes = run_metrics.get('episodes', [])
report = run_evals('${GOLDEN_SET}', episodes)

with open('${EVAL_OUTPUT}', 'w') as f:
    json.dump(report.to_dict(), f, indent=2)

rate = report.aggregate_pass_rate
print(f'Eval pass rate: {rate:.1%} ({report.passed_scorers}/{report.total_scorers})')
if report.failure_taxonomy:
    print(f'Failures: {report.failure_taxonomy}')
if rate < ${MIN_PASS_RATE}:
    print(f'ERROR: Pass rate {rate:.1%} below minimum ${MIN_PASS_RATE}')
    sys.exit(1)
" 2>&1 || exit $?
