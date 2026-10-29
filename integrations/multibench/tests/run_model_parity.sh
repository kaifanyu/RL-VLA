#!/usr/bin/env bash
set -euo pipefail
PARITY_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$PARITY_ROOT"
PARITY_OUTPUT="$PARITY_ROOT/integrations/multibench/tests/evidence/model_parity"
mkdir -p "$PARITY_OUTPUT"
bash integrations/multibench/run.sh native -m multibench.tests.check_model_parity native --output "$PARITY_OUTPUT/native"
bash integrations/multibench/run.sh torch -m multibench.tests.check_model_parity torch --output "$PARITY_OUTPUT/torch"
PYTHONPATH="$PARITY_ROOT/integrations" .venv/bin/python -m multibench.tests.check_model_parity compare \
  --first "$PARITY_OUTPUT/native" --second "$PARITY_OUTPUT/torch" --output "$PARITY_OUTPUT/comparison.json"
