#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python scripts/run_shared_gpu_batch.py --tag coverage_full_suite -- \
  python -m pytest tests/ -q -x --durations=8 \
  --junitxml=results/coverage_20261101/full_suite.xml
