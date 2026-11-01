#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python scripts/run_shared_gpu_batch.py --tag decode_full_suite -- \
  python -m pytest tests/ -q -x --durations=10 \
  --junitxml=results/decode_paged_20261101/full_suite.xml
