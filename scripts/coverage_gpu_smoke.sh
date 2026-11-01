#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python scripts/run_shared_gpu_batch.py --tag grouped_smoke -- \
  python -m pytest tests/test_grouped_attention.py -q -x \
  --junitxml=results/coverage_20261101/grouped_smoke.xml
