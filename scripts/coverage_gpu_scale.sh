#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python scripts/run_shared_gpu_batch.py --tag grouped_scale -- \
  python -m pytest tests/test_grouped_scale.py -q -x --durations=4 \
  --junitxml=results/coverage_20261101/grouped_scale.xml
