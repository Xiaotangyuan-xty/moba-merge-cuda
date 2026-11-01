#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python scripts/run_shared_gpu_batch.py --tag decode_smoke -- \
  python -m pytest tests/test_paged_decode.py -q -x \
  --junitxml=results/decode_paged_20261101/smoke.xml
