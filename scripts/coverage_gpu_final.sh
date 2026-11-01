#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python scripts/run_shared_gpu_batch.py --tag coverage_final_suite -- \
  python -m pytest tests/ -q -x --durations=8 \
  --junitxml=results/coverage_20261101/final_suite.xml
# CLI smoke only: too few samples for a performance conclusion.
python scripts/run_shared_gpu_batch.py --tag grouped_cli_smoke -- \
  python benchmarks/benchmark_attention.py --config configs/grouped_coverage.yaml \
  --ids CLI_SMOKE --impl official_gqa_repeat,opt_gqa_cuda --rounds 1 --samples 2 --warmup 1 \
  --out results/coverage_20261101/cli_smoke_only.jsonl
