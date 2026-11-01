#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export CUDA_VISIBLE_DEVICES=""
trap 'status=$?; python -c "import json; from pathlib import Path; Path(\"logs/cpu_build_status.json\").write_text(json.dumps({\"exit_code\":$status}))"' EXIT
python -m pytest tests/test_coverage_contract.py tests/test_reference.py tests/test_merge_fixtures.py -q -p no:cacheprovider
python -c 'from moba_opt.ops.merge_cuda import ext; print("CPU source build:", ext().__file__)'
