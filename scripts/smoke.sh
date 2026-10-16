#!/usr/bin/env bash
# GPU-gated smoke: 接口探针 + 官方正确性测试（方案 6.2 / 6.3）。
# 运行前必须：1) 得到所有者许可；2) nvidia-smi 确认目标卡显存与进程空闲。
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

echo "== step 1/3: environment evidence (CPU-safe) =="
python scripts/check_environment.py

echo "== step 2/3: interface probes (GPU) =="
python scripts/probe_interfaces.py

echo "== step 3/3: official correctness tests (GPU) =="
cd third_party/MoBA
python -m pytest -q tests/test_moba_attn.py

echo "SMOKE_DONE"
