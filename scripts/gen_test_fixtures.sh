#!/usr/bin/env bash
# Regenerate the test fixtures that are intentionally not committed to git
# (results/ is .gitignored). GPU-gated: requires approval + idle check.
#
#   routes/         official routing decisions for smoke configs (S01-S05)
#   merge_fixtures/ real merge-stage inputs/outputs (S02, S03, C01)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh

DEVICE="${1:-cuda:0}"

python reference/export_route.py --config configs/smoke.yaml --ids S01,S02,S03,S04,S05 --device "$DEVICE"
python scripts/dump_merge_fixtures.py --config configs/smoke.yaml --ids S02,S03 --device "$DEVICE"
python scripts/dump_merge_fixtures.py --config configs/core12.yaml --ids C01 --device "$DEVICE"
echo "fixtures regenerated under results/raw/{routes,merge_fixtures}/"
