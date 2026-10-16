"""Cross-check: official merge-stage outputs vs fp64 merge_reference on real
fixtures (dumped by scripts/dump_merge_fixtures.py from the diagnostic copy).

Formula-level check: the official fp32 merge (with bf16 output cast and atomic
index_add_) should agree with plan 4.3's math to within fp32/bf16 rounding.
A systematic mismatch here means the merge semantics were misunderstood, not a
numerical issue. CPU-only; fixtures live in results/raw/merge_fixtures/.
"""

from collections import defaultdict
from pathlib import Path

import pytest
import torch

from reference.merge_reference import merge_reference

FIX_DIR = Path(__file__).resolve().parents[1] / "results" / "raw" / "merge_fixtures"

# bf16 output cast dominates: unit roundoff ~2^-8 relative on O(1) values.
OUT_MAX_ABS = 2e-2
OUT_MEAN_ABS = 1e-3
LSE_MAX_ABS = 1e-3  # official mixed lse is fp32


def reorganize_and_merge(fx):
    self_out = fx["inputs"]["self_out"].to(torch.float64)  # [T,H,D]
    self_lse = fx["inputs"]["self_lse"].to(torch.float64)  # [T,H]
    remote_out = fx["inputs"]["remote_out"].to(torch.float64)  # [M,1,D]
    remote_lse = fx["inputs"]["remote_lse"].to(torch.float64)  # [M]
    mapping = fx["inputs"]["mapping"].long()  # [M], p = s*H + h
    T, H, D = self_out.shape

    groups = defaultdict(list)
    for m, p in enumerate(mapping.tolist()):
        groups[p].append(m)

    O = torch.empty(T, H, D, dtype=torch.float64)
    L = torch.empty(T, H, dtype=torch.float64)
    for p in range(T * H):
        s, h = divmod(p, H)
        branches = [(self_out[s, h].unsqueeze(0), self_lse[s, h].reshape(1))]
        for m in groups.get(p, []):
            branches.append((remote_out[m, 0].unsqueeze(0), remote_lse[m].reshape(1)))
        o_row, l_row = merge_reference(branches)
        O[s, h] = o_row[0]
        L[s, h] = l_row[0]
    return O, L


@pytest.mark.parametrize("fixture", sorted(FIX_DIR.glob("*.pt")), ids=lambda p: p.name)
def test_official_merge_matches_reference(fixture):
    fx = torch.load(fixture, map_location="cpu", weights_only=False)
    O, L = reorganize_and_merge(fx)
    out_exp = fx["expected"]["output"].to(torch.float64)
    lse_exp = fx["expected"]["mixed_lse"].to(torch.float64)

    d_out = (O - out_exp).abs()
    d_lse = (L - lse_exp).abs()
    assert torch.isfinite(O).all() and torch.isfinite(L).all()
    assert d_out.max() < OUT_MAX_ABS, f"output max_abs {d_out.max()}"
    assert d_out.mean() < OUT_MEAN_ABS, f"output mean_abs {d_out.mean()}"
    assert d_lse.max() < LSE_MAX_ABS, f"lse max_abs {d_lse.max()}"
    print(
        f"{fixture.name}: out max_abs={d_out.max():.2e} mean_abs={d_out.mean():.2e}; "
        f"lse max_abs={d_lse.max():.2e}"
    )
