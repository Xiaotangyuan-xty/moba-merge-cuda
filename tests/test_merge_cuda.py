"""Suboperator tests for the CUDA merge kernels (plan 7.2 applied to
ops/merge.cu). Mirrors the Triton matrix; semantics must match the Triton
reference path and the fp64 reference.

GPU-gated: these tests allocate CUDA tensors and JIT-compile the extension.
"""

from pathlib import Path

import pytest
import torch
from test_merge_triton import FIX_DIR, check_vs_fp64, synth

from moba_opt.ops.merge_cuda import merge_forward_cuda
from moba_opt.ops.merge_triton import merge_forward as merge_forward_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def run(so, sl, ro, rl, mp, **kw):
    P, D = so.shape
    return merge_forward_cuda(
        so.view(P, 1, D), sl.view(P, 1), ro.view(-1, 1, D), rl, mp, **kw
    )


def test_no_remote_contribution():
    so, sl, ro, rl, mp = synth([0, 0, 0, 0])
    out, lse = run(so, sl, ro, rl, mp)
    assert torch.equal(out.view(4, -1), so)
    assert torch.equal(lse.view(4), sl)


def test_single_remote_contribution():
    so, sl, ro, rl, mp = synth([1, 1, 1, 1], seed=1)
    out, lse = run(so, sl, ro, rl, mp)
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


def test_multiple_remote_mixed():
    counts = [0, 1, 2, 3, 1, 0, 2, 5, 3, 1]
    so, sl, ro, rl, mp = synth(counts, seed=2)
    out, lse = run(so, sl, ro, rl, mp, kmax=8)
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


def test_lse_extreme_gap_no_nan():
    so, sl, ro, rl, mp = synth([2, 2, 2, 2], seed=3, lse_lo=-1e30, lse_hi=1e30)
    out, lse = run(so, sl, ro, rl, mp)
    assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all()
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


def test_lse_close_values():
    so, sl, ro, rl, mp = synth([3, 3, 3, 3], seed=4, lse_lo=0.0, lse_hi=1e-6)
    out, lse = run(so, sl, ro, rl, mp, kmax=4)
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


@pytest.mark.parametrize("P", [1, 7, 33])
def test_odd_p_tail(P):
    counts = [i % 3 for i in range(P)]
    so, sl, ro, rl, mp = synth(counts, seed=7)
    out, lse = run(so, sl, ro, rl, mp, kmax=4)
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


def test_kmax_not_power_of_two():
    counts = [3, 1, 2, 0, 3, 1]
    so, sl, ro, rl, mp = synth(counts, seed=12)
    out, lse = run(so, sl, ro, rl, mp)  # kmax inferred = 3, kpad = 4
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


def test_bf16_integration():
    counts = [0, 1, 2, 3, 1, 0, 2, 4]
    so, sl, ro, rl, mp = synth(counts, D=128, dtype=torch.bfloat16, seed=9)
    out, lse = run(so, sl, ro, rl, mp, kmax=8)
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp), atol=2e-2, rtol=2e-2)


def test_fp16_integration():
    counts = [2, 3, 1, 0]
    so, sl, ro, rl, mp = synth(counts, D=64, dtype=torch.float16, seed=10)
    out, lse = run(so, sl, ro, rl, mp, kmax=4)
    assert out.dtype == torch.float16
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp), atol=1e-3, rtol=1e-3)


def test_head_dim_128_fp32():
    counts = [2, 3, 1, 0]
    so, sl, ro, rl, mp = synth(counts, D=128, seed=10)
    out, lse = run(so, sl, ro, rl, mp, kmax=4)
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


def test_repeat_calls_bounded_fluctuation():
    counts = [1, 2, 0, 3, 2, 1]
    so, sl, ro, rl, mp = synth(counts, seed=16)
    out0, lse0 = run(so, sl, ro, rl, mp, kmax=4)
    out1, lse1 = run(so, sl, ro, rl, mp, kmax=4)
    # atomic ticket order is arbitrary: lane-level accumulation order may vary
    # between runs; repeated runs must stay within this explicit tolerance
    # (semantics preserved, no bitwise guarantee -- plan 7.4 last row).
    assert torch.allclose(out0, out1, atol=1e-6, rtol=1e-5)
    assert torch.allclose(lse0, lse1, atol=1e-6, rtol=1e-5)


def test_cuda_capacity_overflow_raises():
    # counts exceed kpad -> the launcher must raise (TORCH_CHECK), never read
    # uninitialized slots.
    so, sl, ro, rl, mp = synth([3, 0], seed=6)
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        run(so, sl, ro, rl, mp, kmax=2)


def test_cuda_matches_triton():
    counts = [0, 1, 2, 3, 1, 0, 2, 5, 3, 1]
    so, sl, ro, rl, mp = synth(counts, seed=17)
    P, D = so.shape
    out_t, lse_t = merge_forward_triton(
        so.view(P, 1, D), sl.view(P, 1), ro.view(-1, 1, D), rl, mp, kmax=8, builder="atomic"
    )
    out_c, lse_c = run(so, sl, ro, rl, mp, kmax=8)
    assert torch.allclose(out_t, out_c, atol=1e-6, rtol=1e-5)
    assert torch.allclose(lse_t, lse_c, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("fixture", sorted(FIX_DIR.glob("*.pt")), ids=lambda p: p.name)
def test_real_fixture_vs_official(fixture):
    fx = torch.load(fixture, map_location="cpu", weights_only=False)
    inp = fx["inputs"]
    dev = "cuda"
    out, lse = merge_forward_cuda(
        inp["self_out"].to(dev),
        inp["self_lse"].to(dev),
        inp["remote_out"].to(dev),
        inp["remote_lse"].to(dev),
        inp["mapping"].to(dev),
    )
    out_exp = fx["expected"]["output"].to(dev)
    lse_exp = fx["expected"]["mixed_lse"].to(dev)
    d_out = (out.float() - out_exp.float()).abs()
    d_lse = (lse - lse_exp).abs()
    assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all()
    assert d_out.max() < 2e-2, f"out max_abs {d_out.max()}"
    assert d_out.mean() < 1e-3, f"out mean_abs {d_out.mean()}"
    assert d_lse.max() < 1e-3, f"lse max_abs {d_lse.max()}"
    print(f"{fixture.name}: out max_abs={d_out.max():.2e} mean={d_out.mean():.2e}; lse max_abs={d_lse.max():.2e}")
