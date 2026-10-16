"""Suboperator tests for the Triton merge prototype (plan 7.2), route A.

Covers: no/one/multiple remote contributions, multiple remote rows per query,
LSE magnitudes close or far apart, mapping order permutations, odd P tails,
invalid slots, capacity overflow, non-finite checks, repeat-call consistency,
bf16 integration, and the real dumped fixtures vs official outputs.

GPU-gated: these tests allocate CUDA tensors.
"""

from pathlib import Path

import pytest
import torch

from moba_opt.ops.merge_triton import build_slots_torch, merge_forward
from reference.merge_reference import merge_grouped_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

FP32_ATOL = 1e-5
FP32_RTOL = 1e-4
FIX_DIR = Path(__file__).resolve().parents[1] / "results" / "raw" / "merge_fixtures"


def synth(counts, D=64, dtype=torch.float32, seed=0, lse_lo=-5.0, lse_hi=5.0):
    """Synthetic merge inputs: row p has 1 self branch + counts[p] remote branches."""
    P = len(counts)
    M = int(sum(counts))
    gen = torch.Generator().manual_seed(seed)
    self_out = torch.randn(P, D, generator=gen).cuda().to(dtype)
    self_lse = (torch.rand(P, generator=gen) * (lse_hi - lse_lo) + lse_lo).cuda()
    if M:
        remote_out = torch.randn(M, D, generator=gen).cuda().to(dtype)
        remote_lse = (torch.rand(M, generator=gen) * (lse_hi - lse_lo) + lse_lo).cuda()
        mapping = torch.cat([torch.full((int(c),), p, dtype=torch.long) for p, c in enumerate(counts)]).cuda()
    else:
        remote_out = torch.empty(0, D).cuda().to(dtype)
        remote_lse = torch.empty(0).cuda()
        mapping = torch.empty(0, dtype=torch.long).cuda()
    return self_out, self_lse, remote_out, remote_lse, mapping


def run(self_out, self_lse, remote_out, remote_lse, mapping, **kw):
    P, D = self_out.shape
    return merge_forward(
        self_out.view(P, 1, D), self_lse.view(P, 1),
        remote_out.view(-1, 1, D), remote_lse, mapping, **kw,
    )


def check_vs_fp64(out, lse, ref_in, atol=FP32_ATOL, rtol=FP32_RTOL):
    O, L = merge_grouped_reference(*ref_in)
    o64 = out.detach().reshape(O.shape[0], -1).to(torch.float64).cpu()
    l64 = lse.detach().reshape(-1).to(torch.float64).cpu()
    assert torch.isfinite(o64).all() and torch.isfinite(l64).all(), "NaN/Inf in outputs"
    assert torch.allclose(o64, O, atol=atol, rtol=rtol), f"O max err {(o64 - O).abs().max()}"
    assert torch.allclose(l64, L, atol=atol, rtol=rtol), f"L max err {(l64 - L).abs().max()}"


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


def test_mapping_order_permutation_bitwise():
    counts = [2, 3, 1, 4, 0, 2]
    so, sl, ro, rl, mp = synth(counts, seed=5)
    out0, lse0 = run(so, sl, ro, rl, mp, kmax=8)
    perm = torch.randperm(ro.shape[0]).cuda()
    out1, lse1 = run(so, sl, ro[perm], rl[perm], mp[perm], kmax=8)
    # physical row order changes lane-level accumulation order (the official
    # merge's atomic index_add_ is itself order-nondeterministic): fp32-tight,
    # not bitwise.
    assert torch.allclose(out0, out1, atol=1e-6, rtol=1e-5)
    assert torch.allclose(lse0, lse1, atol=1e-6, rtol=1e-5)


def test_kmax_not_power_of_two():
    # regression: slots row stride (kpad) must differ from kmax here; the
    # kernel must not read across rows.
    counts = [3, 1, 2, 0, 3, 1]
    so, sl, ro, rl, mp = synth(counts, seed=12)
    out, lse = run(so, sl, ro, rl, mp)  # kmax inferred = 3, kpad = 4
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


def test_capacity_overflow_raises():
    so, sl, ro, rl, mp = synth([3, 0], seed=6)
    with pytest.raises(ValueError, match="capacity exceeded"):
        run(so, sl, ro, rl, mp, kmax=2)


def test_atomic_capacity_overflow_raises():
    # same contract on the atomic builder: counts exceed kpad -> explicit error
    # instead of reading uninitialized slots.
    so, sl, ro, rl, mp = synth([3, 0], seed=6)
    with pytest.raises(ValueError, match="capacity exceeded"):
        run(so, sl, ro, rl, mp, kmax=2, builder="atomic")


@pytest.mark.parametrize("P", [1, 7, 33])
def test_odd_p_tail(P):
    counts = [i % 3 for i in range(P)]
    so, sl, ro, rl, mp = synth(counts, seed=7)
    out, lse = run(so, sl, ro, rl, mp, kmax=4)
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


def test_repeat_calls_identical():
    counts = [1, 2, 0, 3]
    so, sl, ro, rl, mp = synth(counts, seed=8)
    out0, lse0 = run(so, sl, ro, rl, mp, kmax=4)
    out1, lse1 = run(so, sl, ro, rl, mp, kmax=4)
    assert torch.equal(out0, out1) and torch.equal(lse0, lse1)


def test_bf16_integration():
    counts = [0, 1, 2, 3, 1, 0, 2, 4]
    so, sl, ro, rl, mp = synth(counts, D=128, dtype=torch.bfloat16, seed=9)
    out, lse = run(so, sl, ro, rl, mp, kmax=8)
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp), atol=2e-2, rtol=2e-2)


def test_head_dim_128_fp32():
    counts = [2, 3, 1, 0]
    so, sl, ro, rl, mp = synth(counts, D=128, seed=10)
    out, lse = run(so, sl, ro, rl, mp, kmax=4)
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


@pytest.mark.parametrize("fixture", sorted(FIX_DIR.glob("*.pt")), ids=lambda p: p.name)
def test_real_fixture_vs_official(fixture):
    fx = torch.load(fixture, map_location="cpu", weights_only=False)
    inp = fx["inputs"]
    dev = "cuda"
    self_out = inp["self_out"].to(dev)
    self_lse = inp["self_lse"].to(dev)
    remote_out = inp["remote_out"].to(dev)
    remote_lse = inp["remote_lse"].to(dev)
    mapping = inp["mapping"].to(dev)
    out, lse = merge_forward(self_out, self_lse, remote_out, remote_lse, mapping)

    out_exp = fx["expected"]["output"].to(dev)
    lse_exp = fx["expected"]["mixed_lse"].to(dev)
    d_out = (out.float() - out_exp.float()).abs()
    d_lse = (lse - lse_exp).abs()
    # official merge is fp32 with atomic index_add order + bf16 output cast:
    # differences should be at the 1-ulp-of-bf16 level at most.
    assert torch.isfinite(out.float()).all() and torch.isfinite(lse).all()
    assert d_out.max() < 2e-2, f"out max_abs {d_out.max()}"
    assert d_out.mean() < 1e-3, f"out mean_abs {d_out.mean()}"
    assert d_lse.max() < 1e-3, f"lse max_abs {d_lse.max()}"
    print(f"{fixture.name}: out max_abs={d_out.max():.2e} mean={d_out.mean():.2e}; lse max_abs={d_lse.max():.2e}")


def test_slots_deterministic_and_counts():
    counts = [2, 0, 3, 1]
    _, _, _, _, mp = synth(counts, seed=11)
    P = len(counts)
    slots, cnt = build_slots_torch(mp, P, kmax=4)
    assert cnt.tolist() == counts
    for p, c in enumerate(counts):
        row = slots[p].tolist()
        assert row[:c] == sorted(row[:c]), "slots must be ascending remote row indices"
        assert row[c:] == [-1] * (4 - c), "invalid slots must be -1"
    # global row indices covered exactly once
    valid = slots[slots >= 0].long()
    assert sorted(valid.tolist()) == list(range(mp.numel()))


def test_atomic_builder_slot_sets_and_counts():
    import triton

    from moba_opt.ops.merge_triton import _build_slots_atomic_kernel

    counts = [2, 0, 3, 1, 2]
    _, _, _, _, mp = synth(counts, seed=13)
    P, kmax, kpad = len(counts), 4, 4
    slots_t, _ = build_slots_torch(mp, P, kmax, kpad=kpad)
    slots_a = torch.empty((P, kpad), dtype=torch.int32, device="cuda")
    cnt_a = torch.zeros(P, dtype=torch.int32, device="cuda")
    _build_slots_atomic_kernel[(triton.cdiv(mp.numel(), 256),)](
        mp, slots_a, cnt_a, mp.numel(), KPAD=kpad, BLOCK=256
    )
    assert cnt_a.tolist() == counts
    for p, c in enumerate(counts):
        assert sorted(slots_a[p, :c].tolist()) == sorted(x for x in slots_t[p].tolist() if x >= 0)


def test_atomic_merge_matches_torch_builder():
    counts = [0, 1, 2, 3, 1, 0, 2, 5, 3, 1]
    so, sl, ro, rl, mp = synth(counts, seed=14)
    out_t, lse_t = run(so, sl, ro, rl, mp, kmax=8, builder="torch")
    out_a, lse_a = run(so, sl, ro, rl, mp, kmax=8, builder="atomic")
    assert torch.allclose(out_t, out_a, atol=1e-6, rtol=1e-5)
    assert torch.allclose(lse_t, lse_a, atol=1e-6, rtol=1e-5)


def test_atomic_vs_fp64():
    counts = [3, 1, 2, 0, 3, 1]
    so, sl, ro, rl, mp = synth(counts, seed=15)
    out, lse = run(so, sl, ro, rl, mp, builder="atomic")
    check_vs_fp64(out, lse, (so, sl, ro, rl, mp))


def test_atomic_repeat_fluctuation_bounded():
    counts = [1, 2, 0, 3, 2, 1]
    so, sl, ro, rl, mp = synth(counts, seed=16)
    out0, lse0 = run(so, sl, ro, rl, mp, kmax=4, builder="atomic")
    out1, lse1 = run(so, sl, ro, rl, mp, kmax=4, builder="atomic")
    # atomic ticket order is arbitrary, so lane-level accumulation order may
    # change between runs. Semantics are preserved and repeated runs must stay
    # within this explicit fp32-order tolerance -- no bitwise/per-ulp guarantee
    # is claimed (plan 7.4 last row; the official index_add_ is also atomic).
    assert torch.allclose(out0, out1, atol=1e-6, rtol=1e-5)
    assert torch.allclose(lse0, lse1, atol=1e-6, rtol=1e-5)
