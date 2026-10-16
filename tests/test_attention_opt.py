"""Full-path tests for the optimized merge path (V3): plan 7.3 (gradient
comparison vs frozen official), 7.5 (edge cases and invariants), 3.4 (k=1
fallback untouched).

GPU-gated: these tests allocate CUDA tensors.
"""

import itertools
from pathlib import Path

import pytest
import torch

import moba_opt  # noqa: F401  (registers impls)
import moba_opt.diagnostic as diag
import moba_opt.mixed_attention as opt_mod
from moba_opt.api import run_impl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

ROUTES_DIR = Path(__file__).resolve().parents[1] / "results" / "raw" / "routes"

SMOKE = [
    dict(total_tokens=512, heads=1, head_dim=64, chunk_size=128, topk=2),
    dict(total_tokens=1024, heads=2, head_dim=64, chunk_size=256, topk=3),
    dict(total_tokens=1024, heads=2, head_dim=128, chunk_size=256, topk=3),
    dict(total_tokens=2048, heads=4, head_dim=128, chunk_size=512, topk=4),
    dict(total_tokens=2049, heads=2, head_dim=64, chunk_size=512, topk=2),
    dict(total_tokens=300, heads=1, head_dim=64, chunk_size=512, topk=4),  # single block -> fallback
]


def gen(total, heads, dim, seed=0, lens=None, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)

    def mk():
        return torch.randn(total, heads, dim, generator=g).cuda().to(dtype)

    q, k, v = mk(), mk(), mk()
    lens = lens or [total]
    assert sum(lens) == total and all(x > 0 for x in lens)
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device="cuda", dtype=torch.int32)
    return q, k, v, cu, max(lens)


def cfg_of(d):
    return {"chunk_size": d["chunk_size"], "topk": d["topk"]}


def err(a, b):
    d = (a.float() - b.float()).abs()
    return d.max().item(), d.mean().item()


@pytest.mark.parametrize("d", SMOKE, ids=lambda d: f"T{d['total_tokens']}H{d['heads']}D{d['head_dim']}C{d['chunk_size']}K{d['topk']}")
def test_route_identity_with_official(d):
    q, k, v, cu, ml = gen(d["total_tokens"], d["heads"], d["head_dim"], seed=11)
    captured = {}

    def hook(tag):
        def h(**kw):
            captured[tag] = kw["selected"].cpu()

        return h

    # the frozen upstream module has no hook; the diagnostic copy's routing is
    # line-identical to it (verified bitwise on C01), so it stands in here.
    diag.ROUTE_HOOK = hook("official")
    opt_mod.ROUTE_HOOK = hook("opt")
    try:
        diag.moba_attn_varlen(q, k, v, cu, ml, d["chunk_size"], d["topk"])
        run_impl("opt_merge_atomic", q, k, v, cu, ml, cfg_of(d))
    finally:
        diag.ROUTE_HOOK = None
        opt_mod.ROUTE_HOOK = None
    assert ("official" in captured) == ("opt" in captured), "fallback branch diverged"
    if "official" in captured:
        assert torch.equal(captured["official"], captured["opt"]), "selected block sets differ"


@pytest.mark.parametrize("d", SMOKE, ids=lambda d: f"T{d['total_tokens']}H{d['heads']}D{d['head_dim']}C{d['chunk_size']}K{d['topk']}")
def test_forward_matches_official(d):
    q, k, v, cu, ml = gen(d["total_tokens"], d["heads"], d["head_dim"], seed=12)
    o_off = run_impl("official", q, k, v, cu, ml, cfg_of(d))
    o_opt = run_impl("opt_merge_atomic", q, k, v, cu, ml, cfg_of(d))
    assert torch.isfinite(o_opt.float()).all()
    mx, mn = err(o_off, o_opt)
    if d["total_tokens"] <= d["chunk_size"]:
        assert torch.equal(o_off, o_opt), "fallback branch must be identical"
    else:
        assert mx < 2e-2 and mn < 1e-3, f"forward diff max={mx} mean={mn}"


@pytest.mark.parametrize("d", SMOKE[:4], ids=lambda d: f"T{d['total_tokens']}H{d['heads']}D{d['head_dim']}")
@pytest.mark.parametrize("seed", [21, 22])
def test_backward_matches_official(d, seed):
    q, k, v, cu, ml = gen(d["total_tokens"], d["heads"], d["head_dim"], seed=seed)
    g = torch.randn_like(q)

    grads = {}
    for impl in ("official", "opt_merge_atomic"):
        for t in (q, k, v):
            t.grad = None
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        o = run_impl(impl, q, k, v, cu, ml, cfg_of(d))
        grads[impl] = torch.autograd.grad(o, (q, k, v), grad_outputs=g)
    q.requires_grad_(False)
    k.requires_grad_(False)
    v.requires_grad_(False)

    for name, a, b in zip(("dq", "dk", "dv"), grads["official"], grads["opt_merge_atomic"]):
        assert torch.isfinite(b.float()).all(), f"{name} not finite"
        mx, mn = err(a, b)
        assert mx < 4e-2 and mn < 4e-4, f"{name} diff max={mx} mean={mn}"


@pytest.mark.parametrize("route_file", sorted(ROUTES_DIR.glob("S0*_seed0.pt")), ids=lambda p: p.name)
def test_fixed_route_fp64(route_file):
    from reference.fixed_route_attention import fixed_route_attention

    payload = torch.load(route_file, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    route = payload["route"]
    q = payload["q"].cuda()
    k = payload["k"].cuda()
    v = payload["v"].cuda()
    cu = route["cu_seqlens"].cuda().int()
    o_opt = run_impl("opt_merge_atomic", q, k, v, cu, route["max_seqlen"], cfg)
    o_ref, _ = fixed_route_attention(payload["q"], payload["k"], payload["v"], route["cu_seqlens"], route)
    mx, mn = err(o_opt.cpu(), o_ref)
    assert mx < 4e-2 and mn < 4e-4, f"opt vs fp64 ref: max={mx} mean={mn}"


@pytest.mark.parametrize("lens", [[1], [127], [128], [129], [511, 513], [128, 1, 129]])
@pytest.mark.parametrize("topk", [1, 2, 4])
def test_edge_lengths(lens, topk):
    total = sum(lens)
    q, k, v, cu, ml = gen(total, 2, 64, seed=13, lens=lens)
    cfg = {"chunk_size": 128, "topk": topk}
    o_off = run_impl("official", q, k, v, cu, ml, cfg)
    o_opt = run_impl("opt_merge_atomic", q, k, v, cu, ml, cfg)
    assert torch.isfinite(o_opt.float()).all()
    mx, mn = err(o_off, o_opt)
    assert mx < 2e-2 and mn < 1e-3, f"lens={lens} topk={topk}: max={mx} mean={mn}"


def test_packed_equals_per_sequence():
    lens = [300, 700, 500]
    total = sum(lens)
    q, k, v, cu, ml = gen(total, 2, 64, seed=14, lens=lens)
    cfg = {"chunk_size": 128, "topk": 3}
    o_packed = run_impl("opt_merge_atomic", q, k, v, cu, ml, cfg)
    for b, (s0, s1) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        cu_b = torch.tensor([0, s1 - s0], device="cuda", dtype=torch.int32)
        o_b = run_impl("opt_merge_atomic", q[s0:s1], k[s0:s1], v[s0:s1], cu_b, s1 - s0, cfg)
        mx, mn = err(o_packed[s0:s1], o_b)
        assert mx < 2e-2 and mn < 1e-3, f"seq {b}: packed vs single max={mx} mean={mn}"


def test_sequence_isolation():
    lens = [300, 700]
    total = sum(lens)
    q, k, v, cu, ml = gen(total, 2, 64, seed=15, lens=lens)
    cfg = {"chunk_size": 128, "topk": 3}
    o0 = run_impl("opt_merge_atomic", q, k, v, cu, ml, cfg)
    k2 = k.clone()
    v2 = v.clone()
    k2[lens[0]:] = torch.randn_like(k2[lens[0]:])
    v2[lens[0]:] = torch.randn_like(v2[lens[0]:])
    o1 = run_impl("opt_merge_atomic", q, k2, v2, cu, ml, cfg)
    assert torch.equal(o0[: lens[0]], o1[: lens[0]]), "changing another sequence affected this sequence"


def test_no_future_leakage():
    total = 1024
    q, k, v, cu, ml = gen(total, 2, 64, seed=16)
    cfg = {"chunk_size": 128, "topk": 3}
    o0 = run_impl("opt_merge_atomic", q, k, v, cu, ml, cfg)
    cut = 512
    k2 = k.clone()
    v2 = v.clone()
    k2[cut:] = torch.randn_like(k2[cut:])
    v2[cut:] = torch.randn_like(v2[cut:])
    o1 = run_impl("opt_merge_atomic", q, k2, v2, cu, ml, cfg)
    assert torch.equal(o0[:cut], o1[:cut]), "future K/V changed earlier outputs"


def test_repeat_calls_bounded_fluctuation():
    q, k, v, cu, ml = gen(1024, 2, 64, seed=17)
    cfg = {"chunk_size": 256, "topk": 3}
    o0 = run_impl("opt_merge_atomic", q, k, v, cu, ml, cfg)
    # interleave a different workload, then repeat: no leftover workspace state
    q2, k2, v2, cu2, ml2 = gen(512, 1, 64, seed=18)
    run_impl("opt_merge_atomic", q2, k2, v2, cu2, ml2, {"chunk_size": 128, "topk": 2})
    o1 = run_impl("opt_merge_atomic", q, k, v, cu, ml, cfg)
    mx, _ = err(o0, o1)
    assert mx < 4e-3, f"atomic-order fluctuation beyond 1 bf16 ulp: {mx}"


@pytest.mark.parametrize(
    "bad",
    [
        lambda q, k, v, cu: (q.double(), k, v, cu),
        lambda q, k, v, cu: (q.cpu(), k, v, cu),
        lambda q, k, v, cu: (q, k, v, cu.long()),
        lambda q, k, v, cu: (q, k[:, :1], v, cu),
        lambda q, k, v, cu: (q, k, v, torch.tensor([0, 0, cu[-1]], device="cuda", dtype=torch.int32)),
    ],
    ids=["dtype", "cpu", "cu_long", "head_mismatch", "empty_seq"],
)
def test_invalid_inputs_rejected(bad):
    q, k, v, cu, ml = gen(512, 2, 64, seed=19)
    bq, bk, bv, bcu = bad(q, k, v, cu)
    with pytest.raises((ValueError, AssertionError, RuntimeError)):
        run_impl("opt_merge_atomic", bq, bk, bv, bcu, ml, {"chunk_size": 128, "topk": 2})
