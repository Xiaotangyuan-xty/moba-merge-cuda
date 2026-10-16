"""Full-path checks for the CUDA-backend opt path (opt_merge_cuda):
plan 7.3 gradients vs frozen official, fixed-route fp64 cross-check,
k=1 fallback identity. GPU-gated.
"""

import pytest
import torch
from test_attention_opt import SMOKE, SUPP, cfg_of, err, gen

import moba_opt  # noqa: F401  (registers impls)
from moba_opt.api import run_impl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("d", SMOKE, ids=lambda d: f"T{d['total_tokens']}H{d['heads']}D{d['head_dim']}C{d['chunk_size']}K{d['topk']}")
def test_cuda_fwd_bwd_vs_official(d):
    q, k, v, cu, ml = gen(d["total_tokens"], d["heads"], d["head_dim"], seed=31)
    cfg = cfg_of(d)
    fallback = d["total_tokens"] <= d["chunk_size"]
    outs = {}
    grads = {}
    g = torch.randn_like(q)
    for impl in ("official", "opt_merge_cuda"):
        for t in (q, k, v):
            t.grad = None
            t.requires_grad_(True)
        o = run_impl(impl, q, k, v, cu, ml, cfg)
        outs[impl] = o
        grads[impl] = torch.autograd.grad(o, (q, k, v), grad_outputs=g)
    for t in (q, k, v):
        t.requires_grad_(False)
    if fallback:
        assert torch.equal(outs["official"], outs["opt_merge_cuda"])
        for a, b in zip(grads["official"], grads["opt_merge_cuda"]):
            assert torch.equal(a, b)
        return
    mx, mn = err(outs["official"], outs["opt_merge_cuda"])
    assert mx < 2e-2 and mn < 1e-3, f"fwd max={mx} mean={mn}"
    for name, a, b in zip(("dq", "dk", "dv"), grads["official"], grads["opt_merge_cuda"]):
        mx, mn = err(a, b)
        assert mx < 4e-2 and mn < 4e-4, f"{name} max={mx} mean={mn}"


@pytest.mark.parametrize(
    "d", SUPP, ids=lambda d: f"C{d['chunk_size']}K{d['topk']}T{d['total_tokens']}{'fp16' if d.get('dtype') == torch.float16 else ''}{'ragged' if d.get('lens') else ''}"
)
def test_cuda_supplementary_vs_official(d):
    dtype = d.get("dtype", torch.bfloat16)
    q, k, v, cu, ml = gen(d["total_tokens"], d["heads"], d["head_dim"], seed=32, lens=d.get("lens"), dtype=dtype)
    cfg = cfg_of(d)
    g = torch.randn_like(q)
    outs = {}
    grads = {}
    for impl in ("official", "opt_merge_cuda"):
        for t in (q, k, v):
            t.grad = None
            t.requires_grad_(True)
        o = run_impl(impl, q, k, v, cu, ml, cfg)
        outs[impl] = o
        grads[impl] = torch.autograd.grad(o, (q, k, v), grad_outputs=g)
    for t in (q, k, v):
        t.requires_grad_(False)
    mx, mn = err(outs["official"], outs["opt_merge_cuda"])
    assert mx < 2e-2 and mn < 1e-3, f"fwd max={mx} mean={mn}"
    for name, a, b in zip(("dq", "dk", "dv"), grads["official"], grads["opt_merge_cuda"]):
        mx, mn = err(a, b)
        assert mx < 4e-2 and mn < 4e-4, f"{name} max={mx} mean={mn}"


def test_cuda_k1_fallback_identical():
    q, k, v, cu, ml = gen(1024, 2, 64, seed=33)
    cfg = {"chunk_size": 256, "topk": 1}
    o_off = run_impl("official", q, k, v, cu, ml, cfg)
    o_cuda = run_impl("opt_merge_cuda", q, k, v, cu, ml, cfg)
    assert torch.equal(o_off, o_cuda)


def test_cuda_fixed_route_fp64():
    from pathlib import Path

    from reference.fixed_route_attention import fixed_route_attention

    route_path = Path(__file__).resolve().parents[1] / "results" / "raw" / "routes" / "S02_seed0.pt"
    if not route_path.exists():
        pytest.skip("route fixtures not generated; run scripts/gen_test_fixtures.sh first (GPU-gated)")
    payload = torch.load(route_path, map_location="cpu", weights_only=False)
    cfg = payload["config"]
    route = payload["route"]
    q = payload["q"].cuda()
    k = payload["k"].cuda()
    v = payload["v"].cuda()
    cu = route["cu_seqlens"].cuda().int()
    o_opt = run_impl("opt_merge_cuda", q, k, v, cu, route["max_seqlen"], cfg)
    o_ref, _ = fixed_route_attention(payload["q"], payload["k"], payload["v"], route["cu_seqlens"], route)
    mx, mn = err(o_opt.cpu(), o_ref)
    assert mx < 4e-2 and mn < 4e-4, f"cuda vs fp64 ref: max={mx} mean={mn}"
