#!/usr/bin/env python
"""Private/public flash-attn interface probes + MoBA small-input probe (plan 6.2).

GPU-gated: run only with explicit approval, after an nvidia-smi idle check.
Writes results/environment/probe_<ts>.json; exit code 1 if any probe fails.
"""

import inspect
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "MoBA"))

RESULTS = []


def probe(name):
    def deco(fn):
        try:
            detail = fn() or {}
            RESULTS.append({"name": name, "status": "PASS", **detail})
            print(f"PASS {name} {detail.get('note', '')}")
        except Exception:  # noqa: BLE001
            RESULTS.append({"name": name, "status": "FAIL", "traceback": traceback.format_exc()})
            print(f"FAIL {name}")
            traceback.print_exc()

    return deco


def sig(fn):
    try:
        return str(inspect.signature(fn))
    except Exception as exc:  # noqa: BLE001
        return f"<no signature: {exc}>"


@probe("versions")
def p_versions():
    import flash_attn

    assert flash_attn.__version__ == "2.6.3", f"flash_attn {flash_attn.__version__} != 2.6.3"
    return {"note": f"flash_attn={flash_attn.__version__} torch={torch.__version__}"}


@probe("private_api_signatures")
def p_signatures():
    from flash_attn.flash_attn_interface import (
        _flash_attn_varlen_backward,
        _flash_attn_varlen_forward,
    )

    fwd = sig(_flash_attn_varlen_forward)
    bwd = sig(_flash_attn_varlen_backward)
    for required in ("cu_seqlens_q", "cu_seqlens_k", "max_seqlen_q", "max_seqlen_k", "softmax_scale", "causal", "dropout_p"):
        assert required in fwd, f"missing {required} in forward signature {fwd}"
        assert required in bwd, f"missing {required} in backward signature {bwd}"
    return {"fwd_signature": fwd, "bwd_signature": bwd}


def _tiny_qkv(total=256, heads=2, dim=64, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cuda")
    mk = lambda: torch.randn((total, heads, dim), generator=g, dtype=torch.float32).to(device=dev, dtype=dtype)  # noqa: E731
    cu = torch.tensor([0, total], device=dev, dtype=torch.int32)
    return mk(), mk(), mk(), cu, total


@probe("public_varlen_fwd_bwd")
def p_public():
    from flash_attn import flash_attn_varlen_func

    q, k, v, cu, max_len = _tiny_qkv()
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)
    out = flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len, causal=True)
    g = torch.randn_like(out)
    grads = torch.autograd.grad(out, (q, k, v), grad_outputs=g)
    assert out.shape == q.shape and torch.isfinite(out.float()).all()
    assert all(torch.isfinite(x.float()).all() for x in grads)
    return {"note": f"out {tuple(out.shape)} {out.dtype}"}


@probe("private_varlen_forward_layout")
def p_private_fwd():
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

    for dim in (64, 128):
        q, k, v, cu, max_len = _tiny_qkv(dim=dim)
        ret = _flash_attn_varlen_forward(
            q=q, k=k, v=v,
            cu_seqlens_q=cu, cu_seqlens_k=cu,
            max_seqlen_q=max_len, max_seqlen_k=max_len,
            softmax_scale=q.shape[-1] ** -0.5, causal=True, dropout_p=0.0,
        )
        assert isinstance(ret, tuple) and len(ret) == 8, f"expected 8-tuple, got {len(ret)}"
        out, lse = ret[4], ret[5]
        assert out.shape == q.shape, f"out {out.shape} != q {q.shape}"
        assert lse.shape == (q.shape[1], q.shape[0]), f"lse {lse.shape} != (H,T) {(q.shape[1], q.shape[0])}"
        assert lse.dtype == torch.float32, f"lse dtype {lse.dtype}"
        assert torch.isfinite(lse).all()
    return {"note": "8-tuple; out [T,H,D]; lse [H,T] fp32; D=64,128 ok"}


@probe("moba_small_fwd_vjp")
def p_moba():
    from moba.moba_efficient import moba_attn_varlen

    for dim in (64, 128):
        q, k, v, cu, max_len = _tiny_qkv(total=1024, heads=2, dim=dim, seed=dim)
        q.requires_grad_(True)
        k.requires_grad_(True)
        v.requires_grad_(True)
        out = moba_attn_varlen(q, k, v, cu, max_len, moba_chunk_size=256, moba_topk=3)
        g = torch.randn_like(out)
        grads = torch.autograd.grad(out, (q, k, v), grad_outputs=g)
        assert torch.isfinite(out.float()).all()
        assert all(torch.isfinite(x.float()).all() for x in grads)
    return {"note": "T=1024 H=2 C=256 K=3; D=64,128 fwd+random VJP finite"}


@probe("moba_k1_fallback_identity")
def p_k1():
    from flash_attn import flash_attn_varlen_func
    from moba.moba_efficient import moba_attn_varlen

    q, k, v, cu, max_len = _tiny_qkv(total=1024, heads=2, dim=64, seed=1)
    o_moba = moba_attn_varlen(q, k, v, cu, max_len, moba_chunk_size=256, moba_topk=1)
    o_fa = flash_attn_varlen_func(q, k, v, cu, cu, max_len, max_len, causal=True)
    identical = torch.equal(o_moba, o_fa)
    return {
        "note": f"k=1 == full causal FA: {identical}; max_abs_diff={(o_moba - o_fa).abs().max().item():.3e}",
        "identical": identical,
    }


def main():
    main.p = None
    print("probing on", torch.cuda.get_device_name(0))
    for fn in (p_versions, p_signatures, p_public, p_private_fwd, p_moba, p_k1):
        fn()
    out_dir = ROOT / "results" / "environment"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = out_dir / f"probe_{ts}.json"
    out.write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False))
    n_fail = sum(1 for r in RESULTS if r["status"] != "PASS")
    print(f"wrote {out}; {len(RESULTS) - n_fail}/{len(RESULTS)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
