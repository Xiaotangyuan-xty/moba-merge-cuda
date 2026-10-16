"""Unified call into the frozen official MoBA implementation (plan layer-1 reference).

Validates the mainline support envelope (plan section 2.4) before dispatching.
The upstream code under third_party/MoBA is never modified here.
"""

import torch

from .api import register

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
MAINLINE_HEAD_DIMS = (64, 128)


def _validate(q, k, v, cu_seqlens):
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dim() != 3:
            raise ValueError(f"{name} must be [T,H,D], got shape {tuple(t.shape)}")
        if t.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"{name} dtype {t.dtype} not in {SUPPORTED_DTYPES}")
        if not t.is_cuda:
            raise ValueError(f"{name} must be on CUDA")
    if not (q.shape == k.shape == v.shape):
        raise ValueError(f"mainline is MHA only: q/k/v shapes must match, got {q.shape} {k.shape} {v.shape}")
    if q.shape[-1] not in MAINLINE_HEAD_DIMS:
        raise ValueError(f"head_dim {q.shape[-1]} outside mainline {MAINLINE_HEAD_DIMS}")
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be CUDA int32")
    cu = cu_seqlens.tolist()
    if cu[0] != 0 or cu[-1] != q.shape[0] or any(b <= a for a, b in zip(cu, cu[1:])):
        raise ValueError(f"cu_seqlens must start at 0, end at T, strictly increase (non-empty seqs): {cu}")


def official_attn_varlen(q, k, v, cu_seqlens, max_seqlen, config):
    _validate(q, k, v, cu_seqlens)
    from moba.moba_efficient import moba_attn_varlen

    return moba_attn_varlen(
        q,
        k,
        v,
        cu_seqlens,
        int(max_seqlen),
        moba_chunk_size=int(config["chunk_size"]),
        moba_topk=int(config["topk"]),
    )


register("official", official_attn_varlen)


def official_nvtx(q, k, v, cu_seqlens, max_seqlen, config):
    """V1 ablation seat (plan 9.3): the record_function-marked diagnostic copy,
    numerically identical to the frozen baseline (verified bitwise on C01)."""
    _validate(q, k, v, cu_seqlens)
    from moba_opt.diagnostic import moba_attn_varlen as diag_impl

    return diag_impl(
        q,
        k,
        v,
        cu_seqlens,
        int(max_seqlen),
        moba_chunk_size=int(config["chunk_size"]),
        moba_topk=int(config["topk"]),
    )


register("official_nvtx", official_nvtx)
