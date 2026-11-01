"""Unified call into the frozen official MoBA implementation (plan layer-1 reference).

Validates the mainline support envelope (plan section 2.4) before dispatching.
The upstream code under third_party/MoBA is never modified here.
"""

import numbers

import torch

from .api import register

SUPPORTED_DTYPES = (torch.bfloat16, torch.float16)
MAINLINE_HEAD_DIMS = tuple(range(8, 257, 8))


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def validate_shapes(q_shape, k_shape, v_shape, allow_gqa=False):
    """Pure CPU contract; useful without touching a CUDA runtime."""
    if any(len(s) != 3 or any(d < 1 for d in s) for s in (q_shape, k_shape, v_shape)):
        raise ValueError("q/k/v must be nonempty [T,H,D]")
    if k_shape != v_shape or q_shape[0] != k_shape[0] or q_shape[2] != k_shape[2]:
        raise ValueError("q/k/v must share T,D and k/v must share shape")
    if q_shape[2] not in MAINLINE_HEAD_DIMS:
        raise ValueError("head_dim must be a multiple of 8 in [8,256]")
    if q_shape[1] % k_shape[1] or (not allow_gqa and q_shape[1] != k_shape[1]):
        raise ValueError("Hq must be divisible by Hkv; this entry is MHA only" if not allow_gqa
                         else "Hq must be divisible by Hkv")
    return q_shape[1] // k_shape[1]


def validate_config(config):
    _positive_int(config.get("chunk_size"), "chunk_size")
    topk = _positive_int(config.get("topk"), "topk")
    if topk > 9:
        raise ValueError("topk must be in [1,9] (at most 8 remote contributions)")
    if config.get("causal", True) is not True or config.get("dropout", 0.0) != 0:
        raise ValueError("only causal=True, dropout=0 are supported")


def _validate(q, k, v, cu_seqlens, *, allow_gqa=False, max_seqlen=None, config=None):
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not isinstance(t, torch.Tensor):
            raise ValueError(f"{name} must be a Tensor")
        if t.dim() != 3:
            raise ValueError(f"{name} must be [T,H,D], got shape {tuple(t.shape)}")
        if t.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"{name} dtype {t.dtype} not in {SUPPORTED_DTYPES}")
        if not t.is_cuda:
            raise ValueError(f"{name} must be on CUDA")
        if t.device != q.device or t.dtype != q.dtype:
            raise ValueError("q/k/v must share device and dtype")
        if t.stride(-1) != 1:
            raise ValueError("q/k/v last dimension must be contiguous")
    validate_shapes(tuple(q.shape), tuple(k.shape), tuple(v.shape), allow_gqa)
    if not isinstance(cu_seqlens, torch.Tensor):
        raise ValueError("cu_seqlens must be a Tensor")
    if cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_cuda:
        raise ValueError("cu_seqlens must be CUDA int32")
    if cu_seqlens.device != q.device or cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be 1D [B+1], B>=1, on q.device")
    if not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous")
    cu = cu_seqlens.tolist()
    if cu[0] != 0 or cu[-1] != q.shape[0] or any(b <= a for a, b in zip(cu, cu[1:])):
        raise ValueError(f"cu_seqlens must start at 0, end at T, strictly increase (non-empty seqs): {cu}")
    actual_max = max(b - a for a, b in zip(cu, cu[1:]))
    if max_seqlen is not None and _positive_int(max_seqlen, "max_seqlen") != actual_max:
        raise ValueError("max_seqlen must equal the maximum length in cu_seqlens")
    if config is not None:
        validate_config(config)


def official_attn_varlen(q, k, v, cu_seqlens, max_seqlen, config):
    _validate(q, k, v, cu_seqlens, max_seqlen=max_seqlen, config=config)
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
    _validate(q, k, v, cu_seqlens, max_seqlen=max_seqlen, config=config)
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
