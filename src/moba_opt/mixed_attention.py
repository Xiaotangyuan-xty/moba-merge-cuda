"""Optimized MixedAttention path (route A, V3): official routing and backward
structure preserved; the forward merge is replaced by the Triton
global-softmax merge (plan 4.2 route A, 4.5).

Routing (gate, topk, packing) is kept line-identical to the frozen upstream
moba_efficient.py @ b5d5836 so selected block sets are bit-identical, and
calc_chunks is imported from upstream (shared metadata cache). The k=1
fallback branch is untouched (plan 3.4).

Backward keeps the official structure and formulas: both private FA backwards
receive the GLOBAL output and GLOBAL LSE produced by the new forward merge
(plan 4.5). The merge subkernel needs no separate backward: it sits inside
this autograd.Function whose backward is fully custom (plan 4.5 last item).
"""

import torch
from einops import rearrange
from flash_attn import flash_attn_varlen_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_backward,
    _flash_attn_varlen_forward,
)

from moba.moba_efficient import calc_chunks  # upstream, shared metadata cache

from .api import register
from .baseline_adapter import _validate
from .ops.merge_triton import merge_forward

ROUTE_HOOK = None  # test-only; None in all performance paths

# merge backends for OptMixedAttention: 0 = Triton (atomic slot build),
# 1 = CUDA extension (ops/merge.cu). Passed explicitly per call so both can
# run in one process for paired benchmarks.
MERGE_BACKEND_TRITON = 0
MERGE_BACKEND_CUDA = 1
MERGE_BACKEND_CUDA_SORTED = 2


class OptMixedAttention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
        moba_topk,
        merge_backend,
        remote_max_seqlen=None,
    ):
        ctx.max_seqlen = max_seqlen
        ctx.remote_max_seqlen = max_seqlen if remote_max_seqlen is None else remote_max_seqlen
        ctx.has_remote_max = remote_max_seqlen is not None
        ctx.moba_chunk_size = moba_chunk_size
        ctx.softmax_scale = softmax_scale = q.shape[-1] ** (-0.5)

        # self attn (identical to official)
        _, _, _, _, self_attn_out_sh, self_attn_lse_hs, _, _ = (
            _flash_attn_varlen_forward(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=self_attn_cu_seqlen,
                cu_seqlens_k=self_attn_cu_seqlen,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                softmax_scale=softmax_scale,
                causal=True,
                dropout_p=0.0,
            )
        )

        # moba attn (identical to official)
        _, _, _, _, moba_attn_out, moba_attn_lse_hs, _, _ = _flash_attn_varlen_forward(
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=ctx.remote_max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
        )

        # replaced stage: global-softmax merge (plan 4.3 math), Triton or CUDA
        # check_capacity=False: the official routing above selects at most
        # moba_topk chunks per query/head, so counts <= moba_topk = kmax <= kpad
        # holds structurally; the standalone merge APIs keep the check on.
        if merge_backend in (MERGE_BACKEND_CUDA, MERGE_BACKEND_CUDA_SORTED):
            from .ops.merge_cuda import merge_forward_cuda

            output, mixed_attn_lse_sh = merge_forward_cuda(
                self_attn_out_sh,
                self_attn_lse_hs.t().contiguous(),
                moba_attn_out,
                moba_attn_lse_hs.reshape(-1),
                moba_q_sh_indices,
                kmax=moba_topk,
                check_capacity=False,
                deterministic=merge_backend == MERGE_BACKEND_CUDA_SORTED,
            )
        else:
            output, mixed_attn_lse_sh = merge_forward(
                self_attn_out_sh,
                self_attn_lse_hs.t().contiguous(),
                moba_attn_out,
                moba_attn_lse_hs.reshape(-1),
                moba_q_sh_indices,
                kmax=moba_topk,
                builder="atomic",
                check_capacity=False,
            )

        ctx.save_for_backward(
            output,
            mixed_attn_lse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        )

        return output

    @staticmethod
    def backward(ctx, d_output):
        # identical to the official MixedAttention.backward (plan 4.5):
        # both private FA backwards use the merged GLOBAL output and LSE.
        max_seqlen = ctx.max_seqlen
        moba_chunk_size = ctx.moba_chunk_size
        softmax_scale = ctx.softmax_scale

        (
            output,
            mixed_attn_vlse_sh,
            q,
            k,
            v,
            self_attn_cu_seqlen,
            moba_q,
            moba_kv,
            moba_cu_seqlen_q,
            moba_cu_seqlen_kv,
            moba_q_sh_indices,
        ) = ctx.saved_tensors

        d_output = d_output.contiguous()

        dq, dk, dv, _ = _flash_attn_varlen_backward(
            dout=d_output,
            q=q,
            k=k,
            v=v,
            out=output,
            softmax_lse=mixed_attn_vlse_sh.t().contiguous(),
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=self_attn_cu_seqlen,
            cu_seqlens_k=self_attn_cu_seqlen,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=softmax_scale,
            causal=True,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        headdim = q.shape[-1]
        d_moba_output = (
            d_output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )
        moba_output = (
            output.view(-1, headdim).index_select(0, moba_q_sh_indices).unsqueeze(1)
        )

        mixed_attn_vlse = (
            mixed_attn_vlse_sh.view(-1).index_select(0, moba_q_sh_indices).view(1, -1)
        )

        dmq, dmk, dmv, _ = _flash_attn_varlen_backward(
            dout=d_moba_output,
            q=moba_q,
            k=moba_kv[:, 0],
            v=moba_kv[:, 1],
            out=moba_output,
            softmax_lse=mixed_attn_vlse,
            dq=None,
            dk=None,
            dv=None,
            cu_seqlens_q=moba_cu_seqlen_q,
            cu_seqlens_k=moba_cu_seqlen_kv,
            max_seqlen_q=ctx.remote_max_seqlen,
            max_seqlen_k=moba_chunk_size,
            softmax_scale=softmax_scale,
            causal=False,
            dropout_p=0.0,
            window_size=(-1, -1),
            softcap=0.0,
            alibi_slopes=None,
            deterministic=True,
        )

        dmkv = torch.stack((dmk, dmv), dim=1)
        grads = (dq, dk, dv, None, dmq, dmkv, None, None, None, None, None, None, None)
        return grads + (None,) if ctx.has_remote_max else grads


def moba_attn_varlen_opt(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    moba_chunk_size: int,
    moba_topk: int,
    merge_backend: int = MERGE_BACKEND_TRITON,
) -> torch.Tensor:
    """Same interface and routing as upstream moba_attn_varlen; only the
    forward merge stage and the MixedAttention class differ."""

    kv = torch.stack((k, v), dim=1)

    """ some basic variables """
    # qkv shape = [ S, H, D ]
    seqlen, num_head, head_dim = q.shape

    """ prepare chunk meta """
    (
        cu_chunk,
        filtered_chunk_indices,
        num_filtered_chunk,
        chunk_to_batch,
    ) = calc_chunks(cu_seqlens, moba_chunk_size)

    # we will adjust selective topk to moba_topk - 1, as the last chunk is always chosen
    moba_topk = min(moba_topk - 1, num_filtered_chunk)
    need_moba_attn = moba_topk > 0

    # corner case: if no moba attn needed, just return self attn (plan 3.4: untouched)
    if not need_moba_attn:
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    self_attn_cu_seqlen = cu_chunk

    # filtered_kv is a dense matrix that only contains filtered chunk of kv
    filtered_kv_indices = torch.arange(
        0, moba_chunk_size, dtype=torch.int32, device=q.device
    )[None, :].repeat(num_filtered_chunk, 1)
    filtered_kv_indices += cu_chunk[filtered_chunk_indices][:, None]
    filtered_kv = kv.index_select(0, filtered_kv_indices.view(-1))

    """ calc key_gate_weight and gate """

    # key_gate_weight [ F_N_CHUNK, HEAD, HEAD_DIM ]
    key_gate_weight = (
        filtered_kv[:, 0]
        .view(num_filtered_chunk, moba_chunk_size, num_head, head_dim)
        .mean(dim=1)
        .float()
    )
    q = q.type(torch.float32)  # float logit on the fly for better gate logit perception
    key_gate_weight = key_gate_weight.type(
        torch.float32
    )  # float logit for better gate logit perception
    gate = torch.einsum(
        "nhd,shd->nhs", key_gate_weight, q
    )  # gate [ F_N_CHUNK, HEAD, SEQ ]
    key_gate_weight = key_gate_weight.type_as(k)
    q = q.type_as(k)

    # pose process gate, masking unchosen batch and apply causal mask to current chunk
    gate_seq_idx = torch.arange(0, seqlen, device=q.device, dtype=torch.int32)[
        None, :
    ].repeat(num_filtered_chunk, 1)
    chunk_end = cu_chunk[filtered_chunk_indices + 1]
    batch_end = cu_seqlens[chunk_to_batch[filtered_chunk_indices] + 1]
    gate_chunk_end_mask = gate_seq_idx < chunk_end[:, None]
    gate_batch_end_mask = gate_seq_idx >= batch_end[:, None]
    gate_inf_mask = gate_chunk_end_mask | gate_batch_end_mask
    gate.masked_fill_(gate_inf_mask.unsqueeze(1), -float("inf"))

    """ find moba q that needs moba attn """
    # find topk chunks
    # gate_mask [ N_CHUNK, HEAD, SEQ ], true indicates that needs attention
    _, gate_top_k_idx = torch.topk(gate, k=moba_topk, dim=0, largest=True, sorted=False)
    # apply causal mask
    gate_mask = torch.logical_not(gate.isinf())
    # select topk chunks
    gate_idx_mask = torch.zeros(gate_mask.shape, dtype=torch.bool, device=q.device)
    gate_idx_mask = gate_idx_mask.scatter_(dim=0, index=gate_top_k_idx, value=True)
    gate_mask = torch.logical_and(gate_mask, gate_idx_mask)

    if ROUTE_HOOK is not None:
        ROUTE_HOOK(
            cu_chunk=cu_chunk,
            filtered_chunk_indices=filtered_chunk_indices,
            chunk_to_batch=chunk_to_batch,
            selected=gate_mask,
        )

    # varlen trick: combining all q index that needs moba attn
    # the result will be like [ C0H0 ][ C0H1 ][ C0H2 ][ ... ][ CnHm ]
    moba_q_indices = gate_mask.reshape(gate_mask.shape[0], -1).nonzero(as_tuple=True)[
        -1
    ]  # [ HS indices ] * N
    # moba_seqlen_q indicates that how many q chunks are selected for each kv chunk - head
    moba_seqlen_q = gate_mask.sum(dim=-1).flatten()
    # select all q that needs moba attn based on the moba_q_indices
    moba_q = rearrange(q, "s h d -> ( h s ) d").index_select(
        0, moba_q_indices
    )  # [ selected_S, D ]
    moba_q = moba_q.unsqueeze(1)
    # moba_q_sh_indices represents the position in the origin q tensor of each q token inside moba_q
    moba_q_sh_indices = moba_q_indices % seqlen * num_head + moba_q_indices // seqlen

    """ prepare moba kv """
    # Since moba_q is organized as HS * N, we need to reorganize kv to adapt to q

    # cut off zero experts
    q_zero_mask = moba_seqlen_q == 0
    valid_expert_mask = ~q_zero_mask
    zero_expert_count = q_zero_mask.sum()
    # only keep the kv that has q select > 0
    if zero_expert_count > 0:
        moba_seqlen_q = moba_seqlen_q[valid_expert_mask]
    # moba cu_seqlen for flash attn
    moba_cu_seqlen_q = torch.cat(
        (
            torch.tensor([0], device=q.device, dtype=moba_seqlen_q.dtype),
            moba_seqlen_q.cumsum(dim=0),
        ),
        dim=0,
    ).to(torch.int32)
    moba_kv = rearrange(filtered_kv, "s x h d -> h s x d")
    moba_kv = moba_kv.split(moba_chunk_size, dim=1)
    moba_kv = torch.cat(moba_kv, dim=0)
    if zero_expert_count > 0:
        assert valid_expert_mask.sum() == moba_kv.shape[0] - zero_expert_count
        moba_kv = moba_kv[
            valid_expert_mask
        ]  # cut off zero Q expert from kv , or the grad may be nan
    moba_kv = moba_kv.flatten(start_dim=0, end_dim=1).unsqueeze(2)
    moba_cu_seqlen_kv = (
        torch.arange(
            0,
            num_filtered_chunk * num_head + 1 - zero_expert_count,
            dtype=torch.int32,
            device=q.device,
        )
        * moba_chunk_size
    )

    # Shape check
    assert (
        moba_cu_seqlen_kv.shape == moba_cu_seqlen_q.shape
    ), f"moba_cu_seqlen_kv.shape != moba_cu_seqlen_q.shape {moba_cu_seqlen_kv.shape} != {moba_cu_seqlen_q.shape}"

    return OptMixedAttention.apply(
        q,
        k,
        v,
        self_attn_cu_seqlen,
        moba_q,
        moba_kv,
        moba_cu_seqlen_q,
        moba_cu_seqlen_kv,
        max_seqlen,
        moba_chunk_size,
        moba_q_sh_indices,
        moba_topk,
        merge_backend,
    )


def _opt_adapter(q, k, v, cu_seqlens, max_seqlen, config, merge_backend):
    _validate(q, k, v, cu_seqlens, max_seqlen=max_seqlen, config=config)
    return moba_attn_varlen_opt(
        q,
        k,
        v,
        cu_seqlens,
        int(max_seqlen),
        moba_chunk_size=int(config["chunk_size"]),
        moba_topk=int(config["topk"]),
        merge_backend=merge_backend,
    )


def opt_merge_atomic(q, k, v, cu_seqlens, max_seqlen, config):
    return _opt_adapter(q, k, v, cu_seqlens, max_seqlen, config, MERGE_BACKEND_TRITON)


def opt_merge_cuda(q, k, v, cu_seqlens, max_seqlen, config):
    return _opt_adapter(q, k, v, cu_seqlens, max_seqlen, config, MERGE_BACKEND_CUDA)


register("opt_merge_atomic", opt_merge_atomic)
register("opt_merge_cuda", opt_merge_cuda)
