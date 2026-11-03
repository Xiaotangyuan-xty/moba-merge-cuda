"""Native grouped-query MoBA prefill, without a full-sequence K/V repeat.

Local FA consumes Hq/Hkv directly. Remote experts are (chunk, KV head), so
queries sharing a KV head reuse the same packed KV block. Routing retains
one independent top-k decision per query head. Only block representatives
(F,Hkv,D), not the token K/V tensors, expand to Hq for routing.
"""
import torch
from flash_attn import flash_attn_varlen_func
from moba.moba_efficient import calc_chunks, moba_attn_varlen

from .api import register
from .baseline_adapter import _validate
from .mixed_attention import OptMixedAttention, MERGE_BACKEND_CUDA, MERGE_BACKEND_CUDA_SORTED

ROUTE_HOOK = None


def official_gqa_repeat(q, k, v, cu_seqlens, max_seqlen, config):
    """Explicit repeated-K/V reference; its expansion participates in autograd."""
    _validate(q, k, v, cu_seqlens, allow_gqa=True, max_seqlen=max_seqlen, config=config)
    groups = q.shape[1] // k.shape[1]
    return moba_attn_varlen(
        q, k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1),
        cu_seqlens, max_seqlen, config["chunk_size"], config["topk"],
    )


def opt_gqa_cuda(q, k, v, cu_seqlens, max_seqlen, config):
    _validate(q, k, v, cu_seqlens, allow_gqa=True, max_seqlen=max_seqlen, config=config)
    total, hq, dim = q.shape
    hkv = k.shape[1]
    groups = hq // hkv
    chunk_size = int(config["chunk_size"])
    precision=config.get('route_precision','upstream')
    deterministic=config.get('deterministic_merge',False)
    stable=config.get('stable_routes',False)
    if precision not in ('upstream','fp32'):raise ValueError('route_precision')
    if not isinstance(deterministic,bool) or not isinstance(stable,bool):raise ValueError('routing flags must be bool')
    cu_chunk, filtered, num_chunks, chunk_batch = calc_chunks(cu_seqlens, chunk_size)
    remote_k = min(int(config["topk"]) - 1, num_chunks)
    if remote_k == 0:
        # Preserve the fixed upstream K=1 / no-candidate full-sequence fallback.
        return flash_attn_varlen_func(
            q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True
        )

    kv = torch.stack((k, v), dim=1)
    indices = cu_chunk[filtered, None] + torch.arange(chunk_size, device=q.device)[None, :]
    filtered_kv = kv.index_select(0, indices.reshape(-1))
    keys=filtered_kv[:, 0].reshape(num_chunks, chunk_size, hkv, dim)
    means = keys.float().mean(1) if precision=='fp32' else keys.mean(1).float()
    # Match upstream's head-wise dot-product layout; this is F representatives,
    # not T token K/V, and no V replication occurs.
    gate = torch.einsum("nhd,shd->nhs", means.repeat_interleave(groups, dim=1), q.float())
    positions = torch.arange(total, device=q.device)
    ends = cu_chunk[filtered + 1]
    seq_ends = cu_seqlens[chunk_batch[filtered] + 1]
    invalid = (positions[None, :] < ends[:, None]) | (positions[None, :] >= seq_ends[:, None])
    gate.masked_fill_(invalid[:, None, :], -float("inf"))
    if stable:
        # argmax explicitly returns the first (lowest chunk) on exact ties.
        # Repeated reductions avoid materializing a full int64 argsort matrix.
        remaining=gate.clone();chosen=[]
        for _ in range(remote_k):
            index=remaining.argmax(dim=0,keepdim=True)
            chosen.append(index)
            remaining.scatter_(0,index,-float('inf'))
        top=torch.cat(chosen,dim=0)
    else:top = torch.topk(gate, remote_k, dim=0, sorted=False).indices
    selected = torch.zeros_like(gate, dtype=torch.bool).scatter_(0, top, True) & ~gate.isinf()
    if ROUTE_HOOK is not None:
        ROUTE_HOOK(cu_chunk=cu_chunk, filtered_chunk_indices=filtered,
                   chunk_to_batch=chunk_batch, selected=selected)

    # Expert order: (chunk, kv_head); within expert: (group_head, token).
    # Each KV block is packed once for all its selected query heads.
    expert_mask = selected.reshape(num_chunks * hkv, groups * total)
    expert, inner = expert_mask.nonzero(as_tuple=True)
    mapping = inner.remainder(total) * hq + expert.remainder(hkv) * groups + inner.div(total, rounding_mode="floor")
    remote_q = q.reshape(total * hq, dim).index_select(0, mapping).unsqueeze(1)
    counts = expert_mask.sum(-1)
    active = counts.nonzero(as_tuple=True)[0]
    active_counts = counts.index_select(0, active)
    cu_q = torch.cat((torch.zeros(1, dtype=torch.int32, device=q.device),
                      active_counts.cumsum(0).to(torch.int32)))
    remote_kv = filtered_kv.reshape(num_chunks, chunk_size, 2, hkv, dim)
    remote_kv = remote_kv.permute(0, 3, 1, 2, 4).reshape(num_chunks * hkv, chunk_size, 2, dim)
    remote_kv = remote_kv.index_select(0, active).reshape(-1, 2, 1, dim)
    cu_kv = torch.arange(active.numel() + 1, device=q.device, dtype=torch.int32) * chunk_size

    # An expert now contains up to groups*max_seqlen query rows. Passing the
    # upstream max_seqlen would understate this length for grouped experts.
    remote_max = int(active_counts.max().item())
    return OptMixedAttention.apply(
        q, k, v, cu_chunk, remote_q, remote_kv, cu_q, cu_kv,
        max_seqlen, chunk_size, mapping, remote_k,
        MERGE_BACKEND_CUDA_SORTED if deterministic else MERGE_BACKEND_CUDA, remote_max,
    )


register("official_gqa_repeat", official_gqa_repeat)
register("opt_gqa_cuda", opt_gqa_cuda)
