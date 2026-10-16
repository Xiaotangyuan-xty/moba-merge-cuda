"""Layer-3 independent reference (plan 7.1): fixed-route attention in float64.

Given an exported route (the legal/selected block sets produced by the official
routing code), explicitly build the sparse mask and compute attention per
sequence. Shares no code with the optimized or official implementations, so a
bug in their merge/pack logic cannot be inherited here.

Small inputs only: the mask is [L, L] per (batch, head).
"""

import torch

ROUTE_KEYS = (
    "cu_chunk",
    "chunk_to_batch",
    "filtered_chunk_indices",
    "selected",
    "cu_seqlens",
    "chunk_size",
    "topk",
    "max_seqlen",
)


def _as_long_cpu(t):
    return t.detach().to(device="cpu", dtype=torch.long)


def validate_route(route, total_tokens, num_heads):
    """Route legality (plan 7.5 items 6-7): no cross-sequence, no future blocks.

    selected[f, h, s] == True means query s of head h attends the *entire*
    filtered chunk filtered_chunk_indices[f]; such rows must lie at or after
    that chunk's end (chunk fully in the past) and inside the same sequence.
    """
    cu_chunk = _as_long_cpu(route["cu_chunk"])
    chunk_to_batch = _as_long_cpu(route["chunk_to_batch"])
    filtered = _as_long_cpu(route["filtered_chunk_indices"])
    selected = route["selected"].detach().to(device="cpu", dtype=torch.bool)
    cu = _as_long_cpu(route["cu_seqlens"])

    if selected.shape != (filtered.numel(), num_heads, total_tokens):
        raise ValueError(
            f"selected shape {tuple(selected.shape)} != (F,H,T) "
            f"{(filtered.numel(), num_heads, total_tokens)}"
        )
    if int(cu[-1]) != total_tokens:
        raise ValueError("cu_seqlens[-1] != T")
    problems = []
    for f in range(filtered.numel()):
        c = int(filtered[f])
        b = int(chunk_to_batch[c])
        chunk_end = int(cu_chunk[c + 1])
        seq_start, seq_end = int(cu[b]), int(cu[b + 1])
        rows = selected[f].nonzero(as_tuple=False)
        for h, s in rows.tolist():
            if s < chunk_end:
                problems.append(f"route sees future: chunk {c} (end {chunk_end}) selected by s={s}")
            if not (seq_start <= s < seq_end):
                problems.append(f"route crosses sequence: chunk {c} (batch {b}) selected by s={s}")
    if problems:
        raise ValueError("illegal route:\n" + "\n".join(problems[:20]))
    return cu_chunk, chunk_to_batch, filtered, selected, cu


def fixed_route_attention(q, k, v, cu_seqlens, route, softmax_scale=None, compute_dtype=torch.float64):
    """Compute O and global LSE under the exported route, in high precision.

    Args:
        q, k, v: [T, H, D] (any dtype/device; cast to compute_dtype on CPU)
        cu_seqlens: [B+1]
        route: dict with ROUTE_KEYS, as produced by reference.export_route
    Returns:
        O [T, H, D] and LSE [T, H], both compute_dtype on CPU.
    """
    total, heads, dim = q.shape
    if softmax_scale is None:
        softmax_scale = dim ** -0.5
    cu_chunk, chunk_to_batch, filtered, selected, cu = validate_route(route, total, heads)

    qc = q.detach().to(device="cpu", dtype=compute_dtype)
    kc = k.detach().to(device="cpu", dtype=compute_dtype)
    vc = v.detach().to(device="cpu", dtype=compute_dtype)
    cu_ref = _as_long_cpu(cu_seqlens)
    if not torch.equal(cu_ref, cu):
        raise ValueError("cu_seqlens argument and route['cu_seqlens'] differ")

    o = torch.zeros_like(qc)
    lse = torch.zeros((total, heads), dtype=compute_dtype)

    batch_of_chunk = chunk_to_batch
    num_chunk = chunk_to_batch.numel()
    chunks_by_batch = {}
    for c in range(num_chunk):
        chunks_by_batch.setdefault(int(batch_of_chunk[c]), []).append(c)

    for b, (s0, s1) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        length = s1 - s0
        qb, kb, vb = qc[s0:s1], kc[s0:s1], vc[s0:s1]
        # current-block causal part, shared across heads: block i rows attend cols <= row within block i
        causal = torch.zeros((length, length), dtype=torch.bool)
        for c in chunks_by_batch.get(b, []):
            a0, a1 = int(cu_chunk[c]) - s0, int(cu_chunk[c + 1]) - s0
            rows = torch.arange(a0, a1)
            causal[a0:a1, a0:a1] |= rows[:, None] >= rows[None, :]
        for h in range(heads):
            mask = causal.clone()
            for f in range(filtered.numel()):
                c = int(filtered[f])
                if int(chunk_to_batch[c]) != b:
                    continue
                a0, a1 = int(cu_chunk[c]) - s0, int(cu_chunk[c + 1]) - s0
                rows = selected[f, h, s0:s1]
                mask[rows, a0:a1] = True
            logits = torch.einsum("sd,td->st", qb[:, h], kb[:, h]) * softmax_scale
            logits = logits.masked_fill(~mask, float("-inf"))
            lse[s0:s1, h] = torch.logsumexp(logits, dim=-1)
            o[s0:s1, h] = torch.softmax(logits, dim=-1) @ vb[:, h]
    return o, lse
