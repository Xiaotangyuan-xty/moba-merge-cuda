"""Differentiable CPU FP64 fixed-route GQA oracle; no FlashAttention/merge reuse."""
import torch
from .fixed_route_attention import validate_route


def grouped_reference(q, k, v, cu_seqlens, route):
    total, hq, dim = q.shape
    hkv = k.shape[1]
    if hq % hkv:
        raise ValueError("Hq must divide into KV groups")
    chunks, chunk_batch, filtered, selected, cu = validate_route(route, total, hq)
    if not torch.equal(cu, cu_seqlens.cpu().long()):
        raise ValueError("route sequence boundaries differ")
    # Deliberately preserve autograd, unlike the earlier output-only oracle.
    qc, kc, vc = (t.to(device="cpu", dtype=torch.float64) for t in (q, k, v))
    groups = hq // hkv
    parts = []
    for b, (start, end) in enumerate(zip(cu[:-1].tolist(), cu[1:].tolist())):
        length = end - start
        local_mask = torch.zeros(length, length, dtype=torch.bool)
        for c, bb in enumerate(chunk_batch.tolist()):
            if bb == b:
                lo, hi = int(chunks[c]) - start, int(chunks[c + 1]) - start
                local_mask[lo:hi, lo:hi] = torch.ones(hi-lo, hi-lo, dtype=torch.bool).tril()
        heads = []
        for h in range(hq):
            mask = local_mask.clone()
            for f, c in enumerate(filtered.tolist()):
                if int(chunk_batch[c]) == b:
                    lo, hi = int(chunks[c]) - start, int(chunks[c + 1]) - start
                    mask[selected[f, h, start:end], lo:hi] = True
            logits = qc[start:end, h] @ kc[start:end, h // groups].T * dim**-0.5
            probs = torch.softmax(logits.masked_fill(~mask, -float("inf")), dim=-1)
            heads.append(probs @ vc[start:end, h // groups])
        parts.append(torch.stack(heads, dim=1))
    return torch.cat(parts, dim=0)
