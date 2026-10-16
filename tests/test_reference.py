"""CPU/fp64 self-validation for the layer-3 independent reference (plan 7.1).

Two independent implementations are checked against each other (vectorized
mask builder vs per-query gather loop), plus dense-equivalence for an
all-legal route and finite-difference checks of the merge VJP formulas.
No GPU, no official MoBA code involved.
"""

import itertools

import pytest
import torch

from reference.fixed_route_attention import fixed_route_attention
from reference.merge_reference import merge_reference, merge_vjp_reference

ATOL = 1e-12


def build_chunks(lens, C):
    """Chunk metadata with official semantics (last chunk of each seq is not a candidate)."""
    cu = [0] + list(itertools.accumulate(lens))
    cu_chunk = [0]
    chunk_to_batch = []
    for b, L in enumerate(lens):
        n = (L + C - 1) // C
        for i in range(n):
            chunk_to_batch.append(b)
            cu_chunk.append(cu_chunk[-1] + min(C, L - i * C))
    last = {max(c for c in range(len(chunk_to_batch)) if chunk_to_batch[c] == b) for b in range(len(lens))}
    filtered = [c for c in range(len(chunk_to_batch)) if c not in last]
    return cu, cu_chunk, chunk_to_batch, filtered


def make_route(lens, C, H, select_fn):
    """select_fn(f, c, b, chunk_end, s) -> bool; builds a selected tensor over legal rows."""
    cu, cu_chunk, chunk_to_batch, filtered = build_chunks(lens, C)
    T = cu[-1]
    selected = torch.zeros(len(filtered), H, T, dtype=torch.bool)
    for f, c in enumerate(filtered):
        b = chunk_to_batch[c]
        end = cu_chunk[c + 1]
        for s in range(cu[b], cu[b + 1]):
            if s >= end and select_fn(f, c, b, end, s):
                selected[f, :, s] = True
    return {
        "cu_chunk": torch.tensor(cu_chunk),
        "chunk_to_batch": torch.tensor(chunk_to_batch),
        "filtered_chunk_indices": torch.tensor(filtered),
        "selected": selected,
        "cu_seqlens": torch.tensor(cu),
        "chunk_size": C,
        "topk": 4,
        "max_seqlen": max(lens),
    }


def dense_causal_ref(q, k, v, cu, scale):
    """Standard per-sequence causal attention, independent mask construction."""
    T, H, D = q.shape
    o = torch.zeros(T, H, D, dtype=torch.float64)
    lse = torch.zeros(T, H, dtype=torch.float64)
    for b in range(len(cu) - 1):
        s0, s1 = int(cu[b]), int(cu[b + 1])
        L = s1 - s0
        tril = torch.tril(torch.ones(L, L, dtype=torch.bool))
        for h in range(H):
            logits = q[s0:s1, h] @ k[s0:s1, h].T * scale
            logits = logits.masked_fill(~tril, float("-inf"))
            lse[s0:s1, h] = torch.logsumexp(logits, dim=-1)
            o[s0:s1, h] = torch.softmax(logits, dim=-1) @ v[s0:s1, h]
    return o, lse


def gather_loop_ref(q, k, v, cu, route, scale):
    """Per-(batch, head, query) gather of exactly the routed tokens; slowest and
    most independent implementation."""
    T, H, D = q.shape
    cu_chunk = route["cu_chunk"].tolist()
    chunk_to_batch = route["chunk_to_batch"].tolist()
    filtered = route["filtered_chunk_indices"].tolist()
    selected = route["selected"]
    o = torch.zeros(T, H, D, dtype=torch.float64)
    lse = torch.zeros(T, H, dtype=torch.float64)

    def chunk_of(s, b):
        for c in range(len(chunk_to_batch)):
            if chunk_to_batch[c] == b and cu_chunk[c] <= s < cu_chunk[c + 1]:
                return c
        raise AssertionError(f"no chunk for s={s} b={b}")

    for b in range(len(cu) - 1):
        s0, s1 = int(cu[b]), int(cu[b + 1])
        for h in range(H):
            for s in range(s0, s1):
                c = chunk_of(s, b)
                idx = list(range(cu_chunk[c], s + 1))
                for f, cc in enumerate(filtered):
                    if chunk_to_batch[cc] == b and bool(selected[f, h, s]):
                        idx += list(range(cu_chunk[cc], cu_chunk[cc + 1]))
                idx = torch.tensor(idx, dtype=torch.long)
                logits = (q[s, h] * k[idx, h]).sum(-1) * scale
                l = torch.logsumexp(logits, dim=0)
                o[s, h] = torch.exp(logits - l) @ v[idx, h]
                lse[s, h] = l
    return o, lse


@pytest.mark.parametrize("lens,C,H,D", [([9, 5], 4, 3, 8), ([1], 4, 2, 4), ([7], 3, 2, 8), ([4, 4, 1], 4, 2, 4)])
def test_all_legal_route_equals_dense(lens, C, H, D):
    torch.manual_seed(0)
    T = sum(lens)
    q, k, v = (torch.randn(T, H, D, dtype=torch.float64) for _ in range(3))
    route = make_route(lens, C, H, select_fn=lambda *a: True)
    cu = torch.tensor([0] + list(itertools.accumulate(lens)))
    scale = D ** -0.5
    o_ref, lse_ref = dense_causal_ref(q, k, v, cu.tolist(), scale)
    o, lse = fixed_route_attention(q, k, v, cu, route, softmax_scale=scale)
    assert torch.allclose(o, o_ref, atol=ATOL)
    assert torch.allclose(lse, lse_ref, atol=ATOL)


def test_random_route_equals_gather_loop():
    lens, C, H, D = [6, 8, 1], 4, 3, 4
    torch.manual_seed(1)
    T = sum(lens)
    q, k, v = (torch.randn(T, H, D, dtype=torch.float64) for _ in range(3))
    rng = torch.Generator().manual_seed(2)
    route = make_route(lens, C, H, select_fn=lambda *a: bool(torch.rand((), generator=rng) < 0.5))
    cu = torch.tensor([0] + list(itertools.accumulate(lens)))
    scale = D ** -0.5
    o, lse = fixed_route_attention(q, k, v, cu, route, softmax_scale=scale)
    o_ref, lse_ref = gather_loop_ref(q, k, v, cu.tolist(), route, scale)
    assert torch.allclose(o, o_ref, atol=ATOL)
    assert torch.allclose(lse, lse_ref, atol=ATOL)


def test_route_validation_rejects_future():
    route = make_route([9], 4, 1, select_fn=lambda *a: False)
    # illegally select chunk 0 (end=4) for query s=1 (inside/before the chunk end)
    route["selected"][0, 0, 1] = True
    cu = torch.tensor([0, 9])
    q = torch.randn(9, 1, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="future"):
        fixed_route_attention(q, q, q, cu, route)


def test_route_validation_rejects_cross_sequence():
    lens, C, H = [5, 5], 4, 1
    route = make_route(lens, C, H, select_fn=lambda *a: False)
    # chunk 0 belongs to seq 0 (end=4); s=6 is in seq 1 -> cross-sequence selection
    route["selected"][0, 0, 6] = True
    cu = torch.tensor([0, 5, 10])
    q = torch.randn(10, 1, 4, dtype=torch.float64)
    with pytest.raises(ValueError, match="crosses sequence"):
        fixed_route_attention(q, q, q, cu, route)


def test_merge_single_branch_identity():
    torch.manual_seed(3)
    o1, l1 = torch.randn(5, 4, dtype=torch.float64), torch.randn(5, dtype=torch.float64)
    O, L = merge_reference([(o1, l1)])
    assert torch.equal(O, o1)
    assert torch.allclose(L, l1, atol=ATOL)


def test_merge_two_branch_manual():
    torch.manual_seed(4)
    N, D = 6, 3
    o1, l1 = torch.randn(N, D, dtype=torch.float64), torch.randn(N, dtype=torch.float64)
    o2, l2 = torch.randn(N, D, dtype=torch.float64), torch.randn(N, dtype=torch.float64)
    O, L = merge_reference([(o1, l1), (o2, l2)])
    O_exp = torch.empty_like(O)
    L_exp = torch.empty_like(L)
    for i in range(N):
        m = max(l1[i], l2[i])
        w1, w2 = torch.exp(l1[i] - m), torch.exp(l2[i] - m)
        O_exp[i] = (w1 * o1[i] + w2 * o2[i]) / (w1 + w2)
        L_exp[i] = m + torch.log(w1 + w2)
    assert torch.allclose(O, O_exp, atol=ATOL)
    assert torch.allclose(L, L_exp, atol=ATOL)


def test_merge_extreme_lse_no_nan():
    torch.manual_seed(5)
    o1 = torch.randn(4, 2, dtype=torch.float64)
    o2 = torch.randn(4, 2, dtype=torch.float64)
    l1 = torch.full((4,), -1e30, dtype=torch.float64)
    l2 = torch.full((4,), 1e30, dtype=torch.float64)
    O, L = merge_reference([(o1, l1), (o2, l2)])
    assert torch.isfinite(O).all() and torch.isfinite(L).all()
    assert torch.allclose(O, o2, atol=1e-8)


def test_merge_vjp_finite_difference():
    torch.manual_seed(6)
    N, D = 3, 2
    o1, l1 = torch.randn(N, D, dtype=torch.float64), torch.randn(N, dtype=torch.float64)
    o2, l2 = torch.randn(N, D, dtype=torch.float64), torch.randn(N, dtype=torch.float64)
    g = torch.randn(N, D, dtype=torch.float64)

    def loss(a1, b1, a2, b2):
        O, _ = merge_reference([(a1, b1), (a2, b2)])
        return (O * g).sum()

    O, L = merge_reference([(o1, l1), (o2, l2)])
    d_os, d_ls = merge_vjp_reference([(o1, l1), (o2, l2)], O, L, g)

    eps = 1e-6
    for tensors, grads in (((o1, o2), d_os), ((l1, l2), d_ls)):
        for t, grad in zip(tensors, grads):
            fd = torch.zeros_like(t)
            for idx in itertools.product(*[range(s) for s in t.shape]):
                orig = t[idx].item()
                t[idx] = orig + eps
                up = loss(o1, l1, o2, l2)
                t[idx] = orig - eps
                dn = loss(o1, l1, o2, l2)
                t[idx] = orig
                fd[idx] = (up - dn) / (2 * eps)
            assert torch.allclose(grad, fd, atol=1e-6, rtol=1e-4), f"VJP mismatch: {grad} vs {fd}"
