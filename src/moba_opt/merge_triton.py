"""Triton prototype for route A (plan 4.2/4.4): global-softmax merge.

V2 scope (plan 9.3): slots prebuilt with deterministic torch ops (builder="torch").
V3 scope: slots built on device by a single atomic-counter kernel
(builder="atomic"); the merge kernel masks lanes by per-row counts capped at
the slot capacity, so no out-of-bounds read is possible even on misuse. Both
builders raise on capacity overflow when check_capacity is on (default).

Capacity contract: each global row p may contribute at most kpad slots
(kpad = pow2 ceil of kmax). The official MoBA routing structurally guarantees
counts <= topk-1 = kmax <= kpad, so the integrated path passes
check_capacity=False; the standalone API keeps the check on.

Atomic assignment order is arbitrary: it only permutes lane-level accumulation
order (small fp32-order fluctuation; the official merge's atomic index_add_ is
order-nondeterministic too). Semantics are preserved; outputs align within the
stated test tolerances -- no bitwise or per-ulp guarantee is claimed.

Math (plan 4.3): m = max_j l_j; w_j = exp(l_j - m); O = sum_j w_j o_j / z;
z = sum_j w_j; L = m + log(z). Compute in fp32; O cast to input dtype,
global LSE fp32.
"""

import torch
import triton
import triton.language as tl

INVALID = -1


def build_slots_torch(mapping, P, kmax, kpad=None):
    """Deterministic inverse mapping (reference/V2 path).

    slots[p, j] = remote row index of the j-th remote contribution to global
    row p (j ordered by ascending remote row index, independent of the input
    mapping order), or INVALID when absent. counts[p] = number of remote
    contributions. Raises when a row exceeds the allocated slot capacity
    (stride = kpad or kmax); never silently drops (plan 7.2).
    """
    dev = mapping.device
    M = mapping.numel()
    stride = kpad or kmax
    slots = torch.full((P, stride), INVALID, dtype=torch.int32, device=dev)
    counts = torch.zeros(P, dtype=torch.int32, device=dev)
    if M == 0:
        return slots, counts
    mapping = mapping.reshape(-1).to(torch.int64)
    key = mapping * M + torch.arange(M, device=dev)  # lexicographic (p, remote row)
    order = torch.argsort(key)
    sorted_p = mapping[order]
    unique_p, cnt = torch.unique_consecutive(sorted_p, return_counts=True)
    starts = torch.cumsum(cnt, 0) - cnt
    gids = torch.repeat_interleave(torch.arange(unique_p.numel(), device=dev), cnt)
    ranks = torch.arange(M, device=dev) - starts[gids]
    if bool((ranks >= stride).any()):
        raise ValueError(f"capacity exceeded: a query row has more than {stride} remote contributions")
    slots[unique_p[gids], ranks] = order.to(torch.int32)
    counts[unique_p] = cnt.to(torch.int32)
    return slots, counts


@triton.jit
def _build_slots_atomic_kernel(
    mapping_ptr,
    slots_ptr,
    counts_ptr,
    M,
    KPAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One pass over M remote rows: slot = atomic ticket per global row.

    counts must be zeroed before launch. Stores are masked by slot < KPAD;
    overflow shows up as counts > KPAD afterwards and is checked by the caller
    (check_capacity) instead of writing out of bounds.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    p = tl.load(mapping_ptr + offs, mask=mask, other=0)
    slot = tl.atomic_add(counts_ptr + p, 1, mask=mask)
    tl.store(slots_ptr + p * KPAD + slot, offs.to(tl.int32), mask=mask & (slot < KPAD))


@triton.jit
def _merge_kernel(
    self_out_ptr,
    self_lse_ptr,
    remote_out_ptr,
    remote_lse_ptr,
    slots_ptr,
    counts_ptr,
    out_ptr,
    lse_ptr,
    D: tl.constexpr,
    KPAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, KPAD)

    # defensive cap: with the capacity check (or the structural bound of the
    # official routing) cnt <= KPAD always; this only keeps an unchecked
    # misuse from reading out of bounds.
    cnt = tl.minimum(tl.load(counts_ptr + pid), KPAD)
    valid = offs_k < cnt
    slot = tl.load(slots_ptr + pid * KPAD + offs_k, mask=valid, other=0)
    l_self = tl.load(self_lse_ptr + pid)
    l_rem = tl.load(remote_lse_ptr + slot, mask=valid, other=float("-inf"))

    m = tl.maximum(l_self, tl.max(l_rem, axis=0))
    w_self = tl.exp(l_self - m)
    w_rem = tl.where(valid, tl.exp(l_rem - m), 0.0)
    z = w_self + tl.sum(w_rem, axis=0)

    o_self = tl.load(self_out_ptr + pid * D + offs_d).to(tl.float32)
    r = tl.load(
        remote_out_ptr + slot[:, None] * D + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    o = (w_self * o_self + tl.sum(w_rem[:, None] * r, axis=0)) / z

    tl.store(out_ptr + pid * D + offs_d, o.to(out_ptr.dtype.element_ty))
    tl.store(lse_ptr + pid, m + tl.log(z))


def merge_forward(self_out, self_lse, remote_out, remote_lse, mapping, kmax=None, builder="torch", check_capacity=True):
    """Merge branch outputs into global O and LSE.

    self_out [T,H,D] (bf16/fp16/fp32, CUDA); self_lse [T,H] (fp32);
    remote_out [M,1,D] or [M,D]; remote_lse [M]; mapping [M] with p = s*H + h.
    builder: "torch" (deterministic, V2) or "atomic" (device-side, V3).
    check_capacity: raise on slot-capacity overflow (one device sync per call).
    Returns (O [T,H,D] same dtype as self_out, LSE [T,H] fp32).
    """
    if self_out.dim() != 3 or not self_out.is_cuda:
        raise ValueError("self_out must be CUDA [T,H,D]")
    T, H, D = self_out.shape
    if D & (D - 1):
        raise ValueError(f"head_dim {D} must be a power of 2 for the prototype kernel")
    P = T * H
    dev = self_out.device
    ro = remote_out.detach().reshape(-1, D).contiguous()
    rl = remote_lse.detach().reshape(-1).to(torch.float32).contiguous()
    mapping = mapping.to(device=dev, dtype=torch.int64).reshape(-1).contiguous()
    M = mapping.numel()
    if kmax is None:
        kmax = 1
        if M:
            kmax = max(1, int(torch.bincount(mapping, minlength=P).max().item()))
    kpad = 1 << max(0, (kmax - 1)).bit_length()

    with torch.cuda.device(dev):
        if builder == "torch":
            slots, counts = build_slots_torch(mapping, P, kmax, kpad=kpad)
        elif builder == "atomic":
            slots = torch.empty((P, kpad), dtype=torch.int32, device=dev)  # lanes beyond counts never read
            counts = torch.zeros(P, dtype=torch.int32, device=dev)
            if M:
                _build_slots_atomic_kernel[(triton.cdiv(M, 256),)](
                    mapping, slots, counts, M, KPAD=kpad, BLOCK=256
                )
            if check_capacity and M:
                if bool((counts > kpad).any()):
                    raise ValueError(f"capacity exceeded: a query row has more than {kpad} remote contributions")
        else:
            raise ValueError(f"unknown builder {builder!r}")

        so = self_out.reshape(P, D).contiguous()
        sl = self_lse.reshape(P).to(torch.float32).contiguous()
        out = torch.empty((P, D), device=dev, dtype=self_out.dtype)
        lse = torch.empty((P,), device=dev, dtype=torch.float32)
        _merge_kernel[(P,)](
            so, sl, ro, rl, slots, counts, out, lse,
            D=D, KPAD=kpad, num_warps=4,
        )
    return out.view(T, H, D), lse.view(T, H)
