"""Triton prototype for route A (plan 4.2/4.4): global-softmax merge.

V2 scope (plan 9.3): the inverse mapping (slots) is prebuilt with deterministic
torch ops; the Triton kernel only performs the stable merge. A device-side
atomic slot build (with its real cost) belongs to V3.

Math (plan 4.3): m = max_j l_j; w_j = exp(l_j - m); O = sum_j w_j o_j / z;
z = sum_j w_j; L = m + log(z). Compute in fp32; O cast to input dtype,
global LSE fp32. Invalid slots are explicitly masked (weight 0, no NaN).
"""

import torch
import triton
import triton.language as tl

INVALID = -1


def build_slots_torch(mapping, P, kmax, kpad=None):
    """Deterministic inverse mapping.

    slots[p, j] = remote row index of the j-th remote contribution to global
    row p (j ordered by ascending remote row index, independent of the input
    mapping order), or INVALID when absent. counts[p] = number of remote
    contributions. Raises on capacity overflow (plan 7.2: no silent drop).

    Row stride of slots is kpad (>= kmax) when given, so kernels indexing with
    a padded stride never read across rows.
    """
    dev = mapping.device
    M = mapping.numel()
    stride = kpad or kmax
    slots = torch.full((P, stride), INVALID, dtype=torch.int32, device=dev)
    counts = torch.zeros(P, dtype=torch.int64, device=dev)
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
    if bool((ranks >= kmax).any()):
        raise ValueError(f"capacity exceeded: a query has more than kmax={kmax} remote contributions")
    slots[unique_p[gids], ranks] = order.to(torch.int32)
    counts[unique_p] = cnt
    return slots, counts


@triton.jit
def _merge_kernel(
    self_out_ptr,
    self_lse_ptr,
    remote_out_ptr,
    remote_lse_ptr,
    slots_ptr,
    out_ptr,
    lse_ptr,
    D: tl.constexpr,
    KPAD: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, D)
    offs_k = tl.arange(0, KPAD)

    l_self = tl.load(self_lse_ptr + pid)
    slot = tl.load(slots_ptr + pid * KPAD + offs_k)
    valid = slot >= 0
    l_rem = tl.load(remote_lse_ptr + tl.where(valid, slot, 0), mask=valid, other=float("-inf"))

    m = tl.maximum(l_self, tl.max(l_rem, axis=0))
    w_self = tl.exp(l_self - m)
    w_rem = tl.where(valid, tl.exp(l_rem - m), 0.0)
    z = w_self + tl.sum(w_rem, axis=0)

    o_self = tl.load(self_out_ptr + pid * D + offs_d).to(tl.float32)
    r = tl.load(
        remote_out_ptr + tl.where(valid, slot, 0)[:, None] * D + offs_d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    o = (w_self * o_self + tl.sum(w_rem[:, None] * r, axis=0)) / z

    tl.store(out_ptr + pid * D + offs_d, o.to(out_ptr.dtype.element_ty))
    tl.store(lse_ptr + pid, m + tl.log(z))


def merge_forward(self_out, self_lse, remote_out, remote_lse, mapping, kmax=None):
    """Merge branch outputs into global O and LSE (prototype, V2).

    self_out [T,H,D] (bf16/fp16/fp32, CUDA); self_lse [T,H] (fp32);
    remote_out [M,1,D] or [M,D]; remote_lse [M]; mapping [M] with p = s*H + h.
    Returns (O [T,H,D] same dtype as self_out, LSE [T,H] fp32).
    """
    if self_out.dim() != 3 or not self_out.is_cuda:
        raise ValueError("self_out must be CUDA [T,H,D]")
    T, H, D = self_out.shape
    if D & (D - 1):
        raise ValueError(f"head_dim {D} must be a power of 2 for the prototype kernel")
    P = T * H
    ro = remote_out.detach().reshape(-1, D).contiguous()
    rl = remote_lse.detach().reshape(-1).to(torch.float32).contiguous()
    mapping = mapping.to(device=self_out.device, dtype=torch.int64).reshape(-1).contiguous()
    if kmax is None:
        kmax = 1
        if mapping.numel():
            kmax = max(1, int(torch.bincount(mapping, minlength=P).max().item()))
    kpad = 1 << max(0, (kmax - 1)).bit_length()
    slots, _ = build_slots_torch(mapping, P, kmax, kpad=kpad)

    so = self_out.reshape(P, D).contiguous()
    sl = self_lse.reshape(P).to(torch.float32).contiguous()
    out = torch.empty((P, D), device=self_out.device, dtype=self_out.dtype)
    lse = torch.empty((P,), device=self_out.device, dtype=torch.float32)
    _merge_kernel[(P,)](
        so, sl, ro, rl, slots, out, lse,
        D=D, KPAD=kpad, num_warps=4,
    )
    return out.view(T, H, D), lse.view(T, H)
