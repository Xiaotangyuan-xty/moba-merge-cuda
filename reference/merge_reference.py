"""FP64 reference for the merge primitive (plan 4.3) and its VJP (plan 4.5).

Used by suboperator tests: synthetic local O/LSE branches in, merged O and
global LSE out, in float64 on CPU. Independent of the official merge code.
"""

import torch


def merge_reference(branches, compute_dtype=torch.float64):
    """Merge per-branch attention outputs into one softmax-normalized output.

    branches: list of (o_j [N, D], lse_j [N]); branches must cover disjoint KV
    sets and share the softmax scale.
    Returns: O [N, D], L [N]  (m = max_j l_j; w_j = exp(l_j - m);
                               O = sum_j w_j o_j / sum_j w_j; L = m + log sum_j w_j)
    """
    if not branches:
        raise ValueError("need at least one branch")
    os_ = [o.detach().to(dtype=compute_dtype) for o, _ in branches]
    ls = torch.stack([l.detach().to(dtype=compute_dtype) for _, l in branches])  # [J, N]
    m = ls.max(dim=0).values
    w = torch.exp(ls - m)
    z = w.sum(dim=0)
    o_sum = sum(w[j].unsqueeze(-1) * os_[j] for j in range(len(branches)))
    return o_sum / z.unsqueeze(-1), m + torch.log(z)


def merge_vjp_reference(branches, merged_o, merged_lse, g, compute_dtype=torch.float64):
    """VJP of the merge primitive (plan 4.5).

    alpha_j = exp(l_j - L);  d o_j = alpha_j * g;  d l_j = alpha_j * dot(g, o_j - O)
    g: upstream gradient [N, D]. Returns lists d_os [N, D], d_ls [N].
    """
    os_ = [o.detach().to(dtype=compute_dtype) for o, _ in branches]
    ls = torch.stack([l.detach().to(dtype=compute_dtype) for _, l in branches])
    g = g.detach().to(dtype=compute_dtype)
    merged_o = merged_o.detach().to(dtype=compute_dtype)
    merged_lse = merged_lse.detach().to(dtype=compute_dtype)
    alpha = torch.exp(ls - merged_lse.unsqueeze(0))  # [J, N]
    d_os = [alpha[j].unsqueeze(-1) * g for j in range(len(branches))]
    d_ls = [alpha[j] * (g * (os_[j] - merged_o)).sum(dim=-1) for j in range(len(branches))]
    return d_os, d_ls


def merge_grouped_reference(self_out, self_lse, remote_out, remote_lse, mapping, compute_dtype=torch.float64):
    """Row-wise merge_reference over groups p = 0..P-1 (p = s*H + h).

    self_out [P,D], self_lse [P]; remote_out [M,D], remote_lse [M];
    mapping [M] (remote row -> p). Returns O [P,D], L [P] in compute_dtype on CPU.
    """
    so = self_out.detach().to(device="cpu", dtype=compute_dtype).reshape(self_out.shape[0], -1)
    sl = self_lse.detach().to(device="cpu", dtype=compute_dtype).reshape(-1)
    ro = remote_out.detach().to(device="cpu", dtype=compute_dtype).reshape(-1, so.shape[1])
    rl = remote_lse.detach().to(device="cpu", dtype=compute_dtype).reshape(-1)
    mapping = mapping.detach().to(device="cpu", dtype=torch.long).reshape(-1)
    P, D = so.shape
    groups = {}
    for m, p in enumerate(mapping.tolist()):
        groups.setdefault(p, []).append(m)
    O = torch.empty(P, D, dtype=compute_dtype)
    L = torch.empty(P, dtype=compute_dtype)
    for p in range(P):
        branches = [(so[p].unsqueeze(0), sl[p].reshape(1))]
        branches += [(ro[m].unsqueeze(0), rl[m].reshape(1)) for m in groups.get(p, [])]
        o_row, l_row = merge_reference(branches, compute_dtype=compute_dtype)
        O[p] = o_row[0]
        L[p] = l_row[0]
    return O, L
