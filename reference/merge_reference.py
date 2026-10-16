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
