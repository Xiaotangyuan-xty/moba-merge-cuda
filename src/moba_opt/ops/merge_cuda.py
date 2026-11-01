"""Loader + Python wrapper for the CUDA merge extension (plan 4.7).

JIT-compiled with the system nvcc (12.9) for the local GPU arch; build cache
lives in TORCH_EXTENSIONS_DIR (scripts/env.sh points it at the data disk).
No fast-math: numerics must match the Triton reference path.
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.6")
        here = Path(__file__).resolve().parent
        _EXT = load(
            name="moba_opt_merge_cuda",
            sources=[str(here / "bindings.cpp"), str(here / "merge.cu")],
            extra_cuda_cflags=["-O3", "-std=c++17"],
            extra_cflags=["-O3", "-std=c++17"],
            verbose=False,
        )
    return _EXT


def merge_forward_cuda(self_out, self_lse, remote_out, remote_lse, mapping, kmax=None, check_capacity=True):
    """Same contract as merge_triton.merge_forward (atomic build included).

    self_out [T,H,D] (fp32/bf16/fp16, CUDA); self_lse [T,H] fp32;
    remote_out [M,1,D] or [M,D]; remote_lse [M] fp32; mapping [M] int64.
    check_capacity=True makes the slot build raise on capacity overflow
    (one small device->host sync per call). Returns (O, LSE fp32).
    """
    if mapping.dim() != 1 or mapping.dtype not in (torch.int32, torch.int64):
        raise ValueError("mapping must be a 1D int32/int64 tensor")
    if kmax is None:
        P = self_out.shape[0] * self_out.shape[1]
        mapping_n = mapping.reshape(-1)
        kmax = 1
        if mapping_n.numel():
            kmax = max(1, int(torch.bincount(mapping_n, minlength=P).max().item()))
    out, lse = ext().merge_forward(
        self_out,
        self_lse.to(torch.float32),
        remote_out,
        remote_lse.to(torch.float32),
        mapping.to(torch.int64).reshape(-1),
        int(kmax),
        bool(check_capacity),
    )
    return out, lse
