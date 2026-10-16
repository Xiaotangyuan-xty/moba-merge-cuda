"""Device-binding regression tests (review P1): kernels and harness paths must
run on the INPUT's device, not whatever device happens to be current.

Requires two visible GPUs (run e.g. with CUDA_VISIBLE_DEVICES=0,1 after
approval for both cards); skipped otherwise. GPU-gated.
"""

import pytest
import torch
from test_merge_triton import synth

from moba_opt.ops.merge_cuda import merge_forward_cuda
from moba_opt.ops.merge_triton import merge_forward as merge_forward_triton
from reference.merge_reference import merge_grouped_reference

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs >=2 visible devices"),
]


def _inputs_on(dev):
    counts = [0, 1, 2, 3, 1, 0, 2, 4]
    so, sl, ro, rl, mp = synth(counts, D=64, seed=21)
    return tuple(t.to(dev) for t in (so, sl, ro, rl, mp))


def test_cuda_ext_runs_on_input_device():
    # inputs live on cuda:1 while the current device stays cuda:0
    torch.cuda.set_device(0)
    so, sl, ro, rl, mp = _inputs_on(torch.device("cuda:1"))
    assert torch.cuda.current_device() == 0
    out, lse = merge_forward_cuda(
        so.view(8, 1, 64), sl.view(8, 1), ro.view(-1, 1, 64), rl, mp, kmax=8
    )
    assert out.device.index == 1 and lse.device.index == 1
    O, L = merge_grouped_reference(so, sl, ro, rl, mp)
    assert torch.allclose(out.view(8, 64).cpu().double(), O, atol=1e-5, rtol=1e-4)
    assert torch.allclose(lse.view(8).cpu().double(), L, atol=1e-5, rtol=1e-4)
    assert torch.cuda.current_device() == 0, "guard must restore the previous device"


def test_triton_runs_on_input_device():
    torch.cuda.set_device(0)
    so, sl, ro, rl, mp = _inputs_on(torch.device("cuda:1"))
    out, lse = merge_forward_triton(
        so.view(8, 1, 64), sl.view(8, 1), ro.view(-1, 1, 64), rl, mp, kmax=8, builder="atomic"
    )
    assert out.device.index == 1 and lse.device.index == 1
    O, L = merge_grouped_reference(so, sl, ro, rl, mp)
    assert torch.allclose(out.view(8, 64).cpu().double(), O, atol=1e-5, rtol=1e-4)
    assert torch.allclose(lse.view(8).cpu().double(), L, atol=1e-5, rtol=1e-4)
