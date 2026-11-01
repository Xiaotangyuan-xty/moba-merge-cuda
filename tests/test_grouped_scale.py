"""Bounded long-prefill regressions, single approved GPU; no model weights."""
import pytest
import torch
from moba_opt.api import run_impl
from test_grouped_attention import inputs,compare

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("hq,hkv,dim,lens,dtype",[
    (28,4,128,(1537,511),torch.bfloat16),
    (16,2,128,(2048,),torch.float16),
    (16,1,80,(3073,1023),torch.float16),
    (32,8,128,(8192,),torch.bfloat16),
])
def test_long_prefill_against_repeat(hq,hkv,dim,lens,dtype):
    q,k,v,cu,ml,g=inputs(hq,hkv,dim,lens=lens,dtype=dtype,seed=77)
    cfg={"chunk_size":128 if sum(lens)<8192 else 512,"topk":4}
    a=run_impl("official_gqa_repeat",q,k,v,cu,ml,cfg)
    ga=torch.autograd.grad(a,(q,k,v),grad_outputs=g)
    b=run_impl("opt_gqa_cuda",q,k,v,cu,ml,cfg)
    gb=torch.autograd.grad(b,(q,k,v),grad_outputs=g)
    compare(b,a,dtype)
    for x,y in zip(gb,ga):compare(x,y,dtype)
