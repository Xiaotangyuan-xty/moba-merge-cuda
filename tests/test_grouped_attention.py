"""Small single-card GQA/MQA matrix. Run only through an approved GPU batch."""
import itertools
import pytest
import torch
import moba_opt
import moba_opt.grouped_attention as grouped
import moba_opt.diagnostic as diag
from moba_opt.api import run_impl

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def inputs(hq,hkv,dim,lens=(35,62),dtype=torch.bfloat16,seed=61):
    rng = torch.Generator().manual_seed(seed)
    total = sum(lens)
    q,k,v = [torch.randn(total,h,dim,generator=rng).to("cuda",dtype).requires_grad_(True)
             for h in (hq,hkv,hkv)]
    cu = torch.tensor([0]+list(itertools.accumulate(lens)),device="cuda",dtype=torch.int32)
    g = torch.randn(total,hq,dim,generator=rng).to("cuda",dtype)
    return q,k,v,cu,max(lens),g


def compare(a,b,dtype):
    af,bf = a.float(),b.float()
    delta = (af-bf).abs()
    rel_l2 = torch.linalg.vector_norm(af-bf) / torch.linalg.vector_norm(bf).clamp_min(1e-8)
    assert torch.isfinite(af).all()
    if dtype == torch.bfloat16:
        torch.testing.assert_close(af,bf,atol=0.04,rtol=0.03)
        assert rel_l2 < 0.01
    else:
        torch.testing.assert_close(af,bf,atol=0.003,rtol=0.003)
        assert rel_l2 < 0.002


@pytest.mark.parametrize("hq,hkv",[(1,1),(8,1),(8,2),(16,2),(28,4),(32,8)])
@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16])
def test_native_gqa_output_and_vjp(hq,hkv,dtype):
    q,k,v,cu,ml,g = inputs(hq,hkv,128,dtype=dtype)
    cfg={"chunk_size":16,"topk":4}
    outputs=[]; gradients=[]
    for impl in ("official_gqa_repeat","opt_gqa_cuda"):
        out=run_impl(impl,q,k,v,cu,ml,cfg)
        gradients.append(torch.autograd.grad(out,(q,k,v),grad_outputs=g))
        outputs.append(out)
    compare(outputs[1],outputs[0],dtype)
    for a,b in zip(gradients[1],gradients[0]): compare(a,b,dtype)


@pytest.mark.parametrize("dim",range(8,257,8))
@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16])
def test_gqa_head_dimension_matrix(dim,dtype):
    q,k,v,cu,ml,g=inputs(4,2,dim,lens=(17,32),dtype=dtype)
    cfg={"chunk_size":16,"topk":3}
    a=run_impl("official_gqa_repeat",q,k,v,cu,ml,cfg)
    ga=torch.autograd.grad(a,(q,k,v),grad_outputs=g)
    b=run_impl("opt_gqa_cuda",q,k,v,cu,ml,cfg)
    gb=torch.autograd.grad(b,(q,k,v),grad_outputs=g)
    compare(b,a,q.dtype)
    for x,y in zip(gb,ga): compare(x,y,q.dtype)


def test_native_routing_equals_repeat_reference():
    q,k,v,cu,ml,_=inputs(28,4,64,lens=(65,96))
    captured={}
    grouped.ROUTE_HOOK=lambda **x: captured.update(native=x["selected"].clone())
    diag.ROUTE_HOOK=lambda **x: captured.update(reference=x["selected"].clone())
    try:
        run_impl("opt_gqa_cuda",q,k,v,cu,ml,{"chunk_size":16,"topk":4})
        diag.moba_attn_varlen(q,k.repeat_interleave(7,1),v.repeat_interleave(7,1),cu,ml,16,4)
    finally:
        grouped.ROUTE_HOOK=diag.ROUTE_HOOK=None
    assert torch.equal(captured["native"],captured["reference"])


def test_native_fp64_output_and_independent_gradients():
    from reference.grouped_reference import grouped_reference
    q,k,v,cu,ml,g=inputs(4,2,16,lens=(9,12),seed=62)
    route={}
    grouped.ROUTE_HOOK=lambda **kw: route.update({k:t.detach().cpu() for k,t in kw.items()})
    try:
        out=run_impl("opt_gqa_cuda",q,k,v,cu,ml,{"chunk_size":4,"topk":3})
    finally:
        grouped.ROUTE_HOOK=None
    grads=torch.autograd.grad(out,(q,k,v),grad_outputs=g)
    route["cu_seqlens"]=cu.cpu()
    qc,kc,vc=[x.detach().cpu().double().requires_grad_(True) for x in (q,k,v)]
    ref=grouped_reference(qc,kc,vc,cu.cpu(),route)
    rg=torch.autograd.grad(ref,(qc,kc,vc),grad_outputs=g.cpu().double())
    compare(out.cpu(),ref,q.dtype)
    for x,y in zip(grads,rg):compare(x.cpu(),y,q.dtype)


@pytest.mark.parametrize("topk,lens",[(1,(35,62)),(4,(1,3)),(9,(9,12))])
def test_fallback_and_fewer_candidates(topk,lens):
    q,k,v,cu,ml,g=inputs(8,1,64,lens=lens)
    cfg={"chunk_size":16,"topk":topk}
    a=run_impl("official_gqa_repeat",q,k,v,cu,ml,cfg)
    b=run_impl("opt_gqa_cuda",q,k,v,cu,ml,cfg)
    compare(b,a,q.dtype)
    ga=torch.autograd.grad(a,(q,k,v),grad_outputs=g)
    gb=torch.autograd.grad(b,(q,k,v),grad_outputs=g)
    for x,y in zip(gb,ga):compare(x,y,q.dtype)


def test_causal_and_sequence_isolation_native():
    q,k,v,cu,ml,_=inputs(8,2,64,lens=(65,64))
    cfg={"chunk_size":16,"topk":3}
    out=run_impl("opt_gqa_cuda",q,k,v,cu,ml,cfg)
    k2,v2=k.detach().clone(),v.detach().clone()
    k2[32:]=0;v2[32:]=0
    changed=run_impl("opt_gqa_cuda",q,k2,v2,cu,ml,cfg)
    compare(out[:32],changed[:32],q.dtype)
    k2,v2=k.detach().clone(),v.detach().clone()
    k2[65:]=0;v2[65:]=0
    changed=run_impl("opt_gqa_cuda",q,k2,v2,cu,ml,cfg)
    compare(out[:65],changed[:65],q.dtype)
