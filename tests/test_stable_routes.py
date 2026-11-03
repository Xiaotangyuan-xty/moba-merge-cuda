import pytest
import torch
from moba_opt.ops.merge_cuda import merge_forward_cuda
from moba_opt.ops.prepared_decode import stable_topk
from moba_opt.grouped_attention import opt_gqa_cuda

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('dtype',[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize('kmax',[1,3,8])
def test_canonical_merge_repeats_exactly_and_matches_fp64(dtype,kmax):
    torch.manual_seed(361)
    rows=513;dim=64
    counts=torch.arange(rows)% (kmax+1)
    mapping=torch.repeat_interleave(torch.arange(rows),counts)
    mapping=mapping[torch.randperm(mapping.numel())].cuda()
    s=torch.randn(rows,1,dim,device='cuda',dtype=dtype)
    sl=torch.randn(rows,1,device='cuda',dtype=torch.float32)*10
    r=torch.randn(mapping.numel(),dim,device='cuda',dtype=dtype)
    rl=torch.randn(mapping.numel(),device='cuda')*10
    a,l=merge_forward_cuda(s,sl,r,rl,mapping,kmax=kmax,deterministic=True)
    for _ in range(20):
        b,m=merge_forward_cuda(s,sl,r,rl,mapping,kmax=kmax,deterministic=True)
        assert torch.equal(a,b) and torch.equal(l,m)
    sd,rd=s[:,0].double(),r.double()
    expected=[];expected_lse=[]
    for row in range(rows):
        mask=mapping==row
        logs=torch.cat((sl[row].double(),rl[mask].double()))
        values=torch.cat((sd[row:row+1],rd[mask]))
        expected.append((torch.softmax(logs,0)[:,None]*values).sum(0))
        expected_lse.append(torch.logsumexp(logs,0))
    expected=torch.stack(expected).to(dtype)
    torch.testing.assert_close(a[:,0],expected,atol=2e-6 if dtype==torch.float32 else 0.01,rtol=1e-5 if dtype==torch.float32 else 0.01)
    torch.testing.assert_close(l[:,0].double(),torch.stack(expected_lse),atol=3e-6,rtol=1e-6)


def test_stable_topk_exact_ties_and_masked_rows_capture():
    scores=torch.tensor([[2.,2.,1.,2.,-float('inf')],[-float('inf')]*5],device='cuda')[None]
    values=torch.empty((1,2,4),device='cuda');indices=torch.empty((1,2,4),device='cuda',dtype=torch.long)
    stable_topk(scores,values,indices)
    expected=scores.argsort(dim=-1,descending=True,stable=True)[...,:4]
    assert torch.equal(indices,expected)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):stable_topk(scores,values,indices)
    graph.replay();torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(indices,expected)


def test_stable_gqa_repeats_exactly():
    torch.manual_seed(362)
    q=torch.randn(257,4,64,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(257,2,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    cu=torch.tensor([0,129,257],device='cuda',dtype=torch.int32)
    cfg=dict(chunk_size=16,topk=4,deterministic_merge=True,stable_routes=True,route_precision='fp32')
    a=opt_gqa_cuda(q,k,v,cu,129,cfg)
    for _ in range(8):assert torch.equal(a,opt_gqa_cuda(q,k,v,cu,129,cfg))


def test_sorting_merge_preserves_original_discrete_route():
    import moba_opt.grouped_attention as grouped
    torch.manual_seed(363)
    q=torch.randn(129,4,64,device='cuda',dtype=torch.bfloat16)
    k=torch.randn(129,2,64,device='cuda',dtype=torch.bfloat16);v=torch.randn_like(k)
    cu=torch.tensor([0,129],device='cuda',dtype=torch.int32);selected=[]
    grouped.ROUTE_HOOK=lambda **kw:selected.append(kw['selected'].clone())
    try:
        for flag in (False,True):opt_gqa_cuda(q,k,v,cu,129,dict(chunk_size=16,topk=4,deterministic_merge=flag))
    finally:grouped.ROUTE_HOOK=None
    assert len(selected)==2 and torch.equal(*selected)
