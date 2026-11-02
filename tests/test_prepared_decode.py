import pytest
import torch
from moba_opt import PagedKVCache
from reference.decode_reference import decode_reference

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def rand(shape,dtype,seed):
    return torch.randn(shape,generator=torch.Generator().manual_seed(seed)).to("cuda",dtype)


def reference(cache,result,q):
    kv=[cache.materialize(i) for i in range(len(cache.lengths))]
    out,lse=decode_reference(q,[x[0] for x in kv],[x[1] for x in kv],result.sequence_ids,
                             result.positions,chunk_size=cache.chunk_size,
                             selected_blocks=result.selected_blocks)
    tol=0.02 if q.dtype==torch.bfloat16 else 0.002
    torch.testing.assert_close(result.output.cpu().double(),out,atol=tol,rtol=tol)
    torch.testing.assert_close(result.lse.cpu().double(),lse,atol=5e-4,rtol=2e-5)


@pytest.mark.parametrize("graph",[False,True])
@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16])
@pytest.mark.parametrize("page,chunk,dim,hq,hk",[(3,4,33,15,3),(7,6,64,8,1),(16,8,128,8,2)])
def test_prepared_steps_match_fp64_and_original_payload(graph,dtype,page,chunk,dim,hq,hk):
    cache=PagedKVCache(3,40,hk,dim,page_size=page,chunk_size=chunk,dtype=dtype)
    k,v=rand((14,hk,dim),dtype,501),rand((14,hk,dim),dtype,502)
    cache.append(k,v,sequence_ids=[2,0],cu_seqlens=[0,5,14])
    stored_k={2:k[:5],0:k[5:]};stored_v={2:v[:5],0:v[5:]}
    lengths,table=cache.lengths,cache.page_table
    before=[cache.materialize(s) for s in (2,0)]
    with cache.prepare_decode([2,0],hq,topk=3) as plan:
        if graph:
            plan.capture("step").capture("attend")
            assert cache.lengths==lengths and cache.page_table==table
            for sid,(bk,bv) in zip((2,0),before):
                ak,av=cache.materialize(sid);assert torch.equal(ak,bk) and torch.equal(av,bv)
        pointers=[t.data_ptr() for t in (cache._table,cache._lengths,plan.output,plan.scores,plan.selected)]
        for step in range(10):
            q=rand((2,hq,dim),dtype,510+step)
            kn,vn=rand((2,hk,dim),dtype,530+step),rand((2,hk,dim),dtype,550+step)
            result=plan.step(q,kn,vn,use_graph=graph)
            for i,sid in enumerate((2,0)):
                stored_k[sid]=torch.cat((stored_k[sid],kn[i:i+1]))
                stored_v[sid]=torch.cat((stored_v[sid],vn[i:i+1]))
                actual=cache.materialize(sid)
                assert torch.equal(actual[0],stored_k[sid]) and torch.equal(actual[1],stored_v[sid])
            reference(cache,result,q)
            reference(cache,plan.attend(q,use_graph=graph),q)
        assert pointers==[t.data_ptr() for t in (cache._table,cache._lengths,plan.output,plan.scores,plan.selected)]
        assert plan.delta_uploads<=10
        with pytest.raises(ValueError,match="owns cache"):cache.reset()
    cache.reset([2])
    assert cache.lengths==(19,0,0)


@pytest.mark.parametrize("graph",[False,True])
@pytest.mark.parametrize("mode,topk",[("moba",1),("dense",4),("moba",9)])
def test_prepared_empty_and_dense_fallback(graph,mode,topk):
    cache=PagedKVCache(1,17,1,8,page_size=3,chunk_size=4)
    with cache.prepare_decode([0],4,topk=topk,mode=mode) as p:
        if graph:p.capture("step")
        for step in range(9):
            q=rand((1,4,8),cache.dtype,601+step)
            kv=rand((1,1,8),cache.dtype,621+step)
            reference(cache,p.step(q,kv,kv,use_graph=graph),q)


def test_validation_lifecycle_capacity_and_metadata_delta():
    c=PagedKVCache(1,7,1,8,page_size=3,chunk_size=4)
    q=torch.ones((1,1,8),device="cuda",dtype=c.dtype)
    with c.prepare_decode([0],1) as p:
        before=c.lengths
        with pytest.raises(ValueError,match="NaN/Inf"):p.step(q*float("nan"),q,q)
        with pytest.raises(ValueError,match="capture"):p.step(q,q,q,use_graph=True)
        assert c.lengths==before
        for _ in range(7):p.step(q,q,q)
        assert p.delta_uploads==3
        with pytest.raises(ValueError,match="max_seq_len"):p.step(q,q,q)
        assert c.lengths==(7,)
        with torch.cuda.stream(torch.cuda.Stream()):
            with pytest.raises(ValueError,match="cross-stream"):p.attend(q)
    with pytest.raises(ValueError,match="closed"):p.step(q,q,q)
    assert c.lengths==(7,)


def test_input_workspace_alias_and_output_reuse_contract():
    c=PagedKVCache(1,8,1,8,page_size=3,chunk_size=4)
    q=torch.ones((1,1,8),device="cuda",dtype=c.dtype)
    with c.prepare_decode([0],1,check_finite=False) as p:
        with pytest.raises(ValueError,match="aliases"):p.step(p.v,p.k,p.q)
        a=p.step(q,q,q);b=p.step(q,q,q)
        assert a.output.data_ptr()==b.output.data_ptr()
