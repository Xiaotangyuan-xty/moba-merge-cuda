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


def test_capture_after_live_steps_and_graph_capacity_boundary():
    c=PagedKVCache(1,4,1,8,page_size=1,chunk_size=2)
    q=torch.ones(1,1,8,device='cuda',dtype=c.dtype)
    with c.prepare_decode([0],1) as p:
        for value in (1,2,3):p.step(q,q,q*value)
        before=c.materialize(0)
        p.capture('step').capture('attend')
        assert c.lengths==(3,)
        assert torch.equal(before[0],c.materialize(0)[0]) and torch.equal(before[1],c.materialize(0)[1])
        result=p.step(q,q,q*4,use_graph=True)
        reference(c,result,q)
        with pytest.raises(ValueError,match='max_seq_len'):p.step(q,q,q,use_graph=True)
        assert c.lengths==(4,)


def test_graph_closes_large_chunk_at_maximum_head_dimension():
    c=PagedKVCache(1,1026,1,256,page_size=7,chunk_size=1024)
    k,v=rand((1023,1,256),c.dtype,801),rand((1023,1,256),c.dtype,802)
    c.append(k,v,sequence_ids=[0],cu_seqlens=[0,1023])
    q=rand((1,4,256),c.dtype,803);kn,vn=k[:1],v[:1]
    with c.prepare_decode([0],4) as p:
        p.capture('step')
        reference(c,p.step(q,kn,vn,use_graph=True),q)
        expected=torch.cat((k,kn)).float().mean(0).to(c.dtype)
        torch.testing.assert_close(c._means[0,0],expected,atol=1e-5,rtol=0.01)


def test_bound_input_buffers_support_graph_without_staging_copies():
    c=PagedKVCache(1,8,1,1,page_size=4,chunk_size=2)
    with c.prepare_decode([0],1,check_finite=False) as p:
        p.q.fill_(1);p.k.fill_(1);p.v.fill_(2)
        p.capture('step')
        for _ in range(3):
            result=p.step(p.q,p.k,p.v,use_graph=True)
            assert torch.equal(result.output,torch.full_like(result.output,2))
        assert p.delta_uploads==1
        with pytest.raises(ValueError,match='owns cache'):c.prepare_decode([0],1)
