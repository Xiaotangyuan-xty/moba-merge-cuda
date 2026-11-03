import pytest
import torch
pytest.importorskip('transformers')
from transformers import Qwen2Config,Qwen2ForCausalLM
from examples.full_qwen2 import FullQwen2Session

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def model():
    torch.manual_seed(911)
    cfg=Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                   num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
    cfg._attn_implementation='sdpa'
    return Qwen2ForCausalLM(cfg).to('cuda',torch.bfloat16).eval().requires_grad_(False)


@pytest.mark.parametrize('backend',['eager','attention_graph','model_graph'])
@pytest.mark.parametrize('packed',[False,True])
def test_full_dense_logits_and_cache_progress_match_hf(backend,packed):
    m=model();ids=torch.arange(1,13,device='cuda')[None]
    with torch.inference_mode():
        ref=m(ids,use_cache=True)
        with FullQwen2Session(m,24,backend=backend,fused_qkv=packed,chunk=4,
                             dense_backend='triton' if backend=='attention_graph' else 'fa2') as s:
            h=s.prefill_hidden(ids)
            torch.testing.assert_close(m.lm_head(h),ref.logits,atol=0.003,rtol=0.03)
            before=[c.materialize(0) for c in s.caches]
            s.prepare()
            for c,kv in zip(s.caches,before):
                assert c.lengths==(12,) and torch.equal(c.materialize(0)[0],kv[0])
            for token in range(13,18):
                ref=m(torch.tensor([[token]],device='cuda'),past_key_values=ref.past_key_values,use_cache=True)
                out=s.step(token)
                torch.testing.assert_close(out,ref.logits,atol=0.003,rtol=0.03)
            assert all(c.lengths==(17,) for c in s.caches)


def test_sparse_all_blocks_and_model_capacity_errors():
    m=model();ids=torch.arange(1,9,device='cuda')[None]
    with torch.inference_mode(),FullQwen2Session(m,10,mode='moba',chunk=4,topk=4,backend='model_graph') as s:
        h=s.prefill_hidden(ids)
        torch.testing.assert_close(m.lm_head(h),m(ids,use_cache=False).logits,atol=0.003,rtol=0.03)
        s.prepare();s.step(9);s.step(10)
        with pytest.raises(ValueError,match='max_seq_len'):s.step(11)
        assert all(c.lengths==(10,) for c in s.caches)
        with pytest.raises(ValueError,match='token'):s.step(-1)


@pytest.mark.parametrize('backend',['attention_graph','model_graph'])
@pytest.mark.parametrize('packed',[False,True])
def test_sparse_graph_matches_eager_across_page_and_chunk_boundaries(backend,packed):
    m=model();ids=torch.arange(1,12,device='cuda')[None]
    kwargs=dict(mode='moba',chunk=4,topk=2,page=4,fused_qkv=packed)
    with torch.inference_mode(),FullQwen2Session(m,24,backend='eager',**kwargs) as eager, \
            FullQwen2Session(m,24,backend=backend,**kwargs) as graph:
        assert not graph.effective_dense
        eager.prefill_hidden(ids)
        for source,target in zip(eager.caches,graph.caches):
            k,v=source.materialize(0)
            target.append(k,v,sequence_ids=[0],cu_seqlens=[0,11])
        before=[c.materialize(0) for c in graph.caches]
        eager.prepare();graph.prepare()
        for c,(k,v) in zip(graph.caches,before):
            assert c.lengths==(11,)
            actual=c.materialize(0)
            assert torch.equal(actual[0],k) and torch.equal(actual[1],v)
        for token in range(12,24):
            torch.testing.assert_close(eager.step(token),graph.step(token),atol=0,rtol=0)
            for a,b in zip(eager.plans,graph.plans):
                assert torch.equal(a.selected,b.selected)
                assert a.lengths==b.lengths==(token,)
                ak,av=a.cache.materialize(0);bk,bv=b.cache.materialize(0)
                assert torch.equal(ak,bk) and torch.equal(av,bv)


@pytest.mark.parametrize('dense_layers',[(),(0,),(1,)])
def test_stable_hybrid_full_graph_and_independent_prefill(dense_layers):
    m=model();ids=torch.arange(1,12,device='cuda')[None]
    kwargs=dict(mode='moba',chunk=4,topk=2,page=4,dense_layers=dense_layers,
                deterministic_merge=True,stable_routes=True,route_precision='fp32')
    with torch.inference_mode(),FullQwen2Session(m,24,backend='eager',**kwargs) as a, \
            FullQwen2Session(m,24,backend='model_graph',**kwargs) as b:
        assert torch.equal(a.prefill_hidden(ids),b.prefill_hidden(ids))
        a.prepare();b.prepare()
        for token in range(12,24):
            assert torch.equal(a.step(token),b.step(token))
            for i,(p,q) in enumerate(zip(a.plans,b.plans)):
                assert p.lengths==q.lengths
                if not a.layer_dense[i]:assert torch.equal(p.selected,q.selected)


@pytest.mark.parametrize('mode',['dense','moba'])
def test_chunked_prefill_mlp_matches_and_bounds_token_batch(mode):
    m=model();ids=torch.arange(1,18,device='cuda')[None]
    kwargs=dict(mode=mode,chunk=4,topk=2,**dict(deterministic_merge=True,stable_routes=True,route_precision='fp32'))
    with torch.inference_mode(),FullQwen2Session(m,24,**kwargs) as full, \
            FullQwen2Session(m,24,prefill_mlp_chunk=4,**kwargs) as chunked:
        a=full.prefill_hidden(ids);sizes=[]
        handles=[layer.mlp.register_forward_pre_hook(lambda mod,args:sizes.append(args[0].shape[1])) for layer in m.model.layers]
        try:b=chunked.prefill_hidden(ids)
        finally:
            for handle in handles:handle.remove()
        assert sizes and max(sizes)<=4
        torch.testing.assert_close(a,b,atol=0.003,rtol=0.03)
        for x,y in zip(full.caches,chunked.caches):
            assert x.lengths==y.lengths==(17,)
