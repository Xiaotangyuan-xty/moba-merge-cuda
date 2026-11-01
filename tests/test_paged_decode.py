"""GPU-gated incremental/paged decode correctness, no model weights."""
import pytest
import torch

from moba_opt import ContiguousKVCache, PagedKVCache
from moba_opt.api import run_impl
from reference.decode_reference import decode_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def rand(shape, dtype, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)).to("cuda", dtype)


def check(cache, result, q):
    keys, values = [], []
    for sid in range(len(cache.lengths)):
        k, v = cache.materialize(sid)
        keys.append(k)
        values.append(v)
    expected, lse = decode_reference(q, keys, values, result.sequence_ids, result.positions,
                                      chunk_size=cache.chunk_size, selected_blocks=result.selected_blocks)
    out = result.output.cpu().double()
    assert torch.isfinite(out).all() and torch.isfinite(result.lse).all()
    atol, rtol = ((0.02, 0.02) if cache.dtype == torch.bfloat16 else (0.002, 0.002))
    torch.testing.assert_close(out, expected, atol=atol, rtol=rtol)
    rel = torch.linalg.vector_norm(out-expected) / torch.linalg.vector_norm(expected).clamp_min(1e-8)
    assert rel < (0.01 if cache.dtype == torch.bfloat16 else 0.002)
    torch.testing.assert_close(result.lse.cpu().double(), lse, atol=5e-4, rtol=2e-5)


@pytest.mark.parametrize("kind",[PagedKVCache,ContiguousKVCache])
@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16])
@pytest.mark.parametrize("hq,hk",[(1,1),(8,1),(8,2),(28,4),(32,8)])
def test_incremental_decode(kind,dtype,hq,hk):
    kwargs={"page_size":3} if kind is PagedKVCache else {}
    cache=kind(2,32,hk,64,chunk_size=4,dtype=dtype,**kwargs)
    k,v=rand((9,hk,64),dtype,101),rand((9,hk,64),dtype,102)
    cache.append(k,v,sequence_ids=[0,1],cu_seqlens=[0,4,9])
    for step in range(3):
        q=rand((2,hq,64),dtype,103+step)
        k=rand((2,hk,64),dtype,113+step)
        v=rand((2,hk,64),dtype,123+step)
        result=cache.append_and_attend(q,k,v,sequence_ids=[0,1],topk=2)
        check(cache,result,q)
        check(cache,cache.attend(q,sequence_ids=[0,1],mode="dense"),q)


@pytest.mark.parametrize("dim",range(8,257,8))
@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16])
def test_decode_head_dimensions(dim,dtype):
    cache=PagedKVCache(2,64,2,dim,page_size=7,chunk_size=12,dtype=dtype)
    cache.append(rand((68,2,dim),dtype,131),rand((68,2,dim),dtype,132),
                 sequence_ids=[0,1],cu_seqlens=[0,27,68])
    q=rand((3,4,dim),dtype,133)
    for mode in ("moba","dense"):
        result=cache.attend(q,sequence_ids=[0,1,1],positions=[26,40,5],topk=2,mode=mode)
        check(cache,result,q)


@pytest.mark.parametrize("page",[1,3,7,16,31,64])
def test_pages_and_moba_chunks_need_not_align(page):
    cache=PagedKVCache(1,97,2,80,page_size=page,chunk_size=24)
    cache.k_cache.fill_(float("nan"));cache.v_cache.fill_(float("nan"))
    cache.append(rand((79,2,80),cache.dtype,141),rand((79,2,80),cache.dtype,142),sequence_ids=[0],cu_seqlens=[0,79])
    q=rand((2,8,80),cache.dtype,143)
    for topk in (1,2,9):
        check(cache,cache.attend(q,sequence_ids=[0,0],positions=[0,78],topk=topk),q)


def test_recycled_pages_and_request_isolation():
    cache=PagedKVCache(3,12,1,16,page_size=3,num_pages=4,chunk_size=4)
    cache.append(rand((9,1,16),cache.dtype,151),rand((9,1,16),cache.dtype,152),
                 sequence_ids=[0,1],cu_seqlens=[0,6,9])
    q=rand((1,4,16),cache.dtype,153)
    before=cache.attend(q,sequence_ids=[1],mode="dense").output
    cache.reset([0])
    cache.append(rand((6,1,16),cache.dtype,154),rand((6,1,16),cache.dtype,155),sequence_ids=[2],cu_seqlens=[0,6])
    assert cache.page_table[2][:2]==(1,0)
    torch.testing.assert_close(cache.attend(q,sequence_ids=[1],mode="dense").output,before)
    check(cache,cache.attend(q,sequence_ids=[2],topk=2),q)
    with pytest.raises(ValueError,match="cached tokens"):cache.attend(q,sequence_ids=[0])


def test_chunked_append_is_causal():
    dtype=torch.bfloat16
    k,v=rand((19,2,32),dtype,161),rand((19,2,32),dtype,162)
    q=rand((7,8,32),dtype,163)
    cache=PagedKVCache(2,32,2,32,page_size=3,chunk_size=4)
    cache.append(torch.cat((k[:5],k[9:16])),torch.cat((v[:5],v[9:16])),
                 sequence_ids=[0,1],cu_seqlens=[0,5,12])
    result=cache.append_and_attend(q,torch.cat((k[5:9],k[16:])),torch.cat((v[5:9],v[16:])),
                                   sequence_ids=[0,1],cu_seqlens=[0,4,7],topk=2)
    assert result.positions==(5,6,7,8,7,8,9)
    check(cache,result,q)
    # Changing data after seq0 position 5 must not affect its query at 5.
    other=PagedKVCache(2,32,2,32,page_size=3,chunk_size=4)
    k2,v2=k.clone(),v.clone();k2[6:9]=100;v2[6:9]=-100
    other.append(k2,v2,sequence_ids=[0,1],cu_seqlens=[0,9,19])
    compare=other.attend(q[:1],sequence_ids=[0],positions=[5],topk=2)
    torch.testing.assert_close(compare.output,result.output[:1],atol=0.02,rtol=0.02)


def test_routing_is_per_query_head_and_uses_only_closed_history():
    cache=PagedKVCache(1,16,2,8,page_size=3,chunk_size=4)
    k=torch.zeros(13,2,8,device="cuda",dtype=cache.dtype)
    for block in range(4):
        k[block*4:min((block+1)*4,13),0,0]=block+1
        k[block*4:min((block+1)*4,13),1,0]=4-block
    cache.append(k,rand(k.shape,cache.dtype,171),sequence_ids=[0],cu_seqlens=[0,13])
    q=torch.zeros(2,4,8,device="cuda",dtype=cache.dtype);q[:,:,0]=1
    result=cache.attend(q,sequence_ids=[0,0],positions=[12,7],topk=3)
    route=result.selected_blocks.cpu()
    assert set(route[0,0].tolist())=={3,2,1}
    assert set(route[0,2].tolist())=={3,0,1}
    assert set(route[1,0].tolist())=={1,0,-1}
    check(cache,result,q)


def test_append_partition_invariance():
    k,v=rand((25,2,24),torch.bfloat16,181),rand((25,2,24),torch.bfloat16,182)
    a=PagedKVCache(1,32,2,24,page_size=7,chunk_size=6)
    b=PagedKVCache(1,32,2,24,page_size=7,chunk_size=6)
    a.append(k,v,sequence_ids=[0],cu_seqlens=[0,25])
    for lo,hi in [(0,1),(1,5),(5,13),(13,25)]:
        b.append(k[lo:hi],v[lo:hi],sequence_ids=[0],cu_seqlens=[0,hi-lo])
    torch.testing.assert_close(a._means[:,:4],b._means[:,:4],atol=0,rtol=0)
    q=rand((1,8,24),torch.bfloat16,183)
    x=a.attend(q,sequence_ids=[0],topk=3)
    y=b.attend(q,sequence_ids=[0],topk=3)
    torch.testing.assert_close(x.output,y.output,atol=0.02,rtol=0.02)


@pytest.mark.parametrize("hq,hk",[(8,1),(8,2),(28,4)])
def test_decode_matches_last_prefill_row(hq,hk):
    q,k,v=rand((97,hq,64),torch.bfloat16,191),rand((97,hk,64),torch.bfloat16,192),rand((97,hk,64),torch.bfloat16,193)
    cu=torch.tensor([0,97],device="cuda",dtype=torch.int32)
    ref=run_impl("opt_gqa_cuda",q,k,v,cu,97,{"chunk_size":16,"topk":3})[-1:]
    cache=PagedKVCache(1,128,hk,64,page_size=7,chunk_size=16)
    cache.append(k,v,sequence_ids=[0],cu_seqlens=[0,97])
    result=cache.attend(q[-1:],sequence_ids=[0],topk=3)
    torch.testing.assert_close(result.output,ref,atol=0.02,rtol=0.02)
    check(cache,result,q[-1:])


def test_empty_capacity_and_inference_errors_do_not_mutate_cache():
    cache=PagedKVCache(2,8,1,16,page_size=3,num_pages=1,chunk_size=4)
    q=rand((1,4,16),cache.dtype,201)
    with pytest.raises(ValueError,match="cached tokens"):cache.attend(q,sequence_ids=[0])
    k,v=rand((3,1,16),cache.dtype,202),rand((3,1,16),cache.dtype,203)
    cache.append(k,v,sequence_ids=[0],cu_seqlens=[0,3])
    before=cache.materialize(0)
    with pytest.raises(ValueError,match="pool exhausted"):
        cache.append(k[:1],v[:1],sequence_ids=[1])
    assert cache.lengths==(3,0)
    for actual,expected in zip(cache.materialize(0),before):torch.testing.assert_close(actual,expected)
    with pytest.raises(ValueError,match="inference-only"):
        cache.append_and_attend(q.requires_grad_(True),k[:1],v[:1],sequence_ids=[0])
    assert cache.lengths==(3,0)


def test_cross_stream_is_rejected_explicitly():
    cache=PagedKVCache(1,8,1,8,page_size=3,chunk_size=4)
    k,v=rand((3,1,8),cache.dtype,211),rand((3,1,8),cache.dtype,212)
    cache.append(k,v,sequence_ids=[0],cu_seqlens=[0,3])
    q=rand((1,1,8),cache.dtype,213)
    with torch.cuda.stream(torch.cuda.Stream()):
        with pytest.raises(ValueError,match="cross-stream"):cache.attend(q,sequence_ids=[0])
    check(cache,cache.attend(q,sequence_ids=[0]),q)


def test_long_context_single_token_decode():
    cache=PagedKVCache(1,8193,8,128,page_size=16,chunk_size=128)
    cache.append(rand((8192,8,128),cache.dtype,221),rand((8192,8,128),cache.dtype,222),
                 sequence_ids=[0],cu_seqlens=[0,8192])
    q,k,v=rand((1,32,128),cache.dtype,223),rand((1,8,128),cache.dtype,224),rand((1,8,128),cache.dtype,225)
    result=cache.append_and_attend(q,k,v,sequence_ids=[0],topk=4)
    assert cache.lengths==(8193,) and result.positions==(8192,)
    check(cache,result,q)


@pytest.mark.parametrize("chunk,dim,length",[(1,8,5),(1024,256,1025)])
def test_chunk_size_extremes(chunk,dim,length):
    cache=PagedKVCache(1,length+1,1,dim,page_size=3,chunk_size=chunk)
    cache.append(rand((length,1,dim),cache.dtype,231),rand((length,1,dim),cache.dtype,232),
                 sequence_ids=[0],cu_seqlens=[0,length])
    q=rand((1,4,dim),cache.dtype,233)
    check(cache,cache.attend(q,sequence_ids=[0],topk=2),q)


def test_extreme_logits_use_stable_online_softmax():
    cache=PagedKVCache(1,16,1,8,page_size=3,chunk_size=4)
    k=torch.full((13,1,8),1000.0,device="cuda",dtype=cache.dtype)
    k[::2]=-1000
    cache.append(k,rand((13,1,8),cache.dtype,241),sequence_ids=[0],cu_seqlens=[0,13])
    q=torch.full((1,2,8),1000.0,device="cuda",dtype=cache.dtype);q[:,1]=-1000
    check(cache,cache.attend(q,sequence_ids=[0],mode="dense"),q)
