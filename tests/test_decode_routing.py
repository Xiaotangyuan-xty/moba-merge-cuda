"""Regression against the frozen cache, plus validation/metadata contracts."""
from unittest.mock import patch

import pytest
import torch

from benchmarks._decode_v1 import PagedKVCache as FrozenCache
from moba_opt import PagedKVCache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("seed",[301,302,303])
@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float16])
@pytest.mark.parametrize("hq,hk",[(15,3),(8,1)])
def test_request_local_routing_matches_frozen_global_routing(seed,dtype,hq,hk):
    gen=torch.Generator().manual_seed(seed)
    def rand(shape):return torch.randn(shape,generator=gen).to("cuda",dtype)
    old=FrozenCache(4,80,hk,33,page_size=7,chunk_size=8,dtype=dtype)
    new=PagedKVCache(4,80,hk,33,page_size=7,chunk_size=8,dtype=dtype)
    k,v=rand((91,hk,33)),rand((91,hk,33))
    for cache in (old,new):
        cache.append(k,v,sequence_ids=[0,1,2,3],cu_seqlens=[0,1,9,26,91])
    q=rand((7,hq,33))
    kwargs=dict(sequence_ids=[3,0,2,1,3,2,3],positions=[64,0,16,7,0,5,23],topk=9)
    a,b=old.attend(q,**kwargs),new.attend(q,**kwargs)
    assert torch.equal(torch.sort(a.selected_blocks,-1).values,
                       torch.sort(b.selected_blocks,-1).values)
    torch.testing.assert_close(a.output,b.output,atol=0.02,rtol=0.02)
    torch.testing.assert_close(a.lse,b.lse,atol=5e-4,rtol=2e-5)


def test_no_history_preserves_padding_without_global_candidate_tensor():
    cache=PagedKVCache(2,65,1,8,page_size=7,chunk_size=8)
    data=torch.ones(65,1,8,device="cuda",dtype=cache.dtype)
    cache.append(data,data,sequence_ids=[1],cu_seqlens=[0,65])
    q=torch.ones(1,4,8,device="cuda",dtype=cache.dtype)
    result=cache.attend(q,sequence_ids=[1],positions=[0],topk=9)
    assert result.selected_blocks.shape==(1,4,9)
    assert (result.selected_blocks[:,:,0]==0).all()
    assert (result.selected_blocks[:,:,1:]==-1).all()
    assert not hasattr(cache,"_candidate_ids")
    cache.reset()
    assert cache._candidate_count==0


def test_combined_api_checks_each_tensor_once():
    cache=PagedKVCache(1,8,1,8,page_size=3,chunk_size=4)
    q=torch.ones(1,4,8,device="cuda",dtype=cache.dtype)
    kv=torch.ones(1,1,8,device="cuda",dtype=cache.dtype)
    with patch.object(cache,"_tensor",wraps=cache._tensor) as checked:
        cache.append_and_attend(q,kv,kv,sequence_ids=[0])
    assert [c.args[1] for c in checked.call_args_list]==["k","v","q"]


def test_metadata_is_packed_and_combined_api_reuses_gpu_query_indices():
    cache=PagedKVCache(2,16,1,8,page_size=3,chunk_size=4)
    q=torch.ones(3,4,8,device="cuda",dtype=cache.dtype)
    kv=torch.ones(3,1,8,device="cuda",dtype=cache.dtype)
    with patch.object(cache,"_gpu",wraps=cache._gpu) as copies:
        result=cache.append_and_attend(q,kv,kv,sequence_ids=[1,0],cu_seqlens=[0,2,3])
    assert copies.call_count==1
    assert result.sequence_ids==(1,1,0) and result.positions==(0,1,0)
    with patch.object(cache,"_gpu",wraps=cache._gpu) as copies:
        again=cache.attend(q,sequence_ids=[1,1,0],positions=[0,1,0])
    assert copies.call_count==1
    torch.testing.assert_close(result.output,again.output)
    assert cache._table.untyped_storage().data_ptr()==cache._lengths.untyped_storage().data_ptr()


@pytest.mark.parametrize("bad_name",["q","k","v"])
def test_combined_api_still_rejects_nonfinite_before_append(bad_name):
    cache=PagedKVCache(1,8,1,8,page_size=3,chunk_size=4)
    tensors={"q":torch.ones(1,4,8,device="cuda",dtype=cache.dtype),
             "k":torch.ones(1,1,8,device="cuda",dtype=cache.dtype),
             "v":torch.ones(1,1,8,device="cuda",dtype=cache.dtype)}
    tensors[bad_name][0,0,0]=float("nan")
    with pytest.raises(ValueError,match="NaN/Inf"):
        cache.append_and_attend(**tensors,sequence_ids=[0])
    assert cache.lengths==(0,)
