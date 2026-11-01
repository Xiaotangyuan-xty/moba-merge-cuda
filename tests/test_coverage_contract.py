"""CPU-only support contract checks; no GPU work."""
import pytest
from moba_opt.baseline_adapter import validate_shapes, validate_config


@pytest.mark.parametrize("hq,hkv", [(1,1), (28,4), (32,8), (16,2), (8,1)])
def test_grouped_shapes(hq, hkv):
    assert validate_shapes((11,hq,128), (11,hkv,128), (11,hkv,128), True) == hq // hkv


@pytest.mark.parametrize("dim", range(8,257,8))
def test_head_dimensions(dim):
    assert validate_shapes((3,2,dim), (3,2,dim), (3,2,dim)) == 1


@pytest.mark.parametrize("qs,ks,vs", [
    ((0,2,64),(0,2,64),(0,2,64)), ((3,3,64),(3,2,64),(3,2,64)),
    ((3,2,7),(3,2,7),(3,2,7)), ((3,2,264),(3,2,264),(3,2,264)),
    ((3,2,64),(4,2,64),(4,2,64)), ((3,2,64),(3,1,64),(3,2,64)),
])
def test_invalid_shapes(qs, ks, vs):
    with pytest.raises(ValueError):
        validate_shapes(qs, ks, vs, True)


@pytest.mark.parametrize("override", [
    {"chunk_size":0}, {"chunk_size":True}, {"topk":0}, {"topk":10},
    {"topk":2.5}, {"causal":False}, {"dropout":0.1},
])
def test_invalid_configs(override):
    with pytest.raises(ValueError):
        validate_config({"chunk_size":32,"topk":4,**override})


def test_grouped_oracle_gradients_cpu():
    import torch
    from reference.grouped_reference import grouped_reference
    from test_reference import make_route, dense_causal_ref
    torch.manual_seed(51)
    q = torch.randn(9,4,8,dtype=torch.float64,requires_grad=True)
    k = torch.randn(9,2,8,dtype=torch.float64,requires_grad=True)
    v = torch.randn(9,2,8,dtype=torch.float64,requires_grad=True)
    route = make_route([5,4], 2, 4, lambda *a: True)
    cu = torch.tensor([0,5,9])
    out = grouped_reference(q,k,v,cu,route)
    expanded, _ = dense_causal_ref(q,k.repeat_interleave(2,1),v.repeat_interleave(2,1),cu.tolist(),8**-0.5)
    torch.testing.assert_close(out,expanded,atol=1e-12,rtol=1e-12)
    assert torch.autograd.gradcheck(lambda a,b,c: grouped_reference(a,b,c,cu,route),
                                   (q,k,v),fast_mode=True)


@pytest.mark.parametrize("kv_heads",[1,2])
def test_benchmark_generator_preserves_kv_heads(kv_heads):
    import torch
    from benchmarks.benchmark_attention import build_inputs
    cfg={"total_tokens":9,"heads":8,"kv_heads":kv_heads,"head_dim":80,
         "dtype":"bfloat16","seq_lens":[4,5]}
    q,k,v,cu,ml,g=build_inputs(cfg,torch.device("cpu"),seed=81)
    assert q.shape==g.shape==(9,8,80)
    assert k.shape==v.shape==(9,kv_heads,80)
    assert cu.tolist()==[0,4,9] and ml==5
