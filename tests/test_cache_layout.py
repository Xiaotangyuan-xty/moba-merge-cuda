"""CPU allocator/reference tests; safe with all CUDA devices hidden."""
import pytest
import torch
from moba_opt.cache_layout import PageAllocator
from moba_opt.kv_cache import ContiguousKVCache, PagedKVCache, _mode
from reference.decode_reference import decode_reference


def state(a):
    return a.tables, a.lengths, a.free_pages, a.version


def test_append_plans_do_not_publish_state():
    a = PageAllocator(2, 17, 3, 12, 4)
    before = state(a)
    plan = a.plan_append([1,0], [5,7])
    assert state(a) == before
    assert plan.token_sequences == (1,)*5 + (0,)*7
    assert plan.token_positions == tuple(range(5)) + tuple(range(7))
    assert plan.completed_chunks == ((1,0),(0,0))
    a.commit(plan)
    assert a.lengths == (7,5)
    assert a.tables[1][:2] == (0,1) and a.tables[0][:3] == (2,3,4)
    with pytest.raises(RuntimeError, match="stale"):
        a.commit(plan)


def test_page_reuse_can_be_physically_noncontiguous():
    a=PageAllocator(3,12,3,4,4)
    a.commit(a.plan_append([0,1],[6,3]))
    a.reset([0])
    a.commit(a.plan_append([2],[6]))
    assert a.tables[2][:2] == (1,0)
    assert a.tables[1][0] == 2
    assert len(a.free_pages)==1


def test_exhaustion_and_invalid_append_are_atomic():
    a=PageAllocator(2,12,3,2,4)
    a.commit(a.plan_append([0],[4]))
    before=state(a)
    for ids,counts in [([1],[1]),([0],[20]),([0,0],[1,1]),([2],[1]),([0],[0])]:
        with pytest.raises(ValueError):a.plan_append(ids,counts)
        assert state(a)==before


def test_closed_chunk_updates_depend_on_logical_position():
    a=PageAllocator(1,20,3,7,4)
    a.commit(a.plan_append([0],[3]))
    p=a.plan_append([0],[6])
    assert p.completed_chunks==((0,0),(0,1))
    a.commit(p)
    p=a.plan_append([0],[1])
    assert p.completed_chunks==()


@pytest.mark.parametrize("p",[1,3,7,16,31])
def test_page_math_for_nondivisible_sizes(p):
    a=PageAllocator(2,65,p,2*((65+p-1)//p),24)
    a.commit(a.plan_append([0,1],[65,33]))
    owned=[x for row in a.tables for x in row if x>=0]
    assert len(owned)==len(set(owned))
    assert len(owned)+len(a.free_pages)==a.num_pages
    a.reset([0,1])
    assert len(a.free_pages)==a.num_pages and a.lengths==(0,0)


@pytest.mark.parametrize("kwargs",[
    {"head_dim":0},{"head_dim":264},{"num_kv_heads":0},{"page_size":0},
    {"num_pages":0},{"chunk_size":1025},{"dtype":torch.float32},{"device":"cpu"},
])
def test_cache_invalid_contract_before_gpu_initialization(kwargs):
    cfg=dict(batch_size=1,max_seq_len=16,num_kv_heads=2,head_dim=64)
    cfg.update(kwargs)
    with pytest.raises(ValueError):PagedKVCache(**cfg)


@pytest.mark.parametrize("mode,topk",[("other",4),("dense",0),("moba",10),("moba",True)])
def test_decode_mode_contract(mode,topk):
    with pytest.raises(ValueError):_mode(mode,topk)


def test_fp64_dense_uniform_attention_manual():
    q=torch.zeros(2,4,8)
    k=[torch.zeros(5,2,8)]
    v=[torch.arange(5*2*8).reshape(5,2,8).float()]
    out,lse=decode_reference(q,k,v,[0,0],[1,4],chunk_size=2)
    for row,pos in enumerate([1,4]):
        for h in range(4):
            torch.testing.assert_close(out[row,h],v[0][:pos+1,h//2].double().mean(0))
        torch.testing.assert_close(lse[row],torch.full((4,),float(torch.tensor(pos+1).double().log()),dtype=torch.float64))


def test_fp64_rejects_future_and_duplicate_routes():
    q=torch.zeros(1,1,8); kv=[torch.ones(9,1,8)]
    for route in [torch.tensor([[[2,2]]]),torch.tensor([[[2,3]]])]:
        with pytest.raises(ValueError):
            decode_reference(q,kv,kv,[0],[8],chunk_size=4,selected_blocks=route)
