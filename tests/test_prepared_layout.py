import pytest
from moba_opt.cache_layout import PageAllocator,IncrementalLayout


@pytest.mark.parametrize("page",[1,3,7,16])
def test_incremental_matches_transactional_allocator(page):
    old=PageAllocator(3,41,page,3*((41+page-1)//page),8)
    old.commit(old.plan_append([1,0],[5,7]));old.reset([1])
    fast=IncrementalLayout(old,[2,0,1])
    for _ in range(30):
        plan=fast.plan()
        before=(tuple(map(tuple,fast.tables)),tuple(fast.lengths),tuple(fast.free))
        assert fast.plan()==plan
        assert before==(tuple(map(tuple,fast.tables)),tuple(fast.lengths),tuple(fast.free))
        fast.commit(plan);old.commit(old.plan_append([2,0,1],[1,1,1]))
        assert tuple(map(tuple,fast.tables))==old.tables
        assert tuple(fast.lengths)==old.lengths and tuple(fast.free)==old.free_pages
    fast.publish(old)
    assert old.lengths==(37,30,30)
    with pytest.raises(RuntimeError,match="stale"):fast.commit(plan)


def test_incremental_pool_and_length_failure_do_not_mutate():
    old=PageAllocator(2,4,2,1,2)
    old.commit(old.plan_append([0],[2]))
    fast=IncrementalLayout(old,[0,1])
    with pytest.raises(ValueError,match="pool exhausted"):fast.plan()
    assert fast.lengths==[2,0] and fast.free==[]
    old=PageAllocator(1,1,1,1,1);old.commit(old.plan_append([0],[1]))
    with pytest.raises(ValueError,match="max_seq_len"):IncrementalLayout(old,[0]).plan()
