import copy
import json
from pathlib import Path
import pytest
from scripts.multilen_protocol import windows,select,aggregate
from scripts import run_shared_gpu_batch as guard

RECIPE=json.loads((Path(__file__).parents[1]/'configs/multilen_20261103.json').read_text())


def test_end_aligned_targets_and_unused_test_windows():
    for split in ('train','valid','test'):
        rows=windows(RECIPE,split)
        for end in RECIPE['target_boundaries'][split]:
            rr=[r for r in rows if r['end']==end]
            assert len({r['tail_start'] for r in rr})==1
            assert len({r['continuation_end'] for r in rr})==1
    assert min(w['start'] for w in windows(RECIPE,'test'))>=81920


def candidates():
    result=[]
    for count in range(7):
        rows=[]
        for w in windows(RECIPE,'valid'):
            delta=0 if count==6 else 0.01
            if w['length']==16384 and count<5:delta=0.1
            rows.append(dict(w,tail_delta_nll=delta,continuation_delta_nll=delta))
        result.append(dict(dense_layer_count=count,dense_layers=list(range(count)),rows=rows))
    return result


def test_shared_policy_must_pass_every_length_and_window():
    r=select(candidates(),RECIPE,6)
    assert r['shared_policy']['dense_layer_count']==5
    assert r['per_length_policies']['2048']['dense_layer_count']==0
    assert r['per_length_policies']['16384']['dense_layer_count']==5


@pytest.mark.parametrize('bad',['missing','test_data','nan'])
def test_bad_selection_matrix_is_rejected(bad):
    c=candidates()
    if bad=='missing':c[0]['rows'].pop()
    if bad=='test_data':c[0]['rows'][0]['split']='test'
    if bad=='nan':c[0]['rows'][0]['tail_delta_nll']=float('nan')
    with pytest.raises(ValueError):select(c,RECIPE,6)


def test_quality_rejects_missing_window():
    rows=[dict(w,method=m,tail_nll_sum=1,continuation_nll_sum=1)
          for m in ('dense','shared') for w in windows(RECIPE,'test')]
    assert len(aggregate(rows,RECIPE))==12
    rows.pop()
    with pytest.raises(ValueError,match='matrix'):aggregate(rows,RECIPE)


def test_gpu_guard_queries_only_the_configured_device(monkeypatch):
    monkeypatch.setattr(guard,'GPU_UUID','GPU-approved')
    calls=[]
    def query(cmd,**kwargs):
        calls.append(cmd)
        assert '--id=GPU-approved' in cmd
        return '1, GPU-approved, NVIDIA A40, 0, 46068\n' if any(x.startswith('--query-gpu') for x in cmd) else ''
    monkeypatch.setattr(guard.subprocess,'check_output',query)
    used,pids,_=guard.inspect()
    assert used==0 and not pids and len(calls)==2
