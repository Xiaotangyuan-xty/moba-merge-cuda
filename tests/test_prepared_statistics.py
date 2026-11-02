import json
import pytest
from scripts.summarize_prepared import summarize


def records(names,values):
    return [dict(round=r,sample=0,impl=name,wall_ms=v,stream_ms=v,
                 order=list(names) if r%2==0 else list(names[::-1]))
            for r,vals in enumerate(values) for name,v in zip(names,vals)]


def data():
    result=[]
    for contract in ('strict','trusted'):
        for mode in ('eager','graph'):
            samples=records(('cab3_eager','prepared_'+mode),[(1,1),(9,3)])
            result.append(dict(config={'id':'A'},contract=contract,mode=mode,status='ok',
                working_tree_dirty=False,git_head='tested',seed=1,rounds=2,samples=1,
                attend=samples,step=samples,dense_reference=records(('dense_fa_core',),[(1,),(1,)]),
                metadata_delta_uploads=1,append_calls=2))
    return result


def test_contracts_are_not_pooled_and_round_ratios_are_used(tmp_path):
    path=tmp_path/'raw.jsonl';path.write_text('\n'.join(json.dumps(x) for x in data()))
    rows,summary=summarize(path,{'A':{'id':'A'}})
    assert len(rows)==4 and len(summary['groups'])==4
    assert all(r['attend_wall_ms_speedup']==2 for r in rows)


def test_missing_contract_mode_or_invalid_order_is_rejected(tmp_path):
    path=tmp_path/'raw.jsonl'
    raw=data();path.write_text('\n'.join(json.dumps(x) for x in raw[:-1]))
    with pytest.raises(ValueError,match='matrix'):summarize(path,{'A':{'id':'A'}})
    raw[0]['attend'][0]['order']=['prepared_eager','cab3_eager']
    path.write_text('\n'.join(json.dumps(x) for x in raw))
    with pytest.raises(ValueError,match='order'):summarize(path,{'A':{'id':'A'}})
