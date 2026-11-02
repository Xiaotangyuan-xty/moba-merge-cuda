import math
import pytest
from scripts.summarize_full_model import quality_summary,METHODS


def sample():
    rows=[];continuation=[]
    for m in METHODS:
        rows.extend([dict(method=m,offset=0,tokens=1,nll_sum=1),dict(method=m,offset=4,tokens=3,nll_sum=9)])
        continuation.extend([dict(method=m,block=0,tokens=2,nll_sum=2),dict(method=m,block=1,tokens=2,nll_sum=6)])
    return dict(status='ok',dirty=False,result=dict(blocks=2,block_length=4,rows=rows,continuation=continuation))


def test_quality_uses_token_weighted_loss_and_keeps_every_method():
    result=quality_summary(sample())
    assert len(result['rows'])==len(METHODS)
    assert all(r['mean_nll']==2.5 and r['ppl']==pytest.approx(math.exp(2.5)) for r in result['rows'])
    assert all(r['mean_nll']==2 for r in result['continuation'])


def test_missing_or_nonfinite_quality_is_rejected():
    data=sample();data['result']['rows'].pop()
    with pytest.raises(ValueError,match='matrix'):quality_summary(data)
    data=sample();data['result']['rows'][0]['nll_sum']=float('nan')
    with pytest.raises(ValueError,match='loss'):quality_summary(data)
