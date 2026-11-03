import math
import pytest
from scripts.summarize_full_model import quality_summary,latency_summary,METHODS


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


def latency_sample():
    samples=[]
    for i in range(2):
        order=['hf_fa2','dense_modelgraph'] if i==0 else ['dense_modelgraph','hf_fa2']
        for name in order:
            samples.append(dict(round=0,sample=i,impl=name,order=order,wall_ms=2 if name=='hf_fa2' else 3))
    row=dict(method='dense_modelgraph',prefix=256,samples=samples,first_decode_ms={},
             prepare_capture_ms=10,hf_init_ms=1,candidate_init_ms=2,hf_prefill_ms=4,candidate_prefill_ms=6)
    return dict(status='ok',dirty=False,result=dict(rounds=1,samples_per_round=2,rows=[row]))


def test_latency_keeps_regressions_and_does_not_claim_amortization():
    r=latency_summary(latency_sample(),['dense_modelgraph'],[256])[0]
    assert r['paired_speedup']==pytest.approx(2/3)
    assert r['capture_break_even_tokens'] is None
    assert r['request_break_even_decode_steps'] is None


def test_latency_rejects_missing_pairs_and_wrong_ab_order():
    data=latency_sample();data['result']['rows'][0]['samples'].pop()
    with pytest.raises(ValueError,match='samples'):latency_summary(data,['dense_modelgraph'],[256])
    data=latency_sample();data['result']['rows'][0]['samples'][0]['order'].reverse()
    with pytest.raises(ValueError,match='order'):latency_summary(data,['dense_modelgraph'],[256])
