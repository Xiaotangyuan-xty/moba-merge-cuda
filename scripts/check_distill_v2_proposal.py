"""CPU checks for a proposal only; this file cannot run training."""
import hashlib
import json
from pathlib import Path
import random


def training_plan(r):
    lengths=r['context_cycle'];n=r['optimizer_updates_per_arm']
    ends=random.Random(r['seed']).sample(range(max(lengths)+128,r['train_token_count'],128),n)
    rows=[]
    for step,end in enumerate(ends,1):
        length=lengths[(step-1)%len(lengths)];rng=random.Random(r['seed']+100000+step)
        positions=rng.sample(range(length-1),128)
        positions+=rng.sample([i for i in range(length-513,length-1) if i not in positions],128)
        rows.append(dict(step=step,length=length,start=end-length,end=end,positions=sorted(positions)))
    return rows


def check(path='configs/distill_v2_proposal_20261103.json'):
    raw=Path(path).read_bytes();r=json.loads(raw)
    pilot=json.loads(Path('configs/distill_pilot_20261103.json').read_text())
    assert r['version']==2 and r['status']=='frozen_proposal_not_executed' and not r['runner_implemented']
    assert r['dense_layers']==pilot['dense_layers'] and r['sparse_layer_count']==12 and r['total_layers']==24
    assert r['corpus']==pilot['corpus'] and r['model']['config_sha256']==pilot['model_config_sha256']
    assert len(r['arms'])==2 and r['optimizer_updates_per_arm']==120
    plan=(r['context_cycle']*20)
    assert len(plan)==120 and sum(plan)==r['input_token_visits_per_arm']
    assert 120*r['predicted_positions_per_update']==r['supervised_positions_per_arm']
    schedule=training_plan(r)
    for row in schedule:
        assert len(set(row['positions']))==256 and min(row['positions'])>=0 and max(row['positions'])+1<row['length']
    schedule_sha=hashlib.sha256(json.dumps(schedule,separators=(',',':')).encode()).hexdigest()
    assert schedule_sha==r['training_schedule_sha256']
    assert len(set(r['validation_boundaries']))==8 and len(r['validation_lengths'])*8==24
    assert max(r['validation_boundaries'])+256<=270674
    test=r['future_test'];intervals=[]
    assert test['quality_evaluated'] is False
    assert test['excluded_intervals']==pilot['excluded_test_intervals']+[[49152,65792],[212992,229632]]
    for end in test['boundaries']:
        lo,hi=end-max(test['lengths']),end+test['continuation_targets']
        assert lo>=0 and hi<=test['token_count']
        assert not any(lo<b and hi>a for a,b in test['excluded_intervals'])
        assert not any(lo<b and hi>a for a,b in intervals)
        intervals.append((lo,hi))
    assert len(test['boundaries'])*len(test['lengths'])==r['quality_gate']['required_fresh_test_windows']==12
    return dict(status=r['status'],recipe_sha256=hashlib.sha256(raw).hexdigest(),updates_per_arm=120,
        input_token_visits_per_arm=sum(plan),supervised_positions_per_arm=r['supervised_positions_per_arm'],
        context_counts={str(n):plan.count(n) for n in sorted(set(plan))},validation_windows=24,fresh_test_windows=12,
        fresh_test_intervals=[list(x) for x in intervals],training_schedule_sha256=schedule_sha,fresh_test_evaluated=False,training_executed=False)


if __name__=='__main__':print(json.dumps(check(),indent=2))
