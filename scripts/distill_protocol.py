"""Pure CPU experiment schedule and validation-only checkpoint selection."""
import math
import random


def training_plan(num_tokens,recipe):
    conf=recipe['training'];rng=random.Random(recipe['seed'])
    ends=rng.sample(range(max(conf['lengths'])+128,num_tokens,128),conf['updates'])
    result=[]
    for step,end in enumerate(ends,1):
        length=conf['lengths'][1] if step%4==0 else conf['lengths'][0]
        local=random.Random(recipe['seed']+100000+step);n=conf['predicted_positions']
        selected=local.sample(range(length-1),n//2)
        selected+=local.sample([i for i in range(length-513,length-1) if i not in selected],n-n//2)
        result.append(dict(step=step,length=length,start=end-length,end=end,positions=sorted(selected)))
    return result


def learning_rate(step,recipe):
    c=recipe['training']
    if not 1<=step<=c['updates']:raise ValueError('optimizer step outside schedule')
    if step<=c['warmup_updates']:return c['learning_rate']*step/c['warmup_updates']
    p=(step-c['warmup_updates'])/(c['updates']-c['warmup_updates'])
    return c['learning_rate']*(c['minimum_lr_ratio']+(1-c['minimum_lr_ratio'])*(1+math.cos(math.pi*p))/2)


def validation_score(rows,recipe):
    expected={(l,e) for l in recipe['lengths'] for e in recipe['target_boundaries']['valid']}
    actual=[(r['length'],r['end']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected or any(r['split']!='valid' for r in rows):raise ValueError('complete validation matrix required')
    values=[r[k] for r in rows for k in ('tail_delta_nll','continuation_delta_nll')]
    if not all(math.isfinite(x) for x in values):raise ValueError('nonfinite validation')
    return max(values)


def selected_step(checkpoints,recipe):
    expected=recipe['training']['validate_updates']
    if [c['step'] for c in checkpoints]!=expected:raise ValueError('all scheduled checkpoints required')
    scores=[(validation_score(c['rows'],recipe),c['step']) for c in checkpoints]
    return min(scores)[1]
