"""CPU-only, predeclared window construction and validation-only selection."""
import math


def windows(recipe,split):
    if split not in ('train','valid','test'):raise ValueError('unknown split')
    lengths=recipe['lengths'];ends=recipe['target_boundaries'][split]
    if len(lengths)!=len(set(lengths)) or len(ends)!=len(set(ends)):raise ValueError('duplicate windows')
    result=[]
    for length in lengths:
        for end in ends:
            if isinstance(length,bool) or not isinstance(length,int) or length<=recipe['tail_targets'] or end<length:
                raise ValueError('invalid window')
            result.append(dict(split=split,length=length,end=end,start=end-length,
                tail_start=end-recipe['tail_targets'],continuation_end=end+recipe['continuation_targets']))
    return result


def select(candidates,recipe,num_layers):
    expected={(w['length'],w['end']) for w in windows(recipe,'valid')}
    counts=[r['dense_layer_count'] for r in candidates]
    budgets=sorted({round(num_layers*f) for f in recipe['dense_fractions']})
    if counts!=budgets:raise ValueError('incomplete candidate budgets')
    for candidate in candidates:
        rows=candidate['rows'];keys=[(r['length'],r['end']) for r in rows]
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('incomplete validation matrix')
        for row in rows:
            if row['split']!='valid':raise ValueError('selection requires validation data')
            vals=[row[k] for k in ('tail_delta_nll','continuation_delta_nll')]
            if not all(math.isfinite(v) for v in vals):raise ValueError('nonfinite validation score')
            row['passes_calibration']=all(v<=recipe['calibration_delta_nll'] for v in vals)
        candidate['passes_all_lengths']=all(r['passes_calibration'] for r in rows)
    accepted=next((c for c in candidates if c['passes_all_lengths']),None)
    if accepted is None:raise ValueError('dense control did not pass; investigate integration')
    per_length={}
    for length in recipe['lengths']:
        c=next(c for c in candidates if all(r['passes_calibration'] for r in c['rows'] if r['length']==length))
        per_length[str(length)]=dict(dense_layers=c['dense_layers'],dense_layer_count=c['dense_layer_count'])
    return dict(shared_policy=dict(dense_layers=accepted['dense_layers'],dense_layer_count=accepted['dense_layer_count']),
                per_length_policies=per_length,shared_sparse_layers=num_layers-accepted['dense_layer_count'])


def aggregate(rows,recipe):
    if not rows:raise ValueError('empty quality matrix')
    split=rows[0]['split'];methods=list(dict.fromkeys(r['method'] for r in rows))
    expected={(m,w['length'],w['end']) for m in methods for w in windows(recipe,split)}
    actual=[(r['method'],r['length'],r['end']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected or any(r['split']!=split for r in rows):raise ValueError('incomplete quality matrix')
    result=[]
    for length in recipe['lengths']:
        for method in dict.fromkeys(r['method'] for r in rows):
            rr=[r for r in rows if r['length']==length and r['method']==method]
            if not rr:raise ValueError('missing quality method/length')
            for field,n in [('tail',recipe['tail_targets']),('continuation',recipe['continuation_targets'])]:
                total=sum(r[field+'_nll_sum'] for r in rr);targets=len(rr)*n
                if not math.isfinite(total):raise ValueError('nonfinite quality score')
                result.append(dict(length=length,method=method,field=field,targets=targets,
                                   mean_nll=total/targets,ppl=math.exp(total/targets)))
    for row in result:
        base=next(r for r in result if r['method']=='dense' and r['field']==row['field'] and r['length']==row['length'])
        row['delta_nll_vs_dense']=row['mean_nll']-base['mean_nll']
        row['ppl_ratio_vs_dense']=row['ppl']/base['ppl']
        row['meets_heldout_target']=row['delta_nll_vs_dense']<=recipe['heldout_delta_nll']
    return result
