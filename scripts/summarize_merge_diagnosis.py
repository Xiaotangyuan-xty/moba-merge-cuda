"""CPU-only reconstruction of merge ablations and expanded validation."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from scripts.multilen_protocol import windows,aggregate

MODES=('merged_master_bf16','merged_factor_bf16','merged_master_fp32','unmerged_lora_fp32')


def close(a,b):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6):raise ValueError('inconsistent statistic')


def matrix(rows,fields,expected):
    actual=[tuple(r[k] for k in fields) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete/duplicate diagnostic matrix')


def summarize(diagnostic,expanded,recipe,pilot,training,old_eval):
    steps=recipe['diagnostic_checkpoints'];ww=[w for w in windows(recipe,'valid') if w['end'] in recipe['diagnostic_boundaries']]
    dd=diagnostic['result'];ee=expanded['result'];rows=dd['rows'];route=dd['routes'];proj=dd['projections'];weights=dd['weights']
    expect={(s,w['length'],w['end'],m,r) for s in steps for w in ww for m in MODES for r in (False,True)}
    matrix(rows,['step','length','end','mode','replay'],expect)
    summary=[]
    hashes={}
    for row in rows:
        key=(row['length'],row['end'])
        if hashes.setdefault(key,row['input_sha256'])!=row['input_sha256']:raise ValueError('ablation input changed')
        if not all(math.isfinite(row[k]) for k in ('relative_l2','max_abs','kl','top1_agreement')):raise ValueError('nonfinite ablation metric')
        passed=all(row[k]<=v for k,v in pilot['merge_gate'].items())
        if passed!=row['passes_original_gate']:raise ValueError('changed merge gate verdict')
        if row['mode']=='merged_master_bf16' and not row['replay'] and row['step']==training['selected_step']:
            old=next(r for r in old_eval['merge_validation']['rows'] if (r['length'],r['end'])==(row['length'],row['end']))
            for k in ('relative_l2','max_abs','kl','top1_agreement'):close(row[k],old[k])
    for s in steps:
        for mode in MODES:
            for replay in (False,True):
                rr=[r for r in rows if (r['step'],r['mode'],r['replay'])==(s,mode,replay)]
                summary.append(dict(step=s,mode=mode,replay=replay,passes=sum(r['passes_original_gate'] for r in rr),cases=len(rr),
                    mean_relative_l2=statistics.mean(r['relative_l2'] for r in rr),max_relative_l2=max(r['relative_l2'] for r in rr),
                    max_abs=max(r['max_abs'] for r in rr),max_kl=max(r['kl'] for r in rr)))
    layers=set(range(pilot['num_layers']))-set(pilot['dense_layers'])
    matrix(route,['step','length','end','mode','replay','layer'],{x+(l,) for x in expect for l in layers})
    route_stats=[]
    for r in route:
        if not 0<=r['changed_query_heads']<=r['query_heads'] or not 0<=r['removed_edges']<=r['reference_edges']:raise ValueError('invalid route count')
    for s in steps:
        for m in MODES:
            rr=[r for r in route if r['step']==s and r['mode']==m and not r['replay']]
            route_stats.append(dict(step=s,mode=m,changed_query_head_fraction=sum(r['changed_query_heads'] for r in rr)/sum(r['query_heads'] for r in rr),
                removed_reference_edge_fraction=sum(r['removed_edges'] for r in rr)/sum(r['reference_edges'] for r in rr)))
    names={r['projection'] for r in weights}
    if len(names)!=pilot['num_layers']*4:raise ValueError('missing projection modules')
    matrix(weights,['step','projection'],{(s,n) for s in steps for n in names})
    matrix(proj,['step','length','end','projection','mode'],{(s,w['length'],w['end'],n,m) for s in steps for w in ww for n in names for m in (*MODES,'fp32_algebra_control')})
    for r in proj:
        if not all(math.isfinite(r[k]) for k in ('relative_l2','max_abs')):raise ValueError('nonfinite projection metric')
        if r['mode']=='fp32_algebra_control' and r['relative_l2']>1e-5:raise ValueError('FP32 algebra control failed')
    weight_stats=[]
    for s in steps:
        rr=[r for r in weights if r['step']==s]
        for r in rr:
            if not 0<=r['rounded_to_base_entries']<=r['nonzero_delta_entries']:raise ValueError('invalid weight count')
            if not all(math.isfinite(r[k]) for k in ('delta_norm','rounded_update_relative_l2','master_factor_delta_relative_l2')):raise ValueError('nonfinite weight metric')
        weight_stats.append(dict(step=s,rounded_to_base_fraction=sum(r['rounded_to_base_entries'] for r in rr)/sum(r['nonzero_delta_entries'] for r in rr),
            median_update_rounding_relative_l2=statistics.median(r['rounded_update_relative_l2'] for r in rr),
            median_factor_quantization_relative_l2=statistics.median(r['master_factor_delta_relative_l2'] for r in rr)))
    matrix(dd['controls'],['step','length','end'],{(s,w['length'],w['end']) for s in steps for w in ww})
    if not all(r['baseline_exact'] for r in dd['controls']):raise ValueError('baseline control failed')
    qr=ee['rows'];methods=['dense']+[f'step_{s:03d}' for s in recipe['expanded_checkpoints']]
    matrix(qr,['method','length','end'],{(m,w['length'],w['end']) for m in methods for w in windows(recipe,'valid')})
    aa=aggregate(qr,recipe)
    if len(aa)!=len(ee['aggregates']):raise ValueError('incomplete aggregate matrix')
    for a,b in zip(aa,ee['aggregates']):
        for k,v in a.items():
            if isinstance(v,float):close(v,b[k])
            elif v!=b[k]:raise ValueError('aggregate changed')
    targets={};expected_windows={(w['length'],w['end']):w for w in windows(recipe,'valid')}
    for r in qr:
        if r['split']!='valid' or r['sparse_layer_count']!=(0 if r['method']=='dense' else 12):raise ValueError('quality scope changed')
        if r['dense_layer_count']+r['sparse_layer_count']!=pilot['num_layers']:raise ValueError('layer count changed')
        if any(r[k]!=v for k,v in expected_windows[r['length'],r['end']].items()):raise ValueError('window coordinates changed')
        pair=(r['tail_target_sha256'],r['continuation_target_sha256'])
        if targets.setdefault(r['end'],pair)!=pair:raise ValueError('targets differ across lengths')
        base=next(b for b in qr if (b['method'],b['length'],b['end'])==('dense',r['length'],r['end']))
        for k in ('input_sha256','tail_target_sha256','continuation_target_sha256'):
            if r[k]!=base[k]:raise ValueError('quality input mismatch')
        for field in ('tail','continuation'):close(r[field+'_delta_nll'],(r[field+'_nll_sum']-base[field+'_nll_sum'])/recipe[field+'_targets'])
        if r['passes_heldout_window']!=all(r[k]<=recipe['heldout_delta_nll'] for k in ('tail_delta_nll','continuation_delta_nll')):raise ValueError('quality verdict mismatch')
        if r['method']!='dense' and r['end'] in recipe['diagnostic_boundaries']:
            cp=next(c for c in training['checkpoints'] if c['step']==int(r['method'][5:]))
            old=next(b for b in cp['rows'] if (b['length'],b['end'])==(r['length'],r['end']))
            for k in ('tail_nll_sum','continuation_nll_sum'):close(r[k],old[k])
    if ee['original_selected_step_unchanged']!=training['selected_step']:raise ValueError('retrospective checkpoint switch')
    matrix(ee['checkpoint_statistics'],['method'],{(m,) for m in methods})
    for stat in ee['checkpoint_statistics']:
        rr=[r for r in qr if r['method']==stat['method']];v=sorted(r[k] for r in rr for k in ('tail_delta_nll','continuation_delta_nll'))
        if stat['windows']!=len(rr) or stat['passed']!=sum(r['passes_heldout_window'] for r in rr):raise ValueError('checkpoint verdict changed')
        close(stat['mean_delta_nll'],statistics.mean(v));close(stat['worst_delta_nll'],max(v));close(stat['p90_delta_nll'],v[math.ceil(.9*len(v))-1])
    return dict(ablation_summary=summary,route_summary=route_stats,weight_summary=weight_stats,
        expanded_statistics=ee['checkpoint_statistics'],original_selected_step=training['selected_step'],new_test_used=False,new_training=False),dict(
        ablation_rows=rows,projection_rows=proj,route_rows=route,weight_rows=weights,expanded_windows=qr,expanded_aggregates=aa)


def main():
    p=argparse.ArgumentParser();p.add_argument('--diagnostic',required=True);p.add_argument('--expanded',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--recipe',default='configs/merge_diagnosis_20261103.json');args=p.parse_args()
    out=Path(args.out_dir)
    if out.exists():raise SystemExit('preserve evidence')
    recipe=json.loads(Path(args.recipe).read_text());pilot=json.loads(Path(recipe['pilot_recipe']).read_text())
    train=json.loads(Path(recipe['training_result']).read_text());ev=json.loads(Path('docs/data/distill_pilot_20261103_evaluate_01.json').read_text())
    data=[json.loads(Path(f).read_text()) for f in (args.diagnostic,args.expanded)]
    for d,stage in zip(data,['diagnose','expanded']):
        if d['stage']!=stage or d['status']!='ok' or d['dirty'] or not d['base_unchanged'] or d['physical_gpu'] not in (0,1):raise ValueError('bad run provenance')
        if d['recipe_sha256']!=hashlib.sha256(Path(args.recipe).read_bytes()).hexdigest() or d['training_result_sha256']!=hashlib.sha256(Path(recipe['training_result']).read_bytes()).hexdigest():raise ValueError('changed recipe/training result')
        if d['pilot_recipe_sha256']!=hashlib.sha256(Path(recipe['pilot_recipe']).read_bytes()).hexdigest() or d['valid_corpus_sha256']!=pilot['corpus']['valid']['sha256'] or d['valid_token_count']!=270674:raise ValueError('changed corpus/pilot recipe')
        if d['base_digest']!=train['result']['base_digest_after']:raise ValueError('base differs from pilot')
        source='c0b8118b8f0e565f29afa98abfd51cb2137be758' if stage=='diagnose' else '945ccda88f38381055fc55626d1d962edd8127c8'
        if d['git_head']!=source or d['dtype']!='bf16' or d['tf32']:raise ValueError('unexpected GPU source/options')
    result,tables=summarize(*data,recipe,pilot,train['result'],ev['result'])
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in tables.items():
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
