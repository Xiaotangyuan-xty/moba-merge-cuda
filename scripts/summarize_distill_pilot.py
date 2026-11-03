"""CPU-only audit of the fixed-mask100-update pilot, including failed gates."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
from scripts.distill_protocol import training_plan,learning_rate,validation_score,selected_step
from scripts.multilen_protocol import aggregate,windows


def close(a,b):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-6):raise ValueError('inconsistent statistic')


def ids_hash(ids):return hashlib.sha256(struct.pack('<'+str(len(ids))+'I',*ids)).hexdigest()


def quality_audit(result,recipe,methods):
    rows=result['rows']
    if {r['method'] for r in rows}!=set(methods):raise ValueError('incomplete quality methods')
    computed=aggregate(rows,recipe)
    if len(computed)!=len(result['aggregates']):raise ValueError('aggregate matrix mismatch')
    for a,b in zip(computed,result['aggregates']):
        for k,v in a.items():
            if isinstance(v,float):close(v,b[k])
            elif v!=b[k]:raise ValueError('aggregate mismatch')
    dense={(r['length'],r['end']):r for r in rows if r['method']=='dense'};hashes={}
    expected={(w['length'],w['end']):w for w in windows(recipe,rows[0]['split'])}
    for r in rows:
        sparse=0 if r['method']=='dense' else 12
        if r['sparse_layer_count']!=sparse or r['dense_layer_count']!=recipe['num_layers']-sparse:raise ValueError('sparse mask drift')
        if any(r[k]!=v for k,v in expected[r['length'],r['end']].items()):raise ValueError('window coordinates changed')
        pair=(r['tail_target_sha256'],r['continuation_target_sha256'])
        if hashes.setdefault(r['end'],pair)!=pair:raise ValueError('targets not end aligned')
        base=dense[r['length'],r['end']]
        if r['input_sha256']!=base['input_sha256']:raise ValueError('quality input mismatch')
        for field in ('tail','continuation'):close(r[field+'_delta_nll'],(r[field+'_nll_sum']-base[field+'_nll_sum'])/recipe[field+'_targets'])
        passed=all(r[k]<=recipe['heldout_delta_nll'] for k in ('tail_delta_nll','continuation_delta_nll'))
        if passed!=r['passes_heldout_window']:raise ValueError('window verdict mismatch')
    return computed


def summarize(directory,recipe,recipe_sha,pattern):
    sources={};paths={}
    for p in sorted(Path(directory).glob(pattern)):
        a=json.loads(p.read_text())
        if not isinstance(a,dict) or 'recipe_sha256' not in a or a.get('stage') not in ('preflight','train','evaluate','latency','diagnostic'):continue
        if a['status']!='ok' or a['dirty'] or a['recipe_sha256']!=recipe_sha:raise ValueError('bad source provenance')
        source='3ea13ee8ec11861bcd5ef6ea3a6d0ff015ca23b1' if a['stage']=='diagnostic' else '795aae862cb8dc8b3bb2d6972fceec7e7efe326c'
        if a['git_head']!=source or a['physical_gpu']!=1 or a['seed']!=recipe['seed'] or a['dtype']!='bf16' or a['tf32']:raise ValueError('unexpected run provenance')
        if any(a[k]!=recipe[k] for k in ('model_config_sha256','model_weight_sha256')):raise ValueError('wrong model provenance')
        if any(v!=recipe['corpus'][k]['sha256'] for k,v in a['corpus_sha256'].items()):raise ValueError('wrong corpus provenance')
        sources.setdefault(a['stage'],[]).append(a);paths.setdefault(a['stage'],[]).append(p)
    if any(len(sources.get(k,[]))!=1 for k in ('preflight','train','evaluate','diagnostic')) or len(sources.get('latency',[]))!=recipe['latency']['runs']:raise ValueError('incomplete experiment stages')
    train=sources['train'][0];tr=train['result'];ev=sources['evaluate'][0]['result'];diag=sources['diagnostic'][0]['result'];pf=sources['preflight'][0]['result']
    train_hash=hashlib.sha256(paths['train'][0].read_bytes()).hexdigest();ev_hash=hashlib.sha256(paths['evaluate'][0].read_bytes()).hexdigest()
    for k in ('evaluate','latency','diagnostic'):
        for d in sources[k]:
            if d['training_result_sha256']!=train_hash:raise ValueError('training source changed')
            if k!='evaluate' and d['evaluation_result_sha256']!=ev_hash:raise ValueError('evaluation source changed')
    if tr['base_digest_before']!=tr['base_digest_after'] or tr['teacher_logits_hash_before']!=tr['teacher_logits_hash_after']:raise ValueError('teacher/base changed')
    if tr['fixed_sparse_layers']!=12 or tr['trainable_parameters']!=recipe['lora']['parameters'] or not tr['base_weights_frozen'] or ev['mask_changed']:raise ValueError('wrong training scope')
    if not pf['teacher_unchanged'] or pf['base_digest_before']!=pf['base_digest_after'] or not 0<=pf['checkpoint_gradient_relative_l2']<0.03:raise ValueError('preflight failed')
    if len(pf['checkpoint_route_hashes'])!=12 or not all(len(v)>=2 and len(set(v))==1 for v in pf['checkpoint_route_hashes'].values()):raise ValueError('checkpoint routes differ')
    for gate in ('native_teacher_check','zero_adapter_check'):
        for metric in ('max_abs','relative_l2','kl'):close(pf[gate][metric],0)
        close(pf[gate]['top1_agreement'],1)
    probes=pf['memory_probes'];steps=pf['diagnostic_updates']
    if [r['length'] for r in probes]!=recipe['training']['lengths'] or [r['step'] for r in steps]!=[1,2,3] or not pf['diagnostic_updates_excluded_from_formal_training']:raise ValueError('incomplete preflight probes')
    for r in probes+steps:
        if not all(math.isfinite(r[k]) for k in ('loss','kl','ce','a_gradient_norm','b_gradient_norm','preclip_gradient_norm','peak_allocated_mib')) or r['b_gradient_norm']<=0:raise ValueError('invalid preflight gradients')
    if steps[0]['a_gradient_norm']!=0 or any(r['a_gradient_norm']<=0 for r in steps[1:]):raise ValueError('invalid LoRA initialization evolution')
    plan=training_plan(2549174,recipe) # fixed tokenizer on the pinned train corpus
    if tr['optimizer_updates']!=100 or len(tr['updates'])!=100 or tr['input_tokens']!=358400 or tr['supervised_positions']!=6400:raise ValueError('wrong update/token count')
    for row,item in zip(tr['updates'],plan):
        if any(row[k]!=item[k] for k in ('step','length','start','end')) or row['positions_sha256']!=ids_hash(item['positions']):raise ValueError('training schedule mismatch')
        close(row['learning_rate'],learning_rate(row['step'],recipe))
        close(row['loss'],recipe['training']['kl_weight']*recipe['training']['temperature']**2*row['kl']+(1-recipe['training']['kl_weight'])*row['ce'])
        if not all(math.isfinite(row[k]) for k in ('loss','a_gradient_norm','b_gradient_norm','preclip_gradient_norm')) or row['b_gradient_norm']<=0:raise ValueError('invalid update gradients')
    selected=selected_step(tr['checkpoints'],recipe)
    if selected!=tr['selected_step'] or selected!=ev['selected_step'] or selected!=diag['selected_step_unchanged']:raise ValueError('test-based checkpoint switch')
    for cp in tr['checkpoints']:close(cp['score'],validation_score(cp['rows'],recipe))
    files=tr['adapter_files']
    if [f['step'] for f in files]!=recipe['training']['validate_updates'] or tr['selected_adapter']!=next(f for f in files if f['step']==selected) or tr['last_adapter']!=files[-1]:raise ValueError('wrong adapter checkpoint linkage')
    baseline=tr['baseline_validation']
    dense_valid=[dict(r,tail_delta_nll=0.0,continuation_delta_nll=0.0,passes_heldout_window=True) for r in baseline['dense']]
    valid_rows=dense_valid+baseline['untrained']
    quality_audit(dict(rows=valid_rows,aggregates=aggregate(valid_rows,recipe)),recipe,['dense','untrained'])
    for cp in tr['checkpoints']:
        valid_rows=dense_valid+cp['rows']
        quality_audit(dict(rows=valid_rows,aggregates=aggregate(valid_rows,recipe)),recipe,['dense','student'])
    gates=[]
    for label,gate in [('selected',ev['merge_validation']),('last',diag['last_merge_validation'])]:
        expected={(w['length'],w['end']) for w in windows(recipe,'valid')}
        if {(r['length'],r['end']) for r in gate['rows']}!=expected or len(gate['rows'])!=len(expected):raise ValueError('incomplete merge matrix')
        for r in gate['rows']:
            if not all(math.isfinite(r[k]) for k in recipe['merge_gate']):raise ValueError('nonfinite merge metric')
            passed=all(r[k]<=v for k,v in recipe['merge_gate'].items())
            if passed!=r['passes']:raise ValueError('merge verdict mismatch')
            gates.append(dict(checkpoint=label,**r))
        if gate['passes_all']!=all(r['passes'] for r in gate['rows']):raise ValueError('merge verdict mismatch')
    primary=quality_audit(ev,recipe,recipe['evaluation_methods'])
    secondary=quality_audit(diag,recipe,['dense','untrained','last_unmerged','last_merged'])
    previous={(r['method'],r['length'],r['end']):r for r in ev['rows']}
    for r in diag['rows']:
        if r['method'] in ('dense','untrained','last_merged'):
            if r!=previous[r['method'],r['length'],r['end']]:raise ValueError('diagnostic repeat differs')
    if not diag['last_merged_repeat_equal'] or diag['last_step']!=files[-1]['step']:raise ValueError('diagnostic checkpoint mismatch')
    perf=[];conf=recipe['latency']
    for p,data in zip(paths['latency'],sources['latency']):
        rr=data['result']['rows'];keys=[(r['length'],r['method']) for r in rr]
        if len(keys)!=len(set(keys)) or set(keys)!={(l,m) for l in recipe['lengths'] for m in ('untrained','selected_unmerged','selected_merged')}:raise ValueError('incomplete timing matrix')
        for r in rr:
            m=r['method'];ss=r['samples'];expected={(i,j,k) for i in range(conf['rounds']) for j in range(conf['samples']) for k in ('dense',m)}
            actual=[(x['round'],x['sample'],x['method']) for x in ss]
            if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete timing pairs')
            for x in ss:
                order=['dense',m] if (x['round']+x['sample'])%2==0 else [m,'dense']
                if x['order']!=order or x['wall_ms']<=0 or not math.isfinite(x['wall_ms']):raise ValueError('bad timing sample/order')
            sequence=[(i,j,k) for i in range(conf['rounds']) for j in range(conf['samples']) for k in (['dense',m] if (i+j)%2==0 else [m,'dense'])]
            if actual!=sequence:raise ValueError('timing execution order changed')
            a=[statistics.median(x['wall_ms'] for x in ss if x['round']==i and x['method']=='dense') for i in range(conf['rounds'])]
            b=[statistics.median(x['wall_ms'] for x in ss if x['round']==i and x['method']==m) for i in range(conf['rounds'])]
            close(r['paired_speedup'],statistics.median(x/y for x,y in zip(a,b)))
            close(r['dense_step_ms'],statistics.median(a));close(r['candidate_step_ms'],statistics.median(b))
            if r['sparse_layers']!=12 or r['merge_eligible']!=(ev['merge_validation']['passes_all'] if m=='selected_merged' else True):raise ValueError('invalid timing eligibility')
            q=next(x for x in ev['rows'] if x['length']==r['length'] and x['end']==r['end'] and x['method']==m)
            if q['input_sha256']!=r['input_sha256']:raise ValueError('timing input mismatch')
            perf.append(dict(source=p.name,**{k:v for k,v in r.items() if k!='samples'}))
    verdict=[]
    for scope,rows in [('primary',ev['rows']),('posthoc_diagnostic',diag['rows'])]:
        for m in dict.fromkeys(r['method'] for r in rows):
            rr=[r for r in rows if r['method']==m]
            verdict.append(dict(scope=scope,method=m,passed=sum(r['passes_heldout_window'] for r in rr),windows=len(rr),worst_delta_nll=max(max(r['tail_delta_nll'],r['continuation_delta_nll']) for r in rr)))
    summary=dict(selected_step=selected,optimizer_updates=100,trainable_parameters=tr['trainable_parameters'],base_unchanged=True,
        selected_merge_passed=ev['merge_validation']['passes_all'],last_merge_passed=diag['last_merge_validation']['passes_all'],
        verdicts=verdict,timing_rows=len(perf),last_merged_repeat_equal=diag['last_merged_repeat_equal'])
    return summary,dict(training_updates=tr['updates'],validation_checkpoints=[dict(step=c['step'],worst_delta_nll=c['score']) for c in tr['checkpoints']],
        validation_windows=[dict(step=c['step'],**r) for c in tr['checkpoints'] for r in c['rows']],
        quality_primary=primary,quality_diagnostic=secondary,quality_primary_windows=ev['rows'],quality_diagnostic_windows=diag['rows'],
        merge_gates=gates,timing_all_rows=perf)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--recipe',default='configs/distill_pilot_20261103.json');p.add_argument('--pattern',default='distill_pilot_20261103_*.json');args=p.parse_args()
    out=Path(args.out_dir)
    if (out/'summary.json').exists():raise SystemExit('preserve evidence')
    raw=Path(args.recipe).read_bytes();summary,tables=summarize(args.input_dir,json.loads(raw),hashlib.sha256(raw).hexdigest(),args.pattern)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
