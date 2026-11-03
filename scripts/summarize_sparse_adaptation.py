"""CPU-only validation/export of all sparse adaptation experiment rows."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

METHODS={'dense','stable_k4','stable_k8','calibrated_k4','dense_prefill_k4'}
LONG=METHODS-{'dense'}


def close(a,b):
    if not math.isfinite(a) or not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10):raise ValueError('inconsistent statistic')


def quality_rows(data,source):
    rows=data['result']['rows']
    if len(rows)!=5 or {r['method'] for r in rows}!=METHODS:raise ValueError('incomplete quality matrix')
    ref=next(r for r in rows if r['method']=='dense');out=[]
    for row in rows:
        for field,targets in (('prefill',2047),('continuation',128)):
            score=row[field];blocks=score['blocks']
            if len(blocks)!=4 or {b['block'] for b in blocks}!={0,1,2,3} or any(b['tokens']!=targets for b in blocks):raise ValueError('incomplete quality blocks')
            loss=sum(b['nll_sum'] for b in blocks);n=sum(b['tokens'] for b in blocks)
            close(score['mean_nll'],loss/n);close(score['ppl'],math.exp(loss/n))
            out.append(dict(source=source,layers=data['model_config']['num_hidden_layers'],method=row['method'],
                field=field,targets=n,mean_nll=loss/n,ppl=math.exp(loss/n),
                delta_nll_vs_dense=loss/n-ref[field]['mean_nll'],ppl_ratio_vs_dense=math.exp(loss/n)/ref[field]['ppl']))
    return out


def long_rows(data,source):
    rows=data['result']['rows'];keys=[(r['method'],r['prefix']) for r in rows]
    expected={(m,n) for m in LONG for n in (8192,16384,32704)}
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('incomplete long-context matrix')
    out=[]
    for row in rows:
        clean=dict(source=source,layers=data['model_config']['num_hidden_layers'],method=row['method'],prefix=row['prefix'],
                   status=row['status'],dense_step_ms=None,candidate_step_ms=None,speedup=None,tail_delta_nll=None,continuation_delta_nll=None)
        if row['status']=='ok':
            if row['tail_targets']!=512 or row['continuation_targets']!=24 or not all(math.isfinite(row[k]) for k in ('tail_nll_sum','dense_tail_nll_sum','candidate_continuation_nll','dense_continuation_nll')):
                raise ValueError('invalid quality diagnostic')
            samples=row['samples'];name=row['method']
            expected={(r,s,m) for r in range(3) for s in range(8) for m in ('dense',name)}
            actual=[(s['round'],s['sample'],s['method']) for s in samples]
            if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete paired samples')
            for s in samples:
                order=['dense',name] if (s['round']+s['sample'])%2==0 else [name,'dense']
                if s['order']!=order or not math.isfinite(s['wall_ms']) or s['wall_ms']<=0:raise ValueError('invalid AB/BA sample')
            a=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']=='dense') for r in range(3)]
            b=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']==name) for r in range(3)]
            speed=statistics.median(x/y for x,y in zip(a,b));close(speed,row['speedup_vs_dense_graph'])
            clean.update(dense_step_ms=statistics.median(a),candidate_step_ms=statistics.median(b),speedup=speed,
                tail_delta_nll=(row['tail_nll_sum']-row['dense_tail_nll_sum'])/row['tail_targets'],
                continuation_delta_nll=(row['candidate_continuation_nll']-row['dense_continuation_nll'])/row['continuation_targets'])
        elif row['status']!='allocator_oom_within_approved_cap':raise ValueError('unknown row status')
        out.append(clean)
    return out


def summarize(directory,pattern='stable_quality_20261103_*.json'):
    quality=[];timing=[];diagnostics=[];calibrations=[]
    for path in sorted(Path(directory).glob(pattern)):
        data=json.loads(path.read_text());stage=data.get('stage')
        if stage not in ('evaluate','long','retrieve','stability','calibrate','integration'):continue
        if data['status']!='ok' or data['dirty']:raise ValueError('failed/dirty source')
        if stage=='evaluate':quality.extend(quality_rows(data,path.name))
        elif stage=='long':timing.extend(long_rows(data,path.name))
        elif stage=='calibrate':
            result=data['result'];chosen=next(r for r in result['candidates'] if r['passes_calibration'])
            if chosen['policy']!=result['policy'] or result['test_data_used_for_selection']:raise ValueError('invalid frozen policy')
            calibrations.append(dict(source=path.name,policy=result['policy'],sparse_layers=result['sparse_layers'],threshold=result['threshold_delta_nll']))
        elif stage=='stability':
            for row in data['result']['rows']:
                diagnostics.append(dict(source=path.name,method=row['method'],scope='within-process identical prefill',
                    all_logits_equal=all(r['logits_equal'] for r in row['repeats']),
                    max_relative_l2=max(r['relative_l2'] for r in row['repeats']),
                    max_differing_route_layers=max(r['differing_route_layers'] for r in row['repeats'])))
        elif stage=='retrieve':
            rows=data['result']['rows']
            expected={(m,l,p) for m in ('dense','stable_k4','calibrated_k4') for l in (8192,16384,32704) for p in (25,75)}
            actual=[(r['method'],r['length'],int(100*r['needle_token_position']/r['length'])) for r in rows]
            if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete retrieval matrix')
            for name in ('dense','stable_k4','calibrated_k4'):
                rr=[r for r in rows if r['method']==name];ok=[r for r in rr if r['status']=='ok']
                diagnostics.append(dict(source=path.name,method=name,scope='six synthetic cases, preserve OOM',correct=sum(r['exact_match'] for r in ok),completed=len(ok),oom=len(rr)-len(ok)))
    if not quality or not timing or not calibrations:raise ValueError('missing experiment groups')
    return quality,timing,dict(calibrations=calibrations,diagnostics=diagnostics,
        quality_rows=len(quality),timing_rows=len(timing),timing_regressions=sum(r['speedup'] is not None and r['speedup']<1 for r in timing),
        timing_oom=sum(r['status']!='ok' for r in timing))


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--pattern',default='stable_quality_20261103_*.json');args=p.parse_args()
    out=Path(args.out_dir)
    if (out/'summary.json').exists():raise SystemExit('preserve existing summary')
    quality,timing,summary=summarize(args.input_dir,args.pattern);out.mkdir(exist_ok=True,parents=True)
    for name,rows in [('quality_all_rows.csv',quality),('timing_all_rows.csv',timing)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
