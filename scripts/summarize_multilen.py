"""CPU-only audit of frozen multi-length policies and complete paired samples."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from scripts.multilen_protocol import windows,select,aggregate


def same(a,b):
    if not math.isfinite(a) or not math.isclose(a,b,abs_tol=1e-10,rel_tol=1e-10):raise ValueError('inconsistent statistic')


def timing(data,recipe,source):
    rows=data['result']['rows'];expected={(m,l) for m in ('old_2k','shared','per_length') for l in recipe['lengths']}
    actual=[(r['method'],r['length']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete timing configurations')
    out=[];conf=recipe['latency']
    for row in rows:
        name=row['method'];samples=row['samples']
        expected={(r,s,m) for r in range(conf['rounds']) for s in range(conf['samples']) for m in ('dense',name)}
        actual=[(s['round'],s['sample'],s['method']) for s in samples]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete paired samples')
        for s in samples:
            order=['dense',name] if (s['round']+s['sample'])%2==0 else [name,'dense']
            if s['order']!=order or not math.isfinite(s['wall_ms']) or s['wall_ms']<=0:raise ValueError('invalid timing/order')
        a=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']=='dense') for r in range(conf['rounds'])]
        b=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']==name) for r in range(conf['rounds'])]
        speed=statistics.median(x/y for x,y in zip(a,b));same(speed,row['paired_speedup'])
        out.append(dict(source=source,model=data['model_key'],method=name,length=row['length'],sparse_layers=row['sparse_layer_count'],
            dense_ms=statistics.median(a),candidate_ms=statistics.median(b),paired_speedup=speed,
            dense_prefill_ms=row['dense_prefill_ms'],candidate_prefill_ms=row['candidate_prefill_ms'],
            dense_capture_ms=row['dense_capture_ms'],candidate_capture_ms=row['candidate_capture_ms']))
    return out


def summarize(directory,recipe,recipe_sha,pattern):
    records=[]
    for path in sorted(Path(directory).glob(pattern)):
        data=json.loads(path.read_text())
        if data.get('stage') not in ('calibrate','evaluate','latency') or 'recipe_sha256' not in data:continue
        if data['status']!='ok' or data['dirty'] or data['recipe_sha256']!=recipe_sha:raise ValueError('invalid source provenance')
        records.append((path,data))
    aggregates=[];quality=[];perf=[];calibration=[];policies=[]
    for key in recipe['models']:
        rr=[(p,d) for p,d in records if d['model_key']==key]
        cc=[x for x in rr if x[1]['stage']=='calibrate'];ee=[x for x in rr if x[1]['stage']=='evaluate'];tt=[x for x in rr if x[1]['stage']=='latency']
        if len(cc)!=1 or len(ee)!=1 or len(tt)!=recipe['latency']['runs']:raise ValueError('incomplete experiment stages')
        cp,cal=cc[0];n=len(cal['result']['ranking']['layer_rank']);order=cal['result']['ranking']['layer_rank']
        if set(order)!=set(range(n)):raise ValueError('invalid layer rank')
        for c in cal['result']['candidates']:
            if c['dense_layers']!=sorted(order[:c['dense_layer_count']]):raise ValueError('non-nested calibration family')
        selected=select(copy.deepcopy(cal['result']['candidates']),recipe,n)
        if any(selected[k]!=cal['result'][k] for k in selected) or cal['result']['test_used_for_selection']:raise ValueError('invalid frozen selection')
        policies.append(dict(model=key,**selected))
        for c in cal['result']['candidates']:
            for r in c['rows']:calibration.append(dict(model=key,dense_layers=c['dense_layer_count'],length=r['length'],end=r['end'],tail_delta_nll=r['tail_delta_nll'],continuation_delta_nll=r['continuation_delta_nll'],passes=r['passes_calibration']))
        expected_hash=hashlib.sha256(cp.read_bytes()).hexdigest()
        for _,d in ee+tt:
            if d['calibration_sha256']!=expected_hash:raise ValueError('changed calibration hash')
        ep,ev=ee[0];rows=ev['result']['rows'];methods={r['method'] for r in rows}
        if methods!={'dense','old_2k','shared','per_length'}:raise ValueError('incomplete quality methods')
        recomputed=aggregate(rows,recipe)
        if len(recomputed)!=len(ev['result']['aggregates']):raise ValueError('inconsistent quality aggregates')
        for a,b in zip(recomputed,ev['result']['aggregates']):
            for k,v in a.items():
                if isinstance(v,float):same(v,b[k])
                elif v!=b[k]:raise ValueError('inconsistent quality aggregates')
        expected_windows={(w['length'],w['end']):w for w in windows(recipe,'test')}
        target_hashes={}
        bases={(r['length'],r['end']):r for r in rows if r['method']=='dense'}
        for r in rows:
            w=expected_windows[r['length'],r['end']]
            if any(r[k]!=v for k,v in w.items()):raise ValueError('unexpected held-out window')
            hashes=(r['tail_target_sha256'],r['continuation_target_sha256'])
            if target_hashes.setdefault(r['end'],hashes)!=hashes:raise ValueError('targets are not end-aligned')
            base=bases[r['length'],r['end']]
            for field in ('tail','continuation'):same(r[field+'_delta_nll'],(r[field+'_nll_sum']-base[field+'_nll_sum'])/recipe[field+'_targets'])
            passed=all(r[k]<=recipe['heldout_delta_nll'] for k in ('tail_delta_nll','continuation_delta_nll'))
            if passed!=r['passes_heldout_window']:raise ValueError('inconsistent held-out verdict')
            quality.append(dict(model=key,method=r['method'],length=r['length'],end=r['end'],sparse_layers=r['sparse_layer_count'],tail_delta_nll=r['tail_delta_nll'],continuation_delta_nll=r['continuation_delta_nll'],passes=passed))
        for r in recomputed:aggregates.append(dict(model=key,**r))
        for tp,td in tt:
            for r in td['result']['rows']:
                source=next(x for x in rows if x['length']==r['length'] and x['end']==r['end'] and x['method']==r['method'])
                if source['input_sha256']!=r['input_sha256']:raise ValueError('timing input differs from quality input')
            perf.extend(timing(td,recipe,tp.name))
    return dict(policies=policies,quality_window_rows=len(quality),quality_aggregate_rows=len(aggregates),
                timing_rows=len(perf),regressions=sum(r['paired_speedup']<1 for r in perf),
                shared_failed_windows=[r for r in quality if r['method']=='shared' and not r['passes']]), \
           dict(calibration_all_rows=calibration,quality_all_windows=quality,quality_aggregates=aggregates,timing_all_rows=perf)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',required=True);p.add_argument('--out-dir',required=True)
    p.add_argument('--recipe',default='configs/multilen_20261103.json');p.add_argument('--pattern',default='multilen_20261103_*.json');args=p.parse_args()
    out=Path(args.out_dir)
    if (out/'summary.json').exists():raise SystemExit('preserve evidence')
    raw=Path(args.recipe).read_bytes();recipe=json.loads(raw)
    summary,tables=summarize(args.input_dir,recipe,hashlib.sha256(raw).hexdigest(),args.pattern)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in tables.items():
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
