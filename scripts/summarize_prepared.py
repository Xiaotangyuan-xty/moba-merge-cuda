"""CPU-only full-matrix validation and matched-contract aggregation."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from scripts.summarize_decode import metrics


def summarize(path,configs):
    raw=[json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    expected={(cid,contract,mode) for cid in configs for contract in ('strict','trusted') for mode in ('eager','graph')}
    keys=[(x['config']['id'],x['contract'],x['mode']) for x in raw]
    if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('incomplete/duplicate matrix')
    heads={x['git_head'] for x in raw}
    if len(heads)!=1:raise ValueError('mixed source revisions')
    rows=[]
    for x in raw:
        cfg=x['config'];name='prepared_'+x['mode']
        if x['status']!='ok' or x['working_tree_dirty'] or cfg!=configs[cfg['id']]:
            raise ValueError('failed/dirty/config-modified run')
        row=dict(run=path.stem,**cfg,contract=x['contract'],mode=x['mode'],seed=x['seed'],git_head=x['git_head'])
        for workload in ('attend','step'):
            samples=x[workload]
            for r in samples:
                order=['cab3_eager',name] if (r['round']+r['sample'])%2==0 else [name,'cab3_eager']
                if r['order']!=order:raise ValueError('incorrect AB/BA order')
            m=metrics(samples,['cab3_eager',name],x['rounds'],x['samples'])
            for clock in ('wall_ms','stream_ms'):
                a,b=m['cab3_eager'][clock],m[name][clock]
                row[f'{workload}_baseline_{clock}']=statistics.median(a)
                row[f'{workload}_prepared_{clock}']=statistics.median(b)
                row[f'{workload}_{clock}_speedup']=statistics.median(a/b for a,b in zip(a,b))
        dense=metrics(x['dense_reference'],['dense_fa_core'],x['rounds'],x['samples'])
        row['dense_fa_core_wall_ms']=statistics.median(dense['dense_fa_core']['wall_ms'])
        row['metadata_delta_uploads']=x['metadata_delta_uploads']
        row['append_calls']=x['append_calls']
        if x.get('sampled_fp64'):
            row.update(x['sampled_fp64'])
        rows.append(row)
    groups=[]
    for contract in ('strict','trusted'):
        for mode in ('eager','graph'):
            sub=[r for r in rows if r['contract']==contract and r['mode']==mode]
            g=dict(contract=contract,mode=mode,configs=len(sub),workloads={})
            for workload in ('attend','step'):
                vals=[r[f'{workload}_wall_ms_speedup'] for r in sub]
                g['workloads'][workload]=dict(min=min(vals),max=max(vals),geomean=math.exp(statistics.mean(map(math.log,vals))),
                    regression_ids=[r['id'] for r in sub if r[f'{workload}_wall_ms_speedup']<1])
            if contract=='trusted':
                g['dense_core_faster_ids']=[r['id'] for r in sub if r['dense_fa_core_wall_ms']<r['attend_prepared_wall_ms']]
            groups.append(g)
    return rows,dict(input=path.name,git_head=next(iter(heads)),groups=groups)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('inputs',nargs='+')
    p.add_argument('--config',default='configs/decode_perf.json')
    p.add_argument('--out-dir',required=True)
    args=p.parse_args()
    configs={c['id']:c for c in json.loads(Path(args.config).read_text())}
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'all_rows.csv').exists() or (out/'summary.json').exists():raise SystemExit('preserve evidence')
    rows,runs=[],[]
    for path in map(Path,args.inputs):
        rr,ss=summarize(path,configs);rows.extend(rr);runs.append(ss)
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'all_rows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    result=dict(statistic='median of round baseline/prepared median ratios; all config rows retained',runs=runs,
                finite_contract='strict vs strict; trusted vs trusted; FA core has no value checking/cache management')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
