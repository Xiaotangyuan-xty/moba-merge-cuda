"""CPU-only weighted quality and complete paired latency reports."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

METHODS=('hf_sdpa','hf_fa2','dense_eager','moba_k2_eager','moba_k4_eager','moba_k8_eager','moba_k4_packed_eager')


def quality_summary(data):
    if data['status']!='ok' or data['dirty']:raise ValueError('failed/dirty quality source')
    q=data['result'];result={}
    for field in ('rows','continuation'):
        rows=q[field];key='offset' if field=='rows' else 'block'
        keys=[(r['method'],r[key]) for r in rows]
        expected={(m,i*q['block_length'] if field=='rows' else i) for m in METHODS for i in range(q['blocks'])}
        if len(keys)!=len(set(keys)) or set(keys)!=expected:raise ValueError('incomplete quality matrix')
        aggregates=[]
        for method in METHODS:
            rr=[r for r in rows if r['method']==method]
            n=sum(r['tokens'] for r in rr);loss=sum(r['nll_sum'] for r in rr)
            if n<=0 or not math.isfinite(loss):raise ValueError('invalid loss')
            aggregates.append(dict(method=method,tokens=n,nll_sum=loss,mean_nll=loss/n,ppl=math.exp(loss/n)))
        ref=next(r for r in aggregates if r['method']=='hf_fa2')
        for r in aggregates:
            r['delta_nll_vs_hf_fa2']=r['mean_nll']-ref['mean_nll']
            r['ppl_ratio_vs_hf_fa2']=r['ppl']/ref['ppl']
        result[field]=aggregates
    return result


def latency_summary(data,methods,lengths):
    if data['status']!='ok' or data['dirty']:raise ValueError('failed/dirty latency source')
    q=data['result'];rows=q['rows']
    keys=[(r['method'],r['prefix']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!={(m,l) for m in methods for l in lengths}:
        raise ValueError('incomplete latency matrix')
    out=[]
    for row in rows:
        name=row['method'];samples=row['samples']
        expected={(r,s,n) for r in range(q['rounds']) for s in range(q['samples_per_round']) for n in ('hf_fa2',name)}
        actual=[(x['round'],x['sample'],x['impl']) for x in samples]
        if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete timing samples')
        for x in samples:
            order=['hf_fa2',name] if (x['round']+x['sample'])%2==0 else [name,'hf_fa2']
            if x['order']!=order or not math.isfinite(x['wall_ms']) or x['wall_ms']<=0:raise ValueError('invalid timing/order')
        a=[];b=[]
        for r in range(q['rounds']):
            a.append(statistics.median(x['wall_ms'] for x in samples if x['round']==r and x['impl']=='hf_fa2'))
            b.append(statistics.median(x['wall_ms'] for x in samples if x['round']==r and x['impl']==name))
        clean={k:v for k,v in row.items() if k not in ('samples','first_decode_ms')}
        clean['hf_step_ms']=statistics.median(a);clean['candidate_step_ms']=statistics.median(b)
        clean['paired_speedup']=statistics.median(x/y for x,y in zip(a,b))
        saving=clean['hf_step_ms']-clean['candidate_step_ms']
        clean['capture_break_even_tokens']=math.ceil(clean['prepare_capture_ms']/saving) if saving>0 else None
        extra=(clean['candidate_init_ms']+clean['candidate_prefill_ms']+clean['prepare_capture_ms']
               -clean['hf_init_ms']-clean['hf_prefill_ms'])
        clean['extra_request_setup_ms']=extra
        clean['request_break_even_decode_steps']=math.ceil(max(0,extra)/saving) if saving>0 else None
        out.append(clean)
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--quality');p.add_argument('--latency',nargs='*');p.add_argument('--out-dir',required=True)
    p.add_argument('--methods',default='dense_eager,dense_modelgraph,moba_k2_modelgraph,moba_k4_modelgraph,moba_k8_modelgraph,moba_k4_modelgraph_packed')
    p.add_argument('--lengths',default='256,2048,8192');args=p.parse_args()
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise SystemExit('preserve evidence')
    result={};all_rows=[]
    if args.quality:result['quality']=quality_summary(json.loads(Path(args.quality).read_text()))
    for name in args.latency or []:
        rr=latency_summary(json.loads(Path(name).read_text()),args.methods.split(','),list(map(int,args.lengths.split(','))))
        for r in rr:r['run']=Path(name).stem
        all_rows.extend(rr)
    if all_rows:
        with (out/'latency_all_rows.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
        result['latency_rows']=len(all_rows)
        result['regressions']=[dict(run=r['run'],method=r['method'],prefix=r['prefix']) for r in all_rows if r['paired_speedup']<1]
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
