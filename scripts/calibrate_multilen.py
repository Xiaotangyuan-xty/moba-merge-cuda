"""Multi-length, end-aligned calibration; freeze before unseen test evaluation.

GPU1 only through the batch guard. No training, new dependencies or downloads.
One shared static layer policy is primary; length-specific policies secondary.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('RAYON_NUM_THREADS','2')
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from transformers import AutoModelForCausalLM,AutoTokenizer
from examples.full_qwen2 import FullQwen2Session
from moba_opt.grouped_attention import opt_gqa_cuda
from scripts.evaluate_sparse_adaptation import STABLE,cleanup,digest
from scripts.evaluate_full_model import sync_time
from scripts.multilen_protocol import windows,select,aggregate


def ids_hash(ids):return hashlib.sha256(struct.pack('<'+str(len(ids))+'I',*ids)).hexdigest()


def new_session(model,length,recipe,dense_layers):
    return FullQwen2Session(model,length+recipe['continuation_targets']+1,mode='moba',topk=4,
                           dense_layers=dense_layers,backend='model_graph',**STABLE)


def score(model,tokens,w,recipe,dense_layers):
    sample=tokens[w['start']:w['continuation_end']]
    length=w['length'];n=recipe['continuation_targets'];tail=recipe['tail_targets']
    if len(sample)!=length+n:raise ValueError('insufficient tokens for window')
    ids=torch.tensor([sample],device='cuda')
    with new_session(model,length,recipe,dense_layers) as s:
        h=s.prefill_hidden(ids[:,:length]);loss=0.
        for start in range(0,tail,128):
            a=length-tail-1+start;b=min(length-1,a+128)
            logits=model.lm_head(h[:,a:b]).float()
            loss+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,a+1:b+1].reshape(-1),reduction='sum'))
        logits=model.lm_head(h[:,-1:]);del h
        s.prepare();continued=0.
        for i in range(n):
            continued+=float(F.cross_entropy(logits[:,-1].float(),ids[:,length+i],reduction='sum'))
            if i+1<n:logits=s.step(sample[length+i])
        if not math.isfinite(loss+continued):raise ValueError('nonfinite score')
    return dict(w,tail_nll_sum=loss,continuation_nll_sum=continued,
                input_sha256=ids_hash(sample[:length]),tail_target_sha256=ids_hash(sample[length-tail:length]),
                continuation_target_sha256=ids_hash(sample[length:]),dense_layer_count=len(dense_layers),
                sparse_layer_count=len(model.model.layers)-len(dense_layers))


def rank(model,tokens,recipe):
    rows=[];n=len(model.model.layers)
    for w in windows(recipe,'train'):
        length=w['length'];sample=tokens[w['start']:w['end']]
        if len(sample)!=length:raise ValueError('insufficient training tokens')
        ids=torch.tensor([sample],device='cuda');values=[]
        with FullQwen2Session(model,length+1) as s:
            h=model.model.embed_tokens(ids)
            cos,sin=model.model.rotary_emb(h,torch.arange(length,device='cuda')[None])
            cu=torch.tensor([0,length],device='cuda',dtype=torch.int32)
            for i,layer in enumerate(model.model.layers):
                q,k,v=s.project(i,layer.input_layernorm(h),cos,sin)
                dense=flash_attn_func(q[None],k[None],v[None],causal=True)[0]
                sparse=opt_gqa_cuda(q,k,v,cu,length,dict(chunk_size=128,topk=4,**STABLE))
                d=layer.self_attn.o_proj(dense.reshape(1,length,-1))
                r=layer.self_attn.o_proj(sparse.reshape(1,length,-1))
                tail=recipe['tail_targets']
                error=(d[:,-tail:].float()-r[:,-tail:].float()).square().mean()
                scale=(h[:,-tail:].float()+d[:,-tail:].float()).square().mean().clamp_min(1e-12)
                values.append(float(error/scale))
                h=h+d;h=h+layer.mlp(layer.post_attention_layernorm(h))
        rows.append(dict(w,input_sha256=ids_hash(sample),layer_errors=values))
        del s,h,q,k,v,dense,sparse,d,r;cleanup()
    per_length={str(length):[statistics.mean(r['layer_errors'][i] for r in rows if r['length']==length) for i in range(n)] for length in recipe['lengths']}
    if not all(math.isfinite(v) for values in per_length.values() for v in values):raise ValueError('nonfinite ranking')
    worst=[max(scores[i] for scores in per_length.values()) for i in range(n)]
    order=sorted(range(n),key=lambda i:(-worst[i],i))
    return dict(rows=rows,per_length_errors=per_length,worst_length_errors=worst,layer_rank=order)


def calibrate(model,train,valid,recipe,parent=None):
    ranking=rank(model,train,recipe) if parent is None else copy.deepcopy(parent['ranking'])
    order=ranking['layer_rank'];n=len(order)
    reference=[] if parent is None else copy.deepcopy(parent['dense_reference'])
    if parent is None:
        for w in windows(recipe,'valid'):
            r=score(model,valid,w,recipe,list(range(n)));reference.append(r);cleanup()
    bases={(r['length'],r['end']):r for r in reference};candidates=[]
    prior={} if parent is None else {c['dense_layer_count']:c for c in parent['candidates']}
    budgets=recipe.get('dense_counts_by_layers',{}).get(str(n),sorted({round(n*f) for f in recipe['dense_fractions']}))
    for count in budgets:
        if count in prior:
            c=copy.deepcopy(prior[count]);c['reused_parent_candidate']=True;candidates.append(c)
            continue
        dense=sorted(order[:count]);rows=[]
        for w in windows(recipe,'valid'):
            base=bases[w['length'],w['end']]
            if count==n:r=dict(base,reused_dense_control=True)
            else:r=score(model,valid,w,recipe,dense)
            r['tail_delta_nll']=(r['tail_nll_sum']-base['tail_nll_sum'])/recipe['tail_targets']
            r['continuation_delta_nll']=(r['continuation_nll_sum']-base['continuation_nll_sum'])/recipe['continuation_targets']
            rows.append(r);cleanup()
        candidates.append(dict(dense_layer_count=count,dense_layers=dense,rows=rows))
        print(json.dumps(dict(candidate_dense_layers=count,max_tail_delta=max(r['tail_delta_nll'] for r in rows),
                              max_continuation_delta=max(r['continuation_delta_nll'] for r in rows))),flush=True)
    selected=select(candidates,recipe,n)
    return dict(ranking=ranking,dense_reference=reference,candidates=candidates,**selected,
                test_used_for_selection=False,policy_scope='static shared mask;validated at the three specified lengths and end-aligned windows')


def mask(method,length,frozen,recipe,key,n):
    if method=='dense':return list(range(n))
    if method=='old_2k':return recipe['models'][key]['old_policy']['dense_layers']
    if method=='shared':return frozen['shared_policy']['dense_layers']
    if method=='per_length':return frozen['per_length_policies'][str(length)]['dense_layers']
    raise ValueError('unknown method')


def evaluate(model,tokens,frozen,recipe,key):
    rows=[];n=len(model.model.layers)
    for w in windows(recipe,'test'):
        for method in ('dense','old_2k','shared','per_length'):
            r=score(model,tokens,w,recipe,mask(method,w['length'],frozen,recipe,key,n))
            r['method']=method;rows.append(r);cleanup()
            print(json.dumps(dict(length=w['length'],end=w['end'],method=method,
                                  tail_nll=r['tail_nll_sum']/recipe['tail_targets'],
                                  continuation_nll=r['continuation_nll_sum']/recipe['continuation_targets'])),flush=True)
    bases={(r['length'],r['end']):r for r in rows if r['method']=='dense'}
    for row in rows:
        base=bases[row['length'],row['end']]
        row['tail_delta_nll']=(row['tail_nll_sum']-base['tail_nll_sum'])/recipe['tail_targets']
        row['continuation_delta_nll']=(row['continuation_nll_sum']-base['continuation_nll_sum'])/recipe['continuation_targets']
        row['passes_heldout_window']=all(row[k]<=recipe['heldout_delta_nll'] for k in ('tail_delta_nll','continuation_delta_nll'))
    return dict(rows=rows,aggregates=aggregate(rows,recipe),test_windows_frozen_before_evaluation=True)


def latency_row(model,tokens,w,method,frozen,recipe,key):
    conf=recipe['latency'];count=conf['warmup']+conf['rounds']*conf['samples'];length=w['length']
    dense=new_session(model,length,recipe,list(range(len(model.model.layers))))
    selected=mask(method,length,frozen,recipe,key,len(model.model.layers))
    candidate=new_session(model,length,recipe,selected)
    try:
        sample=tokens[w['start']:w['end']+count];ids=torch.tensor([sample[:length]],device='cuda')
        _,a_setup=sync_time(lambda:dense.prefill_hidden(ids));_,b_setup=sync_time(lambda:candidate.prefill_hidden(ids))
        _,a_graph=sync_time(dense.prepare);_,b_graph=sync_time(candidate.prepare)
        samples=[]
        for i,token in enumerate(sample[length:]):
            if i<conf['warmup']:dense.step(token);candidate.step(token);continue
            r=(i-conf['warmup'])//conf['samples'];j=(i-conf['warmup'])%conf['samples']
            order=['dense',method] if (r+j)%2==0 else [method,'dense']
            for name in order:
                _,elapsed=sync_time(lambda:(dense if name=='dense' else candidate).step(token))
                samples.append(dict(round=r,sample=j,method=name,order=order,wall_ms=elapsed))
        a=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']=='dense') for r in range(conf['rounds'])]
        b=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']==method) for r in range(conf['rounds'])]
        return dict(length=length,end=w['end'],method=method,input_sha256=ids_hash(sample[:length]),
                    dense_layers=selected,sparse_layer_count=len(model.model.layers)-len(selected),samples=samples,
                    dense_step_ms=statistics.median(a),candidate_step_ms=statistics.median(b),
                    paired_speedup=statistics.median(x/y for x,y in zip(a,b)),
                    dense_prefill_ms=a_setup,candidate_prefill_ms=b_setup,dense_capture_ms=a_graph,candidate_capture_ms=b_graph)
    finally:dense.close();candidate.close()


def latency(model,tokens,frozen,recipe,key):
    rows=[]
    for w in windows(recipe,'test'):
        if w['end']!=recipe['target_boundaries']['test'][0]:continue
        for method in ('old_2k','shared','per_length'):
            r=latency_row(model,tokens,w,method,frozen,recipe,key);rows.append(r);cleanup()
            print(json.dumps({k:v for k,v in r.items() if k!='samples'}),flush=True)
    return dict(rows=rows,scope='complete model Graph vs same-model auto-split dense Graph;CPU management and token writes included;no sampling;setup single observations')


def require_gpu1():
    uuid=os.environ.get('MOBA_COVERAGE_GPU_UUID')
    if not uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid:
        raise SystemExit('Run through the approved GPU1 guard')
    index=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=index','--format=csv,noheader,nounits'],text=True).strip()
    if index!='1':raise SystemExit('Only physical GPU1 is authorized')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--model-key',choices=['0.5B','1.5B'],required=True)
    p.add_argument('--data-dir',required=True);p.add_argument('--recipe',default='configs/multilen_20261103.json')
    p.add_argument('--stage',choices=['calibrate','evaluate','latency'],required=True);p.add_argument('--calibration');p.add_argument('--out',required=True)
    p.add_argument('--parent-calibration')
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('Preserve existing evidence')
    require_gpu1();recipe=json.loads(Path(args.recipe).read_text());config_sha=digest(Path(args.model)/'config.json')
    if config_sha!=recipe['models'][args.model_key]['config_sha256']:raise ValueError('model configuration mismatch')
    frozen=None;parent=None
    if args.parent_calibration:
        cal=json.loads(Path(args.parent_calibration).read_text())
        if args.stage!='calibrate' or cal['stage']!='calibrate' or cal['status']!='ok' or cal['dirty'] or cal['recipe_sha256']!=recipe.get('parent_recipe_sha256') or cal['model_key']!=args.model_key or cal['model_config_sha256']!=config_sha:
            raise ValueError('parent calibration provenance mismatch')
        parent=cal['result']
    if args.stage!='calibrate':
        if not args.calibration:raise ValueError('frozen calibration required')
        cal=json.loads(Path(args.calibration).read_text())
        if cal['stage']!='calibrate' or cal['status']!='ok' or cal['dirty'] or cal['recipe_sha256']!=digest(args.recipe) or cal['model_key']!=args.model_key:raise ValueError('calibration provenance mismatch')
        frozen=cal['result']
    torch.set_num_threads(2);torch.manual_seed(20261103);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(recipe['resource']['allocator_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    splits=('train','valid') if args.stage=='calibrate' else ('test',)
    data={};data_hashes={}
    for split in splits:
        path=Path(args.data_dir)/f'wikitext2_{split}.txt';sha=digest(path)
        if sha!=recipe['corpus'][split]['sha256']:raise ValueError('corpus hash mismatch')
        data[split]=tok.encode(path.read_text(),add_special_tokens=False,verbose=False);data_hashes[split]=sha
        if max(w['continuation_end'] for w in windows(recipe,split))>len(data[split]):raise ValueError('insufficient corpus')
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',
        local_files_only=True,trust_remote_code=False).cuda().eval().requires_grad_(False)
    record=dict(stage=args.stage,model_key=args.model_key,model_config_sha256=config_sha,recipe_sha256=digest(args.recipe),
        corpus_sha256=data_hashes,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),dtype='bf16',batch=1,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),tf32=False,bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction)
    if args.calibration:record['calibration_sha256']=digest(args.calibration)
    if args.parent_calibration:record['parent_calibration_sha256']=digest(args.parent_calibration)
    with torch.inference_mode():
        if args.stage=='calibrate':result=calibrate(model,data['train'],data['valid'],recipe,parent)
        elif args.stage=='evaluate':result=evaluate(model,data['test'],frozen,recipe,args.model_key)
        else:result=latency(model,data['test'],frozen,recipe,args.model_key)
    record.update(status='ok',result=result);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(stage=args.stage,model=args.model_key,status='ok')),flush=True)


if __name__=='__main__':main()
