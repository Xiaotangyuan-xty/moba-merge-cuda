"""Frozen-data calibration, route reproducibility and long-context experiments.

Inference-only policy adaptation: no weights are trained. Run GPU stages only
through the approved shared-card guard. Test data never chooses a policy.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('RAYON_NUM_THREADS','2')
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from transformers import AutoModelForCausalLM,AutoTokenizer
import moba_opt.grouped_attention as grouped
from examples.full_qwen2 import FullQwen2Session
from scripts.evaluate_full_model import head_nll,dense_metrics,sync_time,Native,assert_dense
from examples.retrieval_cases import retrieval_cases

STABLE=dict(deterministic_merge=True,stable_routes=True,route_precision='fp32')
METHODS=('dense','stable_k4','stable_k8','calibrated_k4','dense_prefill_k4')


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session(model,name,capacity,policy=None,backend='model_graph'):
    opts=dict(STABLE)
    mode='dense' if name=='dense' else 'moba'
    topk=8 if name=='stable_k8' else 4
    if name=='calibrated_k4':
        if policy is None:raise ValueError('frozen calibration required')
        opts['dense_layers']=policy['dense_layers']
    elif name=='dense_prefill_k4':opts['prefill_mode']='dense'
    elif name not in ('dense','stable_k4','stable_k8'):raise ValueError(name)
    return FullQwen2Session(model,capacity,mode=mode,topk=topk,backend=backend,**opts)


def cleanup():gc.collect();torch.cuda.empty_cache()


def integration(model,tokens):
    ids=torch.tensor([tokens[:128]],device='cuda');targets=tokens[128:140]
    native=Native(model)
    refs=[native.prefill(ids).clone()]
    for t in targets:refs.append(native.step(t).clone())
    native.close();rows=[]
    for name in ('dense','stable_k4'):
        with session(model,name,160) as s:
            h=s.prefill_hidden(ids);checks=[dense_metrics(refs[0],model.lm_head(h[:,-1:]))];del h
            s.prepare()
            for i,t in enumerate(targets):checks.append(dense_metrics(refs[i+1],s.step(t)))
            for c in checks:assert_dense(c)
            assert all(c.lengths==(140,) for c in s.caches)
        rows.append(dict(method=name,all_selected=True,checks=checks));del s;cleanup()
    return dict(scope='native HF FA2 versus complete dense/all-selected Graph;128prefix+12steps;fixed prior tolerances',rows=rows)


def score_policy(model,tokens,name,policy,blocks=4,length=2048,continuation=128):
    rows=[];continued=[]
    for block in range(blocks):
        window=tokens[block*length:(block+1)*length]
        if len(window)!=length:raise ValueError('insufficient tokens')
        ids=torch.tensor([window],device='cuda')
        with session(model,name,length+1,policy) as s:
            h=s.prefill_hidden(ids);loss,n=head_nll(model,h,ids)
            rows.append(dict(block=block,tokens=n,nll_sum=loss));del h
        del s;cleanup()
        prefix=window[:-continuation];targets=window[-continuation:]
        gpu_targets=torch.tensor(targets,device='cuda')
        with session(model,name,length+1,policy) as s:
            h=s.prefill_hidden(torch.tensor([prefix],device='cuda'))
            logits=model.lm_head(h[:,-1:]);del h
            s.prepare();loss=0.
            for i,token in enumerate(targets):
                loss+=float(F.cross_entropy(logits[:,-1].float(),gpu_targets[i:i+1],reduction='sum'))
                if i+1<len(targets):logits=s.step(token)
            continued.append(dict(block=block,tokens=continuation,nll_sum=loss))
        del s,logits;cleanup()
    def aggregate(rr):
        n=sum(r['tokens'] for r in rr);loss=sum(r['nll_sum'] for r in rr)
        if not math.isfinite(loss):raise ValueError('nonfinite loss')
        return dict(tokens=n,nll_sum=loss,mean_nll=loss/n,ppl=math.exp(loss/n),blocks=rr)
    return dict(method=name,prefill=aggregate(rows),continuation=aggregate(continued))


def stability(model,tokens):
    rows=[]
    for name,options in [('atomic',{}),('canonical',dict(deterministic_merge=True)),('fp32_stable',STABLE)]:
        first=None;routes0=None;checks=[]
        for repeat in range(4):
            routes=[]
            grouped.ROUTE_HOOK=lambda **kw:routes.append(hashlib.sha256(kw['selected'].cpu().numpy().tobytes()).hexdigest())
            try:
                with FullQwen2Session(model,2049,mode='moba',topk=4,**options) as s:
                    h=s.prefill_hidden(torch.tensor([tokens[:2048]],device='cuda'))
                    logits=model.lm_head(h[:,-16:]).clone();del h
                    if first is None:first=logits.clone();routes0=routes
                    checks.append(dict(repeat=repeat,logits_equal=bool(torch.equal(first,logits)),
                        differing_route_layers=sum(a!=b for a,b in zip(routes0,routes)),route_hashes=routes,
                        **dense_metrics(first,logits)))
            finally:grouped.ROUTE_HOOK=None
            del s,logits;cleanup()
        rows.append(dict(method=name,repeats=checks));del first;cleanup()
        print(json.dumps(rows[-1]),flush=True)
    if not all(c['logits_equal'] and c['differing_route_layers']==0 for row in rows[1:] for c in row['repeats']):
        raise AssertionError('canonical/stable full-model prefill still varies')
    return dict(scope='same device/software, identical tokens and weights; no cross-platform bitwise claim',rows=rows)


def rank_layers(model,tokens):
    scores=[0.]*len(model.model.layers)
    for block in range(2):
        ids=torch.tensor([tokens[block*2048:(block+1)*2048]],device='cuda');t=ids.shape[1]
        if t!=2048:raise ValueError('insufficient training calibration data')
        with FullQwen2Session(model,t+1) as s:
            h=model.model.embed_tokens(ids)
            cos,sin=model.model.rotary_emb(h,torch.arange(t,device='cuda')[None])
            cu=torch.tensor([0,t],device='cuda',dtype=torch.int32)
            for i,layer in enumerate(model.model.layers):
                q,k,v=s.project(i,layer.input_layernorm(h),cos,sin)
                dense=flash_attn_func(q[None],k[None],v[None],causal=True)[0]
                sparse=grouped.opt_gqa_cuda(q,k,v,cu,t,dict(chunk_size=128,topk=4,**STABLE))
                d=layer.self_attn.o_proj(dense.reshape(1,t,-1))
                r=layer.self_attn.o_proj(sparse.reshape(1,t,-1))
                # Only the last quarter: enough history for genuine sparsity.
                error=(d[:,-512:].float()-r[:,-512:].float()).square().mean()
                scale=(h[:,-512:].float()+d[:,-512:].float()).square().mean().clamp_min(1e-12)
                scores[i]+=float(error/scale)/2
                h=h+d;h=h+layer.mlp(layer.post_attention_layernorm(h))
        del s,h,q,k,v,dense,sparse,d,r;cleanup()
    return scores,sorted(range(len(scores)),key=lambda i:(-scores[i],i))


def calibrate(model,train,valid):
    scores,rank=rank_layers(model,train)
    n=len(rank);budgets=sorted({round(n*f) for f in (0,1/6,1/3,1/2,2/3,5/6,1)})
    ref=score_policy(model,valid,'dense',None,blocks=2,continuation=32)
    candidates=[]
    for count in budgets:
        policy=dict(dense_layers=sorted(rank[:count]))
        result=score_policy(model,valid,'calibrated_k4',policy,blocks=2,continuation=32)
        result.update(policy=policy,dense_layer_count=count)
        for field in ('prefill','continuation'):
            result[field]['delta_nll_vs_dense']=result[field]['mean_nll']-ref[field]['mean_nll']
        result['passes_calibration']=all(result[field]['delta_nll_vs_dense']<=0.05 for field in ('prefill','continuation'))
        candidates.append(result);print(json.dumps(result),flush=True)
    accepted=next(r for r in candidates if r['passes_calibration'])
    return dict(protocol='rank on train first2x2048; select smallest dense budget on valid first2x2048 and2x32 continuation',
                threshold_delta_nll=0.05,layer_errors=scores,layer_rank=rank,budgets=budgets,
                dense_reference=ref,candidates=candidates,policy=accepted['policy'],
                sparse_layers=n-accepted['dense_layer_count'],test_data_used_for_selection=False)


def evaluate(model,tokens,policy):
    rows=[]
    for name in METHODS:
        row=score_policy(model,tokens,name,policy)
        rows.append(row);print(json.dumps(row),flush=True)
    ref=rows[0]
    for row in rows:
        for field in ('prefill','continuation'):
            row[field]['delta_nll_vs_dense']=row[field]['mean_nll']-ref[field]['mean_nll']
            row[field]['ppl_ratio_vs_dense']=row[field]['ppl']/ref[field]['ppl']
    return dict(protocol='test first4x2048, 8188 prefill targets and512 separate continuation targets; frozen policy',rows=rows)


def long_row(model,tokens,name,length,policy):
    count=3+3*8;capacity=length+count+2
    dense=session(model,'dense',capacity,policy)
    candidate=session(model,name,capacity,policy)
    try:
        ids=torch.tensor([tokens[:length]],device='cuda')
        def tail_loss(h):
            # Score the final 512 next-token predictions within the prefix.
            tail=h[:,-513:-1];target=ids[:,-512:];nll=0.
            for start in range(0,512,128):
                logits=model.lm_head(tail[:,start:start+128]).float()
                nll+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target[:,start:start+128].reshape(-1),reduction='sum'))
            return nll
        h,dense_prefill=sync_time(lambda:dense.prefill_hidden(ids));dense_nll=tail_loss(h);del h
        h,candidate_prefill=sync_time(lambda:candidate.prefill_hidden(ids));nll=tail_loss(h);del h
        _,dense_capture=sync_time(dense.prepare);_,candidate_capture=sync_time(candidate.prepare)
        samples=[];loss_a=0.;loss_b=0.
        for i,token in enumerate(tokens[length:length+count]):
            if i<3:dense.step(token);candidate.step(token);continue
            r=(i-3)//8;j=(i-3)%8
            order=('dense',name) if (r+j)%2==0 else (name,'dense')
            # Same input token, same current length. Loss is outside timing.
            out={}
            for method in order:
                out[method],elapsed=sync_time(lambda:(dense if method=='dense' else candidate).step(token))
                samples.append(dict(round=r,sample=j,method=method,order=order,wall_ms=elapsed))
            next_token=tokens[length+i+1]
            y=torch.tensor([next_token],device='cuda')
            loss_a+=float(F.cross_entropy(out['dense'][:,-1].float(),y))
            loss_b+=float(F.cross_entropy(out[name][:,-1].float(),y))
        a=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']=='dense') for r in range(3)]
        b=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']==name) for r in range(3)]
        return dict(status='ok',method=name,prefix=length,samples=samples,dense_step_ms=statistics.median(a),
            candidate_step_ms=statistics.median(b),speedup_vs_dense_graph=statistics.median(x/y for x,y in zip(a,b)),
            dense_prefill_ms=dense_prefill,candidate_prefill_ms=candidate_prefill,dense_capture_ms=dense_capture,
            candidate_capture_ms=candidate_capture,tail_targets=512,tail_nll_sum=nll,tail_ppl=math.exp(nll/512),
            dense_tail_nll_sum=dense_nll,dense_tail_ppl=math.exp(dense_nll/512),tail_delta_nll=(nll-dense_nll)/512,
            continuation_targets=24,dense_continuation_nll=loss_a,candidate_continuation_nll=loss_b)
    finally:dense.close();candidate.close()


def long_context(model,tokens,policy,lengths):
    rows=[]
    for length in lengths:
        if length+30>model.config.max_position_embeddings:raise ValueError('context exceeds native position limit')
        for name in ('stable_k4','stable_k8','calibrated_k4','dense_prefill_k4'):
            try:r=long_row(model,tokens,name,length,policy)
            except torch.OutOfMemoryError:
                r=dict(status='allocator_oom_within_approved_cap',method=name,prefix=length)
            rows.append(r);cleanup();print(json.dumps({k:v for k,v in r.items() if k!='samples'}),flush=True)
    return dict(protocol='complete model Graph paired directly with auto-split dense Graph;3warm+3x8 AB/BA;tail512 diagnostic,24 continuation targets',rows=rows)


def retrieve_one(model,tokenizer,policy,case,name):
    """Unwind all GPU references on failure before the caller tries a new case."""
    answer=tokenizer.encode(' '+case['answer'],add_special_tokens=False)
    ids=torch.tensor([case['tokens']],device='cuda');loss=0.;generated=[]
    for kind in ('score','greedy'):
        with session(model,name,case['length']+40,policy) as s:
            h=s.prefill_hidden(ids);logits=model.lm_head(h[:,-1:]);del h
            s.prepare()
            for i in range(len(answer) if kind=='score' else 16):
                token=answer[i] if kind=='score' else int(logits[:,-1].argmax(-1))
                if kind=='score':loss+=float(F.cross_entropy(logits[:,-1].float(),torch.tensor([token],device='cuda')))
                else:
                    if token==tokenizer.eos_token_id:break
                    generated.append(token)
                if i+1<(len(answer) if kind=='score' else 16):logits=s.step(token)
        del s,logits;cleanup()
    text=tokenizer.decode(generated,skip_special_tokens=True)
    found=re.search(r'(?<!\d)\d{4}(?!\d)',text)
    return dict(status='ok',answer_tokens=len(answer),answer_nll=loss,generated_text=text,
                generated_ids=generated,exact_match=bool(found and found.group()==case['answer']))


def retrieve(model,tokenizer,policy,lengths):
    rows=[]
    for case in retrieval_cases(tokenizer,lengths=lengths):
        for name in ('dense','stable_k4','calibrated_k4'):
            row={k:v for k,v in case.items() if k!='tokens'};row['method']=name
            try:row.update(retrieve_one(model,tokenizer,policy,case,name))
            except torch.OutOfMemoryError:row['status']='allocator_oom_within_approved_cap'
            rows.append(row);cleanup();print(json.dumps(row),flush=True)
    return dict(protocol='fixed synthetic passkey,lengths x25/75percent;all cases kept;16 greedy tokens;not general QA',rows=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--corpus',required=True)
    p.add_argument('--train');p.add_argument('--calibration');p.add_argument('--out',required=True)
    p.add_argument('--stage',choices=['integration','stability','calibrate','evaluate','long','retrieve'],required=True)
    p.add_argument('--lengths',default='8192,16384,32704')
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('preserve existing evidence')
    if args.stage in ('evaluate','long','retrieve') and not args.calibration:raise SystemExit('frozen calibration required')
    torch.set_num_threads(2);torch.manual_seed(20261103)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(7*1024**3/torch.cuda.get_device_properties(0).total_memory)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    def tokenize(path):return tokenizer.encode(Path(path).read_text(),add_special_tokens=False,verbose=False)
    tokens=tokenize(args.corpus)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval().requires_grad_(False)
    record=dict(stage=args.stage,model_config=model.config.to_dict(),model_config_sha256=digest(Path(args.model)/'config.json'),
        corpus_sha256=digest(args.corpus),git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),dtype='bf16',batch=1,
        torch=torch.__version__,gpu=torch.cuda.get_device_name(),tf32=False,allocator_gib=7)
    # Model configs may contain the private local checkpoint path.
    record['model_config'].pop('_name_or_path',None)
    policy=None
    if args.calibration:
        cal=json.loads(Path(args.calibration).read_text())
        if cal['model_config_sha256']!=record['model_config_sha256'] or cal['stage']!='calibrate' or cal['status']!='ok' or cal['dirty']:raise ValueError('mismatched calibration')
        if record['corpus_sha256'] in (cal['corpus_sha256'],cal.get('train_sha256')):raise ValueError('calibration/test data overlap')
        policy=cal['result']['policy'];record['calibration_sha256']=digest(args.calibration)
    with torch.inference_mode():
        if args.stage=='integration':result=integration(model,tokens)
        elif args.stage=='stability':result=stability(model,tokens)
        elif args.stage=='calibrate':
            if not args.train or digest(args.train)==record['corpus_sha256']:raise ValueError('separate train and valid data required')
            record['train_sha256']=digest(args.train);result=calibrate(model,tokenize(args.train),tokens)
        elif args.stage=='evaluate':result=evaluate(model,tokens,policy)
        elif args.stage=='retrieve':result=retrieve(model,tokenizer,policy,list(map(int,args.lengths.split(','))))
        else:result=long_context(model,tokens,policy,list(map(int,args.lengths.split(','))))
    record.update(status='ok',result=result)
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(stage=args.stage,status='ok',git_head=record['git_head'])),flush=True)


if __name__=='__main__':main()
