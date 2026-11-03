"""Full-model correctness, held-out quality, latency and profiling.

Existing dependencies only; local safetensors checkpoint, batch=1 BF16.
GPU tasks must run through the shared GPU batch guard.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import subprocess
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('RAYON_NUM_THREADS','2')
import torch
import transformers
import flash_attn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from examples.full_qwen2 import FullQwen2Session
from examples.retrieval_cases import retrieval_cases


def sync_time(fn):
    torch.cuda.synchronize();t=time.perf_counter();result=fn();torch.cuda.synchronize()
    return result,(time.perf_counter()-t)*1000


def dense_metrics(ref,out):
    a,b=ref.float(),out.float()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):raise ValueError('nonfinite logits')
    la,lb=F.log_softmax(a,-1),F.log_softmax(b,-1)
    return dict(max_abs=float((a-b).abs().max()),relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-8)),
                kl=float((la.exp()*(la-lb)).sum(-1).mean()),top1_agreement=float((a.argmax(-1)==b.argmax(-1)).float().mean()))


def assert_dense(record):
    if record['relative_l2']>0.02 or record['kl']>0.002 or record['max_abs']>0.5:
        raise AssertionError('dense integration tolerance exceeded: '+str(record))


class Native:
    def __init__(self,model,backend='flash_attention_2'):
        self.model=model;self.model.config._attn_implementation=backend
        self.input=torch.zeros(1,1,device='cuda',dtype=torch.long)
        self.past=None
    def prefill(self,ids):
        r=self.model(ids,use_cache=True,logits_to_keep=1);self.past=r.past_key_values
        return r.logits
    def prepare(self):pass
    def step(self,token):
        self.input.fill_(token)
        r=self.model(self.input,past_key_values=self.past,use_cache=True,logits_to_keep=1)
        self.past=r.past_key_values;return r.logits
    def close(self):self.past=None


def make_session(model,name,capacity):
    if name=='hf_fa2':return Native(model)
    if not re.fullmatch(r'(dense|moba_k[1-9])_((packed_)?(eager|attngraph|modelgraph)|(eager|attngraph|modelgraph)_packed)',name):
        raise ValueError('unknown method: '+name)
    mode='moba' if name.startswith('moba') else 'dense'
    topk=int(re.search(r'k(\d+)',name).group(1)) if mode=='moba' else 4
    backend='model_graph' if 'modelgraph' in name else 'attention_graph' if 'attngraph' in name else 'eager'
    return FullQwen2Session(model,capacity,mode=mode,chunk=128,topk=topk,backend=backend,fused_qkv='packed' in name)


def prefill(session,ids):
    if isinstance(session,Native):return session.prefill(ids)
    h=session.prefill_hidden(ids)
    return session.model.lm_head(h[:,-1:])


def validate(model,tokens):
    ids=torch.tensor([tokens[:128]],device='cuda')
    continuation=tokens[128:140]
    references={}
    for backend in ('sdpa','flash_attention_2'):
        reference=Native(model,backend)
        ref=[reference.prefill(ids).detach().clone()]
        for token in continuation:ref.append(reference.step(token).detach().clone())
        references[backend]=ref;reference.close()
    envelope=[dense_metrics(a,b) for a,b in zip(references['sdpa'],references['flash_attention_2'])]
    print(json.dumps({'native_sdpa_vs_fa2':envelope}),flush=True)
    ref=references['flash_attention_2']
    rows=[]
    for name in ('hf_fa2','dense_eager','dense_modelgraph','dense_modelgraph_packed','moba_k8_modelgraph'):
        s=make_session(model,name,160)
        try:
            first=prefill(s,ids);initial=dense_metrics(ref[0],first)
            print(json.dumps({'validation_method':name,'position':127,**initial}),flush=True);assert_dense(initial)
            lengths=[c.lengths for c in s.caches] if isinstance(s,FullQwen2Session) else None
            s.prepare()
            if lengths is not None:assert lengths==[c.lengths for c in s.caches]
            checks=[initial]
            for i,token in enumerate(continuation):
                r=dense_metrics(ref[i+1],s.step(token))
                print(json.dumps({'validation_method':name,'position':128+i,**r}),flush=True);assert_dense(r);checks.append(r)
            if lengths is not None:assert all(c.lengths==(140,) for c in s.caches)
            rows.append(dict(method=name,checks=checks,capture_preserves_visible_lengths=True))
        finally:s.close()
        del s;gc.collect();torch.cuda.empty_cache()
    sparse_graph=[]
    for packed in (False,True):
        suffix='_packed' if packed else ''
        eager=make_session(model,'moba_k4_eager'+suffix,1060)
        graph=make_session(model,'moba_k4_modelgraph'+suffix,1060)
        try:
            prefix=torch.tensor([tokens[:1023]],device='cuda')
            initial=dense_metrics(prefill(eager,prefix),prefill(graph,prefix));assert_dense(initial)
            eager.prepare();graph.prepare()
            assert all(c.lengths==(1023,) for c in graph.caches)
            checks=[initial]
            # Cross both a page and a chunk boundary in the actual sparse path.
            for token in tokens[1023:1043]:
                r=dense_metrics(eager.step(token),graph.step(token));assert_dense(r);checks.append(r)
                assert all(torch.equal(a.selected,b.selected) for a,b in zip(eager.plans,graph.plans))
                assert [c.lengths for c in eager.caches]==[c.lengths for c in graph.caches]
            sparse_graph.append(dict(packed_qkv=packed,prefix=1023,steps=20,checks=checks,
                                     selected_blocks_equal=True,capture_preserves_visible_lengths=True))
            print(json.dumps({'sparse_graph_packed':packed,'max_relative_l2':max(r['relative_l2'] for r in checks)}),flush=True)
        finally:eager.close();graph.close()
        del eager,graph;gc.collect();torch.cuda.empty_cache()
    return dict(scope='pretrained full model; fixed teacher-forced dense/all-selected and sparse Graph comparisons',
                primary_reference='native HF FA2',native_sdpa_vs_fa2=envelope,
                tolerance=dict(relative_l2=0.02,kl=0.002,max_abs=0.5),prefix=128,steps=12,methods=rows,
                sparse_graph=sparse_graph)


def head_nll(model,h,ids):
    total=0.;n=ids.shape[1]-1
    for start in range(0,n,128):
        end=min(n,start+128)
        logits=model.lm_head(h[:,start:end]).float()
        total+=float(F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,start+1:end+1].reshape(-1),reduction='sum'))
    if not math.isfinite(total):raise ValueError('nonfinite NLL')
    return total,n


def quality(model,tokens,args):
    methods=('hf_sdpa','hf_fa2','dense_eager','moba_k2_eager','moba_k4_eager','moba_k8_eager','moba_k4_packed_eager')
    rows=[]
    for offset in range(0,args.quality_blocks*args.quality_length,args.quality_length):
        ids=torch.tensor([tokens[offset:offset+args.quality_length]],device='cuda')
        if ids.shape[1]!=args.quality_length:raise ValueError('insufficient corpus')
        for method in methods:
            session=None
            try:
                if method in ('hf_sdpa','hf_fa2'):
                    model.config._attn_implementation='sdpa' if method=='hf_sdpa' else 'flash_attention_2'
                    h=model.model(ids,use_cache=False).last_hidden_state
                else:
                    session=make_session(model,method,args.quality_length+1)
                    h=session.prefill_hidden(ids)
                nll,n=head_nll(model,h,ids)
                rows.append(dict(method=method,offset=offset,tokens=n,nll_sum=nll,mean_nll=nll/n,ppl=math.exp(nll/n)))
                print(json.dumps(rows[-1]),flush=True)
                del h
            finally:
                if session is not None:session.close()
            del session;gc.collect();torch.cuda.empty_cache()
    # A separate cache-continuation score, with the exact same target tokens.
    continuation=[]
    for block in range(args.quality_blocks):
        window=tokens[block*args.quality_length:(block+1)*args.quality_length]
        prefix=window[:-32];targets=window[-32:]
        for method in methods:
            s=Native(model,'sdpa' if method=='hf_sdpa' else 'flash_attention_2') if method in ('hf_sdpa','hf_fa2') else make_session(model,method,len(window)+1)
            try:
                logits=prefill(s,torch.tensor([prefix],device='cuda'));s.prepare();loss=0.
                for i,token in enumerate(targets):
                    loss+=float(F.cross_entropy(logits[:,-1].float(),torch.tensor([token],device='cuda'),reduction='sum'))
                    if i+1<len(targets):logits=s.step(token)
                if not math.isfinite(loss):raise ValueError('nonfinite continuation loss')
                continuation.append(dict(method=method,block=block,tokens=32,nll_sum=loss,mean_nll=loss/32,ppl=math.exp(loss/32)))
            finally:s.close()
            del s;gc.collect();torch.cuda.empty_cache()
    return dict(dataset='WikiText-2 tokenized test fixed prefix blocks',offset_policy='first consecutive blocks, no selection',
                blocks=args.quality_blocks,block_length=args.quality_length,rows=rows,continuation=continuation)


def paired_latency(model,tokens,args):
    rows=[]
    candidates=args.methods.split(',')
    for length in map(int,args.lengths.split(',')):
        count=args.warmup+args.rounds*args.samples
        for method in candidates:
            native,base_init=sync_time(lambda:Native(model))
            candidate,candidate_init=sync_time(lambda:make_session(model,method,length+count+2))
            ids=torch.tensor([tokens[:length]],device='cuda')
            try:
                _,base_prefill=sync_time(lambda:prefill(native,ids))
                _,candidate_prefill=sync_time(lambda:prefill(candidate,ids))
                _,setup=sync_time(candidate.prepare)
                samples=[];first_decode={}
                for i,token in enumerate(tokens[length:length+count]):
                    if i<args.warmup:
                        if i==0:
                            _,first_decode['hf_fa2']=sync_time(lambda:native.step(token))
                            _,first_decode[method]=sync_time(lambda:candidate.step(token))
                        else:native.step(token);candidate.step(token)
                        continue
                    r=(i-args.warmup)//args.samples;s=(i-args.warmup)%args.samples
                    order=('hf_fa2',method) if (r+s)%2==0 else (method,'hf_fa2')
                    for name in order:
                        _,elapsed=sync_time(lambda:(native if name=='hf_fa2' else candidate).step(token))
                        samples.append(dict(round=r,sample=s,impl=name,order=order,wall_ms=elapsed))
                base_m=[statistics.median(x['wall_ms'] for x in samples if x['impl']=='hf_fa2' and x['round']==r) for r in range(args.rounds)]
                cand_m=[statistics.median(x['wall_ms'] for x in samples if x['impl']==method and x['round']==r) for r in range(args.rounds)]
                rows.append(dict(method=method,prefix=length,samples=samples,hf_prefill_ms=base_prefill,
                    candidate_prefill_ms=candidate_prefill,prepare_capture_ms=setup,
                    hf_init_ms=base_init,candidate_init_ms=candidate_init,first_decode_ms=first_decode,
                    hf_first_token_ms=base_init+base_prefill,candidate_first_token_ms=candidate_init+candidate_prefill,
                    semantic_dense_fastpath=getattr(candidate,'all_selected',False),
                    attention_backend='fa2' if getattr(candidate,'effective_dense',True) else 'moba_triton',
                    dense_num_splits=getattr(candidate,'dense_num_splits',None),
                    hf_step_ms=statistics.median(base_m),candidate_step_ms=statistics.median(cand_m),
                    paired_speedup=statistics.median(a/b for a,b in zip(base_m,cand_m))))
                print(json.dumps({k:v for k,v in rows[-1].items() if k!='samples'}),flush=True)
            finally:native.close();candidate.close()
            del native,candidate;gc.collect();torch.cuda.empty_cache()
    return dict(scope='complete model forward, batch1 teacher-forced, CPU management and token staging included; no sampling',
                rounds=args.rounds,samples_per_round=args.samples,warmup=args.warmup,rows=rows)


def profile_model(model,tokens,args,out):
    result=[]
    for method in ('moba_k4_eager','moba_k4_attngraph','moba_k4_modelgraph','moba_k4_modelgraph_packed'):
        s=make_session(model,method,1100)
        try:
            prefill(s,torch.tensor([tokens[:1024]],device='cuda'));s.prepare()
            for token in tokens[1024:1028]:s.step(token)
            torch.cuda.synchronize()
            s.profile=method.endswith('_eager')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
                for token in tokens[1028:1031]:s.step(token)
                torch.cuda.synchronize()
            path=out.parent/(out.stem+'_'+method+'_trace.json')
            if path.exists():raise ValueError('preserve trace')
            p.export_chrome_trace(str(path))
            events=json.loads(path.read_text())['traceEvents']
            kernels=[e for e in events if e.get('cat')=='kernel']
            launches=[e for e in events if e.get('cat')=='cuda_runtime' and 'Launch' in e.get('name','')]
            result.append(dict(method=method,steps=3,kernel_count=len(kernels),kernel_total_us=sum(e.get('dur',0) for e in kernels),
                host_launch_count=len(launches),labels=[dict(name=x.key,count=x.count,cpu_us=x.cpu_time_total) for x in p.key_averages() if x.key.startswith('model/') and x.cpu_time_total>0],
                trace=path.name))
        finally:s.close()
        del s;gc.collect();torch.cuda.empty_cache()
    return dict(scope='diagnostic profiler; does not supply formal latency',rows=result)


def retrieval(model,tokenizer):
    cases=retrieval_cases(tokenizer)
    rows=[]
    for case in cases:
        prefix=torch.tensor([case['tokens']],device='cuda')
        answer=tokenizer.encode(' '+case['answer'],add_special_tokens=False)
        for method in ('hf_sdpa','dense_eager','moba_k2_eager','moba_k4_eager','moba_k8_eager'):
            def session():
                return Native(model,'sdpa') if method=='hf_sdpa' else make_session(model,method,case['length']+40)
            s=session()
            try:
                logits=prefill(s,prefix);s.prepare();loss=0.
                for i,token in enumerate(answer):
                    loss+=float(F.cross_entropy(logits[:,-1].float(),torch.tensor([token],device='cuda'),reduction='sum'))
                    if i+1<len(answer):logits=s.step(token)
            finally:s.close()
            del s;gc.collect();torch.cuda.empty_cache()
            s=session();generated=[]
            try:
                logits=prefill(s,prefix);s.prepare()
                for i in range(16):
                    token=int(logits[:,-1].argmax(-1))
                    if token==tokenizer.eos_token_id:break
                    generated.append(token)
                    if i<15:logits=s.step(token)
            finally:s.close()
            text=tokenizer.decode(generated,skip_special_tokens=True)
            found=re.search(r'(?<!\d)\d{4}(?!\d)',text)
            row={k:v for k,v in case.items() if k!='tokens'}
            row.update(method=method,answer_tokens=len(answer),answer_nll=loss,answer_mean_nll=loss/len(answer),
                generated_ids=generated,generated_text=text,exact_match=bool(found and found.group()==case['answer']))
            rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
            del s;gc.collect();torch.cuda.empty_cache()
    return dict(scope='six synthetic passkey diagnostics on a base model; not a general QA benchmark',
                seed=20261102,max_new_tokens=16,methods=5,cases=6,rows=rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--corpus',required=True)
    p.add_argument('--stage',choices=['validate','quality','latency','profile','retrieval'],required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--quality-length',type=int,default=2048);p.add_argument('--quality-blocks',type=int,default=4)
    p.add_argument('--lengths',default='256,2048,8192')
    p.add_argument('--methods',default='dense_eager,dense_modelgraph,moba_k2_modelgraph,moba_k4_modelgraph,moba_k8_modelgraph,moba_k4_modelgraph_packed')
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--samples',type=int,default=10);p.add_argument('--warmup',type=int,default=5)
    args=p.parse_args();out=Path(args.out)
    if min(args.rounds,args.samples,args.warmup,args.quality_blocks)<1 or args.quality_length<2:
        raise SystemExit('positive counts and quality length >=2 required')
    requested=args.methods.split(',')
    if len(requested)!=len(set(requested)) or 'hf_fa2' in requested:raise SystemExit('duplicate/reference candidate')
    if out.exists():raise SystemExit('preserve existing evidence')
    out.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
    torch.manual_seed(20261102)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    raw=Path(args.corpus).read_bytes();tokens=tokenizer.encode(raw.decode(),add_special_tokens=False,verbose=False)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,trust_remote_code=False,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval().requires_grad_(False)
    record=dict(stage=args.stage,model='Qwen/Qwen2.5-0.5B',dtype='bf16',batch=1,corpus_sha256=hashlib.sha256(raw).hexdigest(),
        model_revision='060db6499f32faf8b98477b0a26969ef7d8b9987',
        model_weight_sha256='88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342',
        model_config_sha256=hashlib.sha256((Path(args.model)/'config.json').read_bytes()).hexdigest(),
        finite_contract='finite internal QKV assumed by both HF and custom model integration',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),torch=torch.__version__,
        transformers=transformers.__version__,flash_attn=flash_attn.__version__,gpu=torch.cuda.get_device_name())
    with torch.inference_mode():
        if args.stage=='validate':record['result']=validate(model,tokens)
        elif args.stage=='quality':record['result']=quality(model,tokens,args)
        elif args.stage=='latency':record['result']=paired_latency(model,tokens,args)
        elif args.stage=='retrieval':record['result']=retrieval(model,tokenizer)
        else:record['result']=profile_model(model,tokens,args,out)
    record['status']='ok';out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='result'}),flush=True)


if __name__=='__main__':main()
