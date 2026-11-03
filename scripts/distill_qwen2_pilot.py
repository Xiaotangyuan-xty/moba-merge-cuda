"""Guarded 0.5B fixed50%-layer sparse LoRA/KL pilot; no base checkpoint writes."""
import argparse
from contextlib import nullcontext
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('RAYON_NUM_THREADS','2')
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from safetensors.torch import save_file,load_file
from examples.qwen2_distillation import Qwen2Distillation,distillation_loss
from examples.full_qwen2 import FullQwen2Session
from scripts.evaluate_full_model import dense_metrics,assert_dense,sync_time
from scripts.evaluate_sparse_adaptation import STABLE,cleanup,digest
from scripts.calibrate_multilen import score as score_window,ids_hash
from scripts.multilen_protocol import windows,aggregate
from scripts.distill_protocol import training_plan,learning_rate,validation_score,selected_step


def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)


def tensor_hash(x):return hashlib.sha256(x.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def quality(d,tokens,recipe,split,method,enabled,dense=False):
    rows=[];layers=list(range(recipe['num_layers'])) if dense else recipe['dense_layers']
    with torch.inference_mode(),d.inference(enabled):
        for w in windows(recipe,split):
            row=score_window(d.model,tokens,w,recipe,layers);row['method']=method;rows.append(row);cleanup()
    return rows


def deltas(rows,reference,recipe):
    lookup={(r['length'],r['end']):r for r in reference}
    for r in rows:
        ref=lookup[r['length'],r['end']]
        for field in ('tail','continuation'):
            r[field+'_delta_nll']=(r[field+'_nll_sum']-ref[field+'_nll_sum'])/recipe[field+'_targets']
        r['passes_heldout_window']=all(r[k]<=recipe['heldout_delta_nll'] for k in ('tail_delta_nll','continuation_delta_nll'))
    return rows


def update(d,ids,positions,optimizer,recipe,step,do_step=True):
    optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    teacher=d.teacher_logits(ids,positions)
    student=d.student_logits(ids,positions,checkpoint_blocks=True)
    loss,parts=distillation_loss(student,teacher,ids[:,positions+1],recipe['training']['temperature'],recipe['training']['kl_weight'])
    if not bool(torch.isfinite(loss)):raise ValueError('nonfinite training loss')
    loss.backward()
    adapters=d.adapter_parameters();adapter_ids={id(p) for p in adapters}
    if any(p.grad is not None for p in d.model.parameters() if id(p) not in adapter_ids):raise AssertionError('frozen parameter acquired a gradient')
    if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in adapters):raise AssertionError('missing/nonfinite adapter gradient')
    a_norm=sum(float(p.grad.float().square().sum()) for name,p in d.model.named_parameters() if name.endswith('lora_a'))**0.5
    b_norm=sum(float(p.grad.float().square().sum()) for name,p in d.model.named_parameters() if name.endswith('lora_b'))**0.5
    if b_norm==0:raise AssertionError('adapter B has zero total gradient')
    norm=torch.nn.utils.clip_grad_norm_(adapters,recipe['training']['gradient_clip'],error_if_nonfinite=True)
    lr=learning_rate(step,recipe)
    for group in optimizer.param_groups:group['lr']=lr
    if do_step:optimizer.step()
    torch.cuda.synchronize()
    return dict(step=step,loss=float(loss.detach()),kl=float(parts['kl']),ce=float(parts['ce']),
                a_gradient_norm=a_norm,b_gradient_norm=b_norm,preclip_gradient_norm=float(norm),learning_rate=lr,
                wall_ms=(time.perf_counter()-start)*1000,peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2)


def optimizer_for(d,recipe):
    return torch.optim.AdamW(d.adapter_parameters(),lr=recipe['training']['learning_rate'],
                            weight_decay=recipe['training']['weight_decay'])


def preflight(model,tokens,recipe):
    plan=training_plan(len(tokens),recipe);probe=plan[0]
    small=torch.tensor([tokens[probe['start']:probe['start']+128]],device='cuda')
    p=torch.arange(64,device='cuda')
    with torch.no_grad():native=model.lm_head(model.model(small,use_cache=False).last_hidden_state.index_select(1,p)).float()
    d=Qwen2Distillation(model,recipe['dense_layers'],recipe['lora']['rank'],recipe['lora']['alpha'])
    initial=d.adapter_state();base0=d.base_digest();teacher0=d.teacher_logits(small,p)
    native_check=dense_metrics(native,teacher0);assert_dense(native_check)
    zero_ids=torch.tensor([tokens[probe['start']:probe['start']+1024]],device='cuda')
    with torch.no_grad():a=d(zero_ids,checkpoint_blocks=False)
    with torch.inference_mode(),d.inference(False),FullQwen2Session(model,1025,mode='moba',dense_layers=recipe['dense_layers'],**STABLE) as s:
        b=s.prefill_hidden(zero_ids)
        zero_check=dense_metrics(model.lm_head(a[:,-16:]),model.lm_head(b[:,-16:]));assert_dense(zero_check)
    del a,b,s;cleanup()
    # Compare actual adapter gradients with and without rematerialization at2K.
    sample=tokens[probe['start']:probe['end']];ids=torch.tensor([sample],device='cuda')
    positions=torch.tensor(probe['positions'],device='cuda');teacher=d.teacher_logits(ids,positions)
    outputs=[]
    for checkpoint_blocks in (False,True):
        d.zero_grad(set_to_none=True);d.route_trace={};d.record_routes=checkpoint_blocks
        logits=d.student_logits(ids,positions,checkpoint_blocks)
        loss,_=distillation_loss(logits,teacher,ids[:,positions+1]);loss.backward()
        outputs.append((logits.detach().clone(),torch.cat([p.grad.flatten().clone() for p in d.adapter_parameters()])))
    grad_l2=float(torch.linalg.vector_norm(outputs[0][1]-outputs[1][1])/torch.linalg.vector_norm(outputs[0][1]).clamp_min(1e-12))
    if grad_l2>=0.03:raise AssertionError('checkpoint adapter gradients differ')
    assert_dense(dense_metrics(outputs[0][0],outputs[1][0]))
    trace=copy.deepcopy(d.route_trace)
    if len(trace)!=12 or not all(len(v)>=2 and len(set(v))==1 for v in trace.values()):raise AssertionError('checkpoint route replay differs')
    d.record_routes=False;d.route_trace={};d.zero_grad(set_to_none=True)
    del outputs,logits,loss,teacher;cleanup()
    opt=optimizer_for(d,recipe);memory=[]
    for length in recipe['training']['lengths']:
        item=next(r for r in plan if r['length']==length)
        ids=torch.tensor([tokens[item['start']:item['end']]],device='cuda');pos=torch.tensor(item['positions'],device='cuda')
        torch.cuda.reset_peak_memory_stats();r=update(d,ids,pos,opt,recipe,1,do_step=False)
        r['length']=length;memory.append(r);d.zero_grad(set_to_none=True);cleanup()
    steps=[];ids=torch.tensor([sample],device='cuda')
    for step in range(1,4):steps.append(update(d,ids,positions,opt,recipe,step))
    if steps[0]['a_gradient_norm']!=0 or steps[1]['a_gradient_norm']==0:raise AssertionError('unexpected zero-initialized LoRA gradient evolution')
    teacher1=d.teacher_logits(small,p);base1=d.base_digest()
    if base0!=base1 or not torch.equal(teacher0,teacher1):raise AssertionError('teacher/base weights changed')
    d.load_adapters(initial)
    return dict(native_teacher_check=native_check,zero_adapter_check=zero_check,checkpoint_gradient_relative_l2=grad_l2,
                checkpoint_route_hashes=trace,memory_probes=memory,diagnostic_updates=steps,
                teacher_unchanged=True,base_digest_before=base0,base_digest_after=base1,
                diagnostic_updates_excluded_from_formal_training=True)


def save_adapter(d,folder,step,recipe_sha):
    name=f'step_{step:03d}.safetensors';path=folder/name
    if path.exists():raise ValueError('refuse checkpoint overwrite')
    tmp=folder/(name+'.tmp');save_file(d.adapter_state(),str(tmp),metadata=dict(step=str(step),recipe_sha256=recipe_sha));tmp.replace(path)
    return dict(step=step,file=name,sha256=digest(path))


def train(model,data,recipe,folder,recipe_sha,out):
    if any(folder.iterdir()):raise ValueError('fresh artifact directory required')
    d=Qwen2Distillation(model,recipe['dense_layers'],recipe['lora']['rank'],recipe['lora']['alpha'])
    count=sum(p.numel() for p in d.adapter_parameters())
    if count!=recipe['lora']['parameters']:raise AssertionError('unexpected trainable parameter count')
    base0=d.base_digest();probe=torch.tensor([data['train'][4096:4224]],device='cuda');p=torch.arange(64,device='cuda')
    teacher0=tensor_hash(d.teacher_logits(probe,p))
    dense=quality(d,data['valid'],recipe,'valid','dense',False,True)
    untrained=deltas(quality(d,data['valid'],recipe,'valid','untrained',False),dense,recipe)
    checkpoints=[];updates=[];files=[];plan=training_plan(len(data['train']),recipe)
    opt=optimizer_for(d,recipe)
    for step in range(recipe['training']['updates']+1):
        if step:
            item=plan[step-1];sample=data['train'][item['start']:item['end']]
            ids=torch.tensor([sample],device='cuda');positions=torch.tensor(item['positions'],device='cuda')
            r=update(d,ids,positions,opt,recipe,step)
            r.update(length=item['length'],start=item['start'],end=item['end'],input_sha256=ids_hash(sample),
                     positions_sha256=ids_hash(item['positions']),label_sha256=ids_hash([sample[i+1] for i in item['positions']]))
            updates.append(r);opt.zero_grad(set_to_none=True)
            if step%5==0:print(json.dumps(r),flush=True)
        if step in recipe['training']['validate_updates']:
            rows=deltas(quality(d,data['valid'],recipe,'valid','student',True),dense,recipe)
            score=validation_score(rows,recipe);checkpoints.append(dict(step=step,score=score,rows=rows))
            files.append(save_adapter(d,folder,step,recipe_sha))
            print(json.dumps(dict(validation_step=step,worst_delta_nll=score)),flush=True)
            write_json(folder/'progress.json',dict(updates=updates,checkpoints=checkpoints,files=files,recipe_sha256=recipe_sha))
    selected=selected_step(checkpoints,recipe);base1=d.base_digest();teacher1=tensor_hash(d.teacher_logits(probe,p))
    if base0!=base1 or teacher0!=teacher1:raise AssertionError('original teacher/base weights changed during training')
    return dict(trainable_parameters=count,input_tokens=sum(r['length'] for r in plan),supervised_positions=len(plan)*recipe['training']['predicted_positions'],
                optimizer_updates=len(updates),updates=updates,baseline_validation=dict(dense=dense,untrained=untrained),
                checkpoints=checkpoints,adapter_files=files,selected_step=selected,selected_by='valid worst per-window deltaNLL;tie earlier',
                selected_adapter=next(f for f in files if f['step']==selected),last_adapter=files[-1],
                base_digest_before=base0,base_digest_after=base1,teacher_logits_hash_before=teacher0,teacher_logits_hash_after=teacher1,
                fixed_sparse_layers=12,base_weights_frozen=True)


def load_adapter(d,folder,entry):
    path=folder/entry['file']
    if digest(path)!=entry['sha256']:raise ValueError('adapter checksum mismatch')
    d.load_adapters(load_file(str(path),device='cpu'))


def plain_quality(model,tokens,recipe,split,method):
    rows=[]
    with torch.inference_mode():
        for w in windows(recipe,split):
            r=score_window(model,tokens,w,recipe,recipe['dense_layers']);r['method']=method;rows.append(r);cleanup()
    return rows


def merge_check(d,merged,tokens,recipe):
    checks=[]
    with torch.inference_mode(),d.inference(True):
        for w in windows(recipe,'valid'):
            length=w['length'];ids=torch.tensor([tokens[w['start']:w['end']]],device='cuda')
            positions=torch.linspace(length-512,length-1,64,device='cuda').long()
            with FullQwen2Session(d.model,length+1,mode='moba',dense_layers=recipe['dense_layers'],**STABLE) as s:
                h=s.prefill_hidden(ids);a=d.model.lm_head(h.index_select(1,positions)).clone();del h
            del s;cleanup()
            with FullQwen2Session(merged,length+1,mode='moba',dense_layers=recipe['dense_layers'],**STABLE) as s:
                h=s.prefill_hidden(ids);b=merged.lm_head(h.index_select(1,positions));del h
                metrics=dense_metrics(a,b)
            passed=all(metrics[k]<=v for k,v in recipe['merge_gate'].items())
            checks.append(dict(length=length,end=w['end'],**metrics,passes=passed));del s,a,b;cleanup()
    return dict(rows=checks,passes_all=all(r['passes'] for r in checks),selection_data='valid only;not test')


def evaluate(model,data,recipe,training,folder):
    d=Qwen2Distillation(model,recipe['dense_layers'],recipe['lora']['rank'],recipe['lora']['alpha'])
    load_adapter(d,folder,training['selected_adapter']);merged=d.merged_model()
    gate=merge_check(d,merged,data['valid'],recipe)
    rows=quality(d,data['test'],recipe,'test','dense',False,True)
    rows+=quality(d,data['test'],recipe,'test','untrained',False)
    rows+=quality(d,data['test'],recipe,'test','selected_unmerged',True)
    rows+=plain_quality(merged,data['test'],recipe,'test','selected_merged')
    del merged;cleanup()
    load_adapter(d,folder,training['last_adapter']);last=d.merged_model()
    rows+=plain_quality(last,data['test'],recipe,'test','last_merged');del last;cleanup()
    dense=[r for r in rows if r['method']=='dense'];deltas(rows,dense,recipe)
    return dict(selected_step=training['selected_step'],merge_validation=gate,rows=rows,
                aggregates=aggregate(rows,recipe),mask_changed=False,teacher_reference='original unchanged base checkpoint')


def latency_row(base,candidate,tokens,w,recipe,method,sparse,eligible):
    conf=recipe['latency'];length=w['length'];n=conf['warmup']+conf['rounds']*conf['samples']
    kwargs=dict(mode='moba',**STABLE)
    a=FullQwen2Session(base,length+recipe['continuation_targets']+1,dense_layers=list(range(24)),backend='model_graph',**kwargs)
    b=FullQwen2Session(candidate,length+recipe['continuation_targets']+1,dense_layers=recipe['dense_layers'],backend='model_graph',**kwargs)
    try:
        ids=torch.tensor([tokens[w['start']:w['end']]],device='cuda')
        _,pa=sync_time(lambda:a.prefill_hidden(ids));_,pb=sync_time(lambda:b.prefill_hidden(ids))
        _,ca=sync_time(a.prepare);_,cb=sync_time(b.prepare);samples=[]
        for i,token in enumerate(tokens[w['end']:w['end']+n]):
            if i<conf['warmup']:a.step(token);b.step(token);continue
            r=(i-conf['warmup'])//conf['samples'];j=(i-conf['warmup'])%conf['samples']
            order=['dense',method] if (r+j)%2==0 else [method,'dense']
            for name in order:
                _,ms=sync_time(lambda:(a if name=='dense' else b).step(token))
                samples.append(dict(round=r,sample=j,method=name,order=order,wall_ms=ms))
        aa=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']=='dense') for r in range(conf['rounds'])]
        bb=[statistics.median(s['wall_ms'] for s in samples if s['round']==r and s['method']==method) for r in range(conf['rounds'])]
        return dict(length=length,end=w['end'],method=method,input_sha256=ids_hash(tokens[w['start']:w['end']]),sparse_layers=sparse,
                    merge_eligible=eligible,samples=samples,dense_step_ms=statistics.median(aa),candidate_step_ms=statistics.median(bb),
                    paired_speedup=statistics.median(x/y for x,y in zip(aa,bb)),dense_prefill_ms=pa,candidate_prefill_ms=pb,dense_capture_ms=ca,candidate_capture_ms=cb)
    finally:a.close();b.close()


def latency(model,tokens,recipe,training,folder,evaluation):
    d=Qwen2Distillation(model,recipe['dense_layers'],recipe['lora']['rank'],recipe['lora']['alpha'])
    load_adapter(d,folder,training['selected_adapter']);merged=d.merged_model();rows=[]
    # Separate base model clone keeps Teacher projections disabled while Student
    # adapters are active. All copies stay on the one explicitly approved card.
    with d.inference(False):base=copy.deepcopy(d.model)
    for method in ('untrained','selected_unmerged','selected_merged'):
        enabled=method=='selected_unmerged'
        with d.inference(enabled),torch.inference_mode():
            candidate=merged if method=='selected_merged' else d.model
            for w in windows(recipe,'test'):
                if w['end']!=recipe['target_boundaries']['test'][0]:continue
                r=latency_row(base,candidate,tokens,w,recipe,method,12,
                              evaluation['merge_validation']['passes_all'] if method=='selected_merged' else True)
                rows.append(r);cleanup();print(json.dumps({k:v for k,v in r.items() if k!='samples'}),flush=True)
    return dict(rows=rows,scope='complete Graph vs original dense Graph;unmerged overhead included;failed merge gate rows diagnostic only')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data-dir',required=True)
    p.add_argument('--recipe',default='configs/distill_pilot_20261103.json');p.add_argument('--stage',choices=['preflight','train','evaluate','latency'],required=True)
    p.add_argument('--artifacts',required=True);p.add_argument('--training-result');p.add_argument('--evaluation-result');p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise SystemExit('preserve existing evidence')
    recipe=json.loads(Path(args.recipe).read_text());recipe_sha=digest(args.recipe)
    uuid=os.environ.get('MOBA_COVERAGE_GPU_UUID');index=int(os.environ.get('MOBA_APPROVED_GPU_INDEX','1'))
    if index not in recipe['resource']['approved_physical_gpus'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid or not uuid:raise SystemExit('approved masked card required')
    actual=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=index','--format=csv,noheader,nounits'],text=True).strip()
    if actual!=str(index):raise SystemExit('GPU index/UUID mismatch')
    if digest(Path(args.model)/'config.json')!=recipe['model_config_sha256']:raise ValueError('wrong model config')
    if len(recipe['dense_layers'])*2!=recipe['num_layers']:raise ValueError('fixed50% layer policy required')
    folder=Path(args.artifacts);folder.mkdir(exist_ok=True,parents=True)
    train_record=None;evaluation=None
    if args.stage in ('evaluate','latency'):
        train_record=json.loads(Path(args.training_result).read_text())
        if train_record['stage']!='train' or train_record['status']!='ok' or train_record['dirty'] or train_record['recipe_sha256']!=recipe_sha:raise ValueError('training provenance mismatch')
        if selected_step(train_record['result']['checkpoints'],recipe)!=train_record['result']['selected_step']:raise ValueError('checkpoint selection mismatch')
    if args.stage=='latency':
        ev=json.loads(Path(args.evaluation_result).read_text())
        if ev['status']!='ok' or ev['stage']!='evaluate' or ev['training_result_sha256']!=digest(args.training_result):raise ValueError('evaluation provenance mismatch')
        evaluation=ev['result']
    torch.set_num_threads(2);torch.manual_seed(recipe['seed']);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(recipe['resource']['allocator_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    splits={'preflight':['train'],'train':['train','valid'],'evaluate':['valid','test'],'latency':['test']}[args.stage]
    data={};hashes={}
    for split in splits:
        path=Path(args.data_dir)/f'wikitext2_{split}.txt';sha=digest(path)
        if sha!=recipe['corpus'][split]['sha256']:raise ValueError('corpus hash mismatch')
        data[split]=tok.encode(path.read_text(),add_special_tokens=False,verbose=False);hashes[split]=sha
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True,trust_remote_code=False).cuda().eval().requires_grad_(False)
    record=dict(stage=args.stage,recipe_sha256=recipe_sha,model_config_sha256=recipe['model_config_sha256'],model_weight_sha256=recipe['model_weight_sha256'],
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
                physical_gpu=index,gpu=torch.cuda.get_device_name(),torch=torch.__version__,corpus_sha256=hashes,seed=recipe['seed'],dtype='bf16',tf32=False)
    if args.training_result:record['training_result_sha256']=digest(args.training_result)
    if args.evaluation_result:record['evaluation_result_sha256']=digest(args.evaluation_result)
    if args.stage=='preflight':result=preflight(model,data['train'],recipe)
    elif args.stage=='train':result=train(model,data,recipe,folder,recipe_sha,out)
    elif args.stage=='evaluate':result=evaluate(model,data,recipe,train_record['result'],folder)
    else:result=latency(model,data['test'],recipe,train_record['result'],folder,evaluation)
    record.update(status='ok',result=result);write_json(out,record)
    print(json.dumps(dict(stage=args.stage,status='ok')),flush=True)


if __name__=='__main__':main()
