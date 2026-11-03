"""Read-only checkpoint diagnosis and expanded validation; never loads test text."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('RAYON_NUM_THREADS','2')
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from examples.qwen2_distillation import Qwen2Distillation
from examples.qwen2_merge_diagnostic import ProjectionAblation,RouteTape,MODES,diagnostic_forward
from scripts.distill_qwen2_pilot import load_adapter,quality,deltas,write_json
from scripts.evaluate_full_model import dense_metrics
from scripts.calibrate_multilen import ids_hash
from scripts.multilen_protocol import windows,aggregate


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def diagnose(d,data,recipe,pilot,training,folder):
    rows=[];projections=[];routes=[];weights=[];controls=[]
    for step in recipe['diagnostic_checkpoints']:
        entry=next(f for f in training['adapter_files'] if f['step']==step);load_adapter(d,folder,entry)
        ablation=ProjectionAblation(d);weights.extend(dict(step=step,**r) for r in ablation.weight_rows)
        for w in windows(recipe,'valid'):
            if w['end'] not in recipe['diagnostic_boundaries']:continue
            sample=data[w['start']:w['end']];ids=torch.tensor([sample],device='cuda')
            pos=torch.linspace(w['length']-512,w['length']-1,recipe['logit_positions'],device='cuda').long()
            tape=RouteTape();local=[]
            ref=diagnostic_forward(ablation,ids,pos,tape=tape,projection_rows=local)
            direct=d.student_logits(ids,pos,checkpoint_blocks=False)
            if not torch.equal(ref,direct):raise AssertionError('diagnostic baseline changed original forward')
            controls.append(dict(step=step,length=w['length'],end=w['end'],baseline_exact=True))
            del direct
            projections.extend(dict(step=step,length=w['length'],end=w['end'],**r) for r in local)
            for mode in MODES:
                for replay in (False,True):
                    tape.rows=[];out=diagnostic_forward(ablation,ids,pos,mode,tape,replay)
                    metrics=dense_metrics(ref,out)
                    passed=all(metrics[k]<=v for k,v in pilot['merge_gate'].items())
                    rows.append(dict(step=step,length=w['length'],end=w['end'],mode=mode,replay=replay,
                        input_sha256=ids_hash(sample),passes_original_gate=passed,**metrics))
                    routes.extend(dict(step=step,length=w['length'],end=w['end'],mode=mode,**r) for r in tape.rows)
                    del out
            print(json.dumps(dict(stage='diagnose',step=step,length=w['length'],end=w['end'],status='ok')),flush=True)
            del ref,tape,ids;gc.collect();torch.cuda.empty_cache()
        del ablation;gc.collect();torch.cuda.empty_cache()
    return dict(rows=rows,projections=projections,routes=routes,weights=weights,controls=controls,
                scope='same-window prefill logits;fixed routes isolate one intervention;no production route override')


def expanded(d,data,recipe,pilot,training,folder):
    # Reuse the pilot's immutable model/layer contract with the new windows.
    recipe=dict(pilot,**recipe)
    rr=quality(d,data,recipe,'valid','dense',False,True)
    dense=rr.copy()
    for step in recipe['expanded_checkpoints']:
        load_adapter(d,folder,next(f for f in training['adapter_files'] if f['step']==step))
        rows=quality(d,data,recipe,'valid',f'step_{step:03d}',True)
        rr+=rows
        print(json.dumps(dict(stage='expanded',step=step,windows=len(rows),status='ok')),flush=True)
    deltas(rr,dense,recipe)
    stats=[]
    for method in dict.fromkeys(r['method'] for r in rr):
        rows=[r for r in rr if r['method']==method]
        values=sorted(r[k] for r in rows for k in ('tail_delta_nll','continuation_delta_nll'))
        stats.append(dict(method=method,windows=len(rows),passed=sum(r['passes_heldout_window'] for r in rows),
            mean_delta_nll=sum(values)/len(values),worst_delta_nll=max(values),p90_delta_nll=values[__import__('math').ceil(.9*len(values))-1]))
    return dict(rows=rr,aggregates=aggregate(rr,recipe),checkpoint_statistics=stats,
                original_selected_step_unchanged=training['selected_step'],scope='retrospective expanded valid only;not a new test;no revised pilot checkpoint')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data-dir',required=True)
    p.add_argument('--artifacts',required=True);p.add_argument('--stage',choices=['diagnose','expanded'],required=True)
    p.add_argument('--recipe',default='configs/merge_diagnosis_20261103.json');p.add_argument('--out',required=True);args=p.parse_args()
    if Path(args.out).exists():raise SystemExit('preserve evidence')
    recipe=json.loads(Path(args.recipe).read_text());pilot=json.loads(Path(recipe['pilot_recipe']).read_text())
    train_path=Path(recipe['training_result']);record=json.loads(train_path.read_text());training=record['result']
    if record['recipe_sha256']!=digest(recipe['pilot_recipe']) or record['status']!='ok':raise ValueError('pilot provenance mismatch')
    index=int(os.environ.get('MOBA_APPROVED_GPU_INDEX','1'));uuid=os.environ.get('MOBA_COVERAGE_GPU_UUID')
    if not uuid or os.environ.get('CUDA_VISIBLE_DEVICES')!=uuid or index not in recipe['resource']['approved_physical_gpus']:raise ValueError('masked approved GPU required')
    actual=subprocess.check_output(['nvidia-smi','--id='+uuid,'--query-gpu=index','--format=csv,noheader,nounits'],text=True).strip()
    if actual!=str(index):raise ValueError('GPU index differs')
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    if dirty:raise ValueError('clean committed source required')
    if digest(Path(args.model)/'config.json')!=pilot['model_config_sha256'] or digest(Path(args.model)/'model.safetensors')!=pilot['model_weight_sha256']:raise ValueError('model changed')
    torch.set_num_threads(2);torch.manual_seed(pilot['seed']);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(recipe['resource']['allocator_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
    path=Path(args.data_dir)/'wikitext2_valid.txt'
    if digest(path)!=pilot['corpus']['valid']['sha256']:raise ValueError('valid corpus changed')
    tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    data=tok.encode(path.read_text(),add_special_tokens=False,verbose=False)
    if max(w['continuation_end'] for w in windows(recipe,'valid'))>len(data):raise ValueError('validation windows out of range')
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True,trust_remote_code=False).cuda().eval().requires_grad_(False)
    d=Qwen2Distillation(model,pilot['dense_layers'],pilot['lora']['rank'],pilot['lora']['alpha'])
    before=d.base_digest();start=time.perf_counter()
    with torch.inference_mode(),d.inference(True):
        result=(diagnose if args.stage=='diagnose' else expanded)(d,data,recipe,pilot,training,Path(args.artifacts))
    after=d.base_digest()
    if before!=after:raise AssertionError('base mutated')
    write_json(args.out,dict(stage=args.stage,status='ok',git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=False,
        recipe_sha256=digest(args.recipe),pilot_recipe_sha256=digest(recipe['pilot_recipe']),training_result_sha256=digest(train_path),
        physical_gpu=index,gpu=torch.cuda.get_device_name(),torch=torch.__version__,dtype='bf16',tf32=False,
        valid_corpus_sha256=digest(path),valid_token_count=len(data),base_unchanged=True,base_digest=after,
        elapsed_seconds=time.perf_counter()-start,result=result))


if __name__=='__main__':main()
