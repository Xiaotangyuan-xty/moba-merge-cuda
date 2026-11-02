"""Matched-contract prepared/eager/Graph measurements. GPU approval required."""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import subprocess
from datetime import datetime,timezone

import torch
from flash_attn import flash_attn_with_kvcache
from moba_opt import PagedKVCache
from ._decode_cab3 import PagedKVCache as Cab3Cache
from .benchmark_decode import paired,stat


class TrustedCab3(Cab3Cache):
    """Same structural checks; explicit finite-input contract, like prepared/FA."""
    def _tensor(self,tensor,name,*,expected_heads=None,expected_tokens=None):
        if not isinstance(tensor,torch.Tensor) or tensor.dim()!=3:raise ValueError("expected [N,H,D]")
        if tensor.device!=self.device or tensor.dtype!=self.dtype:raise ValueError("device/dtype")
        if tensor.requires_grad:raise ValueError("inference-only")
        if tensor.shape[0]<1 or tensor.shape[1]<1 or tensor.shape[2]!=self.head_dim:raise ValueError("shape")
        if tensor.stride(-1)!=1:raise ValueError("stride")
        if expected_heads is not None and tensor.shape[1]!=expected_heads:raise ValueError("heads")
        if expected_tokens is not None and tensor.shape[0]!=expected_tokens:raise ValueError("tokens")


def compare(a,b):
    assert a.positions==b.positions and a.sequence_ids==b.sequence_ids
    if a.selected_blocks is not None:
        for row in range(a.output.shape[0]):
            for h in range(a.output.shape[1]):
                x=a.selected_blocks[row,h];y=b.selected_blocks[row,h]
                assert torch.equal(torch.sort(x[x>=0]).values,torch.sort(y[y>=0]).values)
    tol=0.02 if a.output.dtype==torch.bfloat16 else 0.002
    torch.testing.assert_close(a.output,b.output,atol=tol,rtol=tol)
    torch.testing.assert_close(a.lse,b.lse,atol=5e-4,rtol=2e-5)


def sampled_fp64(old,result,q):
    """Independent Torch gather from the baseline cache; CPU FP64 softmax.

    Sample two heads per query without copying multi-GiB full caches to CPU.
    This does not reuse the prepared page table or its Attention kernel.
    """
    route=result.selected_blocks.cpu()
    errors=[]
    for row,(sid,pos) in enumerate(zip(result.sequence_ids,result.positions)):
        for head in sorted({0,q.shape[1]-1}):
            blocks=[int(x) for x in route[row,head].tolist() if x>=0]
            tokens=sorted(t for block in blocks for t in range(block*old.chunk_size,min((block+1)*old.chunk_size,pos+1)))
            index=torch.tensor(tokens,device='cuda',dtype=torch.int64)
            pages=old._table[sid,index//old.k_cache.shape[1]].long()
            kh=head//(q.shape[1]//old.num_kv_heads)
            k=old.k_cache[pages,index%old.k_cache.shape[1],kh].cpu().double()
            v=old.v_cache[pages,index%old.k_cache.shape[1],kh].cpu().double()
            logits=(k*q[row,head].cpu().double()).sum(-1)*q.shape[-1]**-0.5
            expected=torch.softmax(logits,0)@v
            actual=result.output[row,head].cpu().double()
            torch.testing.assert_close(actual,expected,atol=0.02,rtol=0.02)
            torch.testing.assert_close(result.lse[row,head].cpu().double(),torch.logsumexp(logits,0),atol=5e-4,rtol=2e-5)
            errors.append(float((actual-expected).abs().max()))
    return dict(sampled_query_heads=len(errors),max_output_abs_error=max(errors))


def fixture(cfg,args,strict):
    b,hk,d=cfg['batch'],cfg['kv_heads'],cfg['dim']
    lengths=cfg.get('lengths',[cfg['length']]*b)
    extra=args.warmup+args.rounds*args.samples+2
    dtype=torch.bfloat16 if cfg['dtype']=='bf16' else torch.float16
    gen=torch.Generator(device='cuda').manual_seed(args.seed)
    def rand(shape):return torch.randn(shape,generator=gen,device='cuda',dtype=dtype)
    common=dict(page_size=cfg['page'],chunk_size=cfg['chunk'],dtype=dtype)
    cache_type=Cab3Cache if strict else TrustedCab3
    old=cache_type(b,max(lengths)+extra,hk,d,**common)
    new=PagedKVCache(b,max(lengths)+extra,hk,d,**common)
    if cfg.get('fragmented'):
        pages=list(old._layout.free_pages);random.Random(args.seed).shuffle(pages)
        old._layout.free_pages=new._layout.free_pages=tuple(pages)
    for sid,length in enumerate(lengths):
        for lo in range(0,length,2048):
            n=min(2048,length-lo);k,v=rand((n,hk,d)),rand((n,hk,d))
            for c in (old,new):c.append(k,v,sequence_ids=[sid],cu_seqlens=[0,n])
    p=new.prepare_decode(list(range(b)),cfg['q_heads'],topk=cfg['topk'],check_finite=strict)
    q,k,v=rand((b,cfg['q_heads'],d)),rand((b,hk,d)),rand((b,hk,d))
    assert all(bool(torch.isfinite(x).all()) for x in (q,k,v))
    return old,new,p,q,k,v


def measure(cfg,args,strict,graph):
    old,new,p,q,k,v=fixture(cfg,args,strict)
    ids=list(range(cfg['batch']));topk=cfg['topk']
    if graph:p.capture('attend').capture('step')
    before=[t.data_ptr() for t in (new._table,new._lengths,p.scores,p.selected,p.output)]
    final=p.attend(q,use_graph=graph)
    compare(old.attend(q,sequence_ids=ids,topk=topk),final)
    name='prepared_graph' if graph else 'prepared_eager'
    records=paired({'cab3_eager':lambda:old.attend(q,sequence_ids=ids,topk=topk),
                    name:lambda:p.attend(q,use_graph=graph)},args.rounds,args.samples,args.warmup)
    compare(old.append_and_attend(q,k,v,sequence_ids=ids,topk=topk),p.step(q,k,v,use_graph=graph))
    steps=paired({'cab3_eager':lambda:old.append_and_attend(q,k,v,sequence_ids=ids,topk=topk),
                  name:lambda:p.step(q,k,v,use_graph=graph)},args.rounds,args.samples,args.warmup)
    final=p.attend(q,use_graph=graph)
    compare(old.attend(q,sequence_ids=ids,topk=topk),final)
    oracle=sampled_fp64(old,final,q) if cfg.get('fragmented') else None
    assert before==[t.data_ptr() for t in (new._table,new._lengths,p.scores,p.selected,p.output)]
    result={'attend':records,'step':steps,'summaries':{'attend':stat(records),'step':stat(steps)},
            'metadata_delta_uploads':p.delta_uploads,'append_calls':1+args.warmup+args.rounds*args.samples,
            'table_persistent':True,'graph_scope':'GPU compute; capacity/page planning and input copies outside'}
    result['sampled_fp64']=oracle
    # Free the second full pool before reconstructing a dense FA reference.
    # Reconstruction is outside timing and this is explicitly a distinct
    # dense algorithm/core-call baseline, with no finite-value checking.
    del old
    gc.collect();torch.cuda.empty_cache()
    lengths=cfg.get('lengths',[cfg['length']]*cfg['batch'])
    shape=(cfg['batch'],max(lengths),cfg['kv_heads'],cfg['dim'])
    dk=torch.empty(shape,device='cuda',dtype=q.dtype);dv=torch.empty_like(dk)
    for sid,length in enumerate(lengths):
        logical_k,logical_v=new.materialize(sid)
        dk[sid,:length].copy_(logical_k[:length]);dv[sid,:length].copy_(logical_v[:length])
        del logical_k,logical_v
    dl=torch.tensor(lengths,device='cuda',dtype=torch.int32)
    result['dense_reference']=paired({'dense_fa_core':lambda:flash_attn_with_kvcache(
        q[:,None],dk,dv,cache_seqlens=dl,causal=True)},args.rounds,args.samples,args.warmup)
    result['summaries']['dense_reference']=stat(result['dense_reference'])
    p.close()
    return result


def realistic():
    return json.loads((Path(__file__).resolve().parents[1]/"configs/prepared_realistic.json").read_text())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/decode_perf.json')
    p.add_argument('--realistic',action='store_true')
    p.add_argument('--ids',default='all')
    p.add_argument('--contracts',default='strict,trusted')
    p.add_argument('--modes',default='eager,graph')
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--samples',type=int,default=10)
    p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--seed',type=int,default=701)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if min(args.rounds,args.samples,args.warmup)<1:raise SystemExit('positive counts required')
    contracts=args.contracts.split(',');modes=args.modes.split(',')
    if set(contracts)-{'strict','trusted'} or set(modes)-{'eager','graph'}:raise SystemExit('invalid mode')
    limit=min(6144,int(os.environ.get('MOBA_BATCH_LIMIT_MIB','8192'))-1024)
    torch.cuda.set_per_process_memory_fraction(limit*1024**2/torch.cuda.get_device_properties(0).total_memory)
    configs=realistic() if args.realistic else json.loads(Path(args.config).read_text())
    ids=None if args.ids=='all' else set(args.ids.split(','))
    if ids and ids-{c['id'] for c in configs}:raise SystemExit('unknown IDs')
    out=Path(args.out)
    if out.exists():raise SystemExit('preserve existing output')
    out.parent.mkdir(exist_ok=True,parents=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip())
    for cfg in configs:
        if ids and cfg['id'] not in ids:continue
        for contract in contracts:
            for mode in modes:
                row=dict(config=cfg,contract=contract,mode=mode,seed=args.seed,rounds=args.rounds,
                         samples=args.samples,warmup=args.warmup,git_head=head,working_tree_dirty=dirty,
                         timestamp_utc=datetime.now(timezone.utc).isoformat(),
                         baseline='cab3c35132e2d7cf5bc4be841198485aa859e173',
                         metric='synchronized CPU wall; CUDA stream interval includes host idle')
                try:
                    with torch.inference_mode():row.update(status='ok',**measure(cfg,args,contract=='strict',mode=='graph'))
                except Exception as exc:
                    row.update(status='failed',error=repr(exc))
                    with out.open('a') as f:f.write(json.dumps(row)+'\n')
                    raise
                with out.open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps({k:row[k] for k in ('config','contract','mode','summaries')}),flush=True)
                gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
