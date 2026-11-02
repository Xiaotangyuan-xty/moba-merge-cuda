"""Local pretrained first-attention-layer evidence; not full-model generation.

Read only selected tensors from existing safetensors shards. GPU1 approval is
required. Checkpoint path belongs in private env.local.sh, never this source.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch
from safetensors import safe_open
from transformers import AutoTokenizer,Qwen2Config,DynamicCache
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention,Qwen2RMSNorm,Qwen2RotaryEmbedding
from examples.qwen2_attention import project_qkv,output_projection,decode_layer
from moba_opt import PagedKVCache
from benchmarks._decode_cab3 import PagedKVCache as Cab3Cache
from reference.decode_reference import decode_reference


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--model-label',default='DeepSeek-R1-Distill-Qwen-7B')
    ap.add_argument('--prefix',type=int,default=64)
    ap.add_argument('--steps',type=int,default=24)
    ap.add_argument('--out',required=True)
    args=ap.parse_args()
    out=Path(args.out)
    if out.exists():raise SystemExit('preserve existing evidence')
    root=Path(args.checkpoint)
    config=Qwen2Config.from_pretrained(root,local_files_only=True)
    config._attn_implementation='eager'
    weight_map=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    tokenizer=AutoTokenizer.from_pretrained(root,local_files_only=True,trust_remote_code=False)
    text='请解释稀疏注意力、KV缓存和分组查询注意力如何协作，并分析长上下文推理中的延迟。'
    tokens=tokenizer.encode(text*12,add_special_tokens=True)[:args.prefix+args.steps]
    if len(tokens)!=args.prefix+args.steps:raise ValueError('not enough teacher-forced tokens')
    hashes={};read_bytes=0
    def read(key):
        nonlocal read_bytes
        with safe_open(root/weight_map[key],framework='pt',device='cpu') as f:t=f.get_tensor(key)
        hashes[key]=hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
        read_bytes+=t.numel()*t.element_size()
        return t
    layer=Qwen2Attention(config,layer_idx=0).to(dtype=torch.bfloat16)
    state={k:read('model.layers.0.self_attn.'+k) for k in layer.state_dict()}
    layer.load_state_dict(state);del state
    norm=Qwen2RMSNorm(config.hidden_size,eps=config.rms_norm_eps).to(dtype=torch.bfloat16)
    norm.weight.data.copy_(read('model.layers.0.input_layernorm.weight'))
    key='model.embed_tokens.weight'
    with safe_open(root/weight_map[key],framework='pt',device='cpu') as f:
        slices=f.get_slice(key)
        embeddings=torch.cat([slices[t:t+1] for t in tokens],dim=0)
    read_bytes+=embeddings.numel()*embeddings.element_size()
    hashes['selected_token_embeddings']=hashlib.sha256(embeddings.view(torch.uint8).numpy().tobytes()).hexdigest()
    layer=layer.cuda().eval().requires_grad_(False)
    norm=norm.cuda().eval().requires_grad_(False)
    hidden=embeddings.cuda()[None]
    rope=Qwen2RotaryEmbedding(config,device='cuda')
    hq,hk,d=config.num_attention_heads,config.num_key_value_heads,layer.head_dim
    cfg=dict(batch_size=1,max_seq_len=len(tokens)+1,num_kv_heads=hk,head_dim=d,
             page_size=7,chunk_size=8,dtype=torch.bfloat16)
    sparse=PagedKVCache(**cfg);old=Cab3Cache(**cfg);dense=PagedKVCache(**cfg)
    checks=[];timings=[];sparse_dense=[]
    with torch.inference_mode():
        cos,sin=rope(hidden,torch.arange(len(tokens),device='cuda')[None])
        prefix=norm(hidden[:,:args.prefix])
        q,k,v=project_qkv(layer,prefix,cos[:,:args.prefix],sin[:,:args.prefix])
        for c in (sparse,old,dense):c.append(k,v,sequence_ids=[0],cu_seqlens=[0,args.prefix])
        logical_k,logical_v=[k.cpu()],[v.cpu()]
        hf_cache=DynamicCache()
        causal=torch.full((args.prefix,args.prefix),torch.finfo(torch.bfloat16).min,device='cuda').triu(1)[None,None]
        hf_eager=layer(prefix,position_embeddings=(cos[:,:args.prefix],sin[:,:args.prefix]),
                       attention_mask=causal,past_key_value=DynamicCache())[0]
        # HF eager materializes BF16 QK logits; SDPA/FA and our kernel retain
        # FP32 score accumulation. Keep eager drift as diagnostic evidence and
        # independently check the raw Attention result against CPU FP64.
        config._attn_implementation='sdpa'
        hf_prefill=layer(prefix,position_embeddings=(cos[:,:args.prefix],sin[:,:args.prefix]),
                         attention_mask=causal,past_key_value=hf_cache)[0]
        p=sparse.prepare_decode([0],hq,topk=4,check_finite=True)
        pd=dense.prepare_decode([0],hq,mode='dense',check_finite=True)
        p.capture('step');pd.capture('step')
        tail=pd.attend(q[-1:])
        exact,exact_lse=decode_reference(q[-1:],logical_k,logical_v,[0],[args.prefix-1],chunk_size=8)
        torch.testing.assert_close(tail.output.cpu().double(),exact,atol=0.02,rtol=0.02)
        torch.testing.assert_close(tail.lse.cpu().double(),exact_lse,atol=5e-4,rtol=2e-5)
        dense_tail=output_projection(layer,tail.output)
        prefill_diagnostics=dict(
            raw_dense_vs_fp64_max_abs=float((tail.output.cpu().double()-exact).abs().max()),
            hf_eager_vs_sdpa_max_abs=float((hf_eager-hf_prefill).abs().max()),
            hf_eager_vs_sdpa_relative_l2=float(torch.linalg.vector_norm((hf_eager-hf_prefill).float())/torch.linalg.vector_norm(hf_prefill.float()).clamp_min(1e-8)))
        print(json.dumps({'prefill_diagnostics':prefill_diagnostics}),flush=True)
        tol=0.03
        torch.testing.assert_close(dense_tail,hf_prefill[:,-1],atol=tol,rtol=tol)
        for i in range(args.steps):
            pos=args.prefix+i
            x=hidden[:,pos:pos+1];cs,ss=cos[:,pos:pos+1],sin[:,pos:pos+1]
            results={}
            names=['cab3_strict','prepared_graph_strict'] if i%2==0 else ['prepared_graph_strict','cab3_strict']
            for name in names:
                torch.cuda.synchronize();start=time.perf_counter()
                result=decode_layer(layer,norm,x,cs,ss,old if name=='cab3_strict' else p,
                                    graph=name=='prepared_graph_strict')
                torch.cuda.synchronize()
                timings.append(dict(step=i,impl=name,wall_ms=(time.perf_counter()-start)*1000))
                results[name]=result
            yo,ro,qo,ko,vo=results['cab3_strict']
            yp,rp,qp,kp,vp=results['prepared_graph_strict']
            torch.testing.assert_close(yo,yp,atol=0.02,rtol=0.02)
            logical_k[0]=torch.cat((logical_k[0],kp.cpu()));logical_v[0]=torch.cat((logical_v[0],vp.cpu()))
            actual=sparse.materialize(0)
            assert torch.equal(actual[0].cpu(),logical_k[0]) and torch.equal(actual[1].cpu(),logical_v[0])
            ref,lse=decode_reference(qp,logical_k,logical_v,[0],[pos],chunk_size=8,selected_blocks=rp.selected_blocks)
            torch.testing.assert_close(rp.output.cpu().double(),ref,atol=0.02,rtol=0.02)
            torch.testing.assert_close(rp.lse.cpu().double(),lse,atol=5e-4,rtol=2e-5)
            hf=layer(norm(x),position_embeddings=(cs,ss),attention_mask=None,past_key_value=hf_cache)[0][:,0]
            dense_result=pd.step(qp,kp,vp,use_graph=True)
            yd=output_projection(layer,dense_result.output)
            torch.testing.assert_close(yd,hf,atol=tol,rtol=tol)
            checks.append(dict(step=i,sparse_vs_eager_max_abs=float((yo-yp).abs().max()),
                dense_vs_hf_relative_l2=float(torch.linalg.vector_norm((yd-hf).float())/torch.linalg.vector_norm(hf.float()).clamp_min(1e-8)),
                sparse_fp64_max_abs=float((rp.output.cpu().double()-ref).abs().max())))
            sparse_dense.append(float(torch.linalg.vector_norm((yp-yd).float())/torch.linalg.vector_norm(yd.float()).clamp_min(1e-8)))
        p.close();pd.close()
    record=dict(status='ok',model_label=args.model_label,checkpoint_revision=root.name,
        config_sha256=hashlib.sha256((root/'config.json').read_bytes()).hexdigest(),tensor_hashes=hashes,
        hf_dense_reference='transformers Qwen2Attention SDPA',prefill_diagnostics=prefill_diagnostics,
        selected_weight_and_embedding_bytes=read_bytes,q_heads=hq,kv_heads=hk,head_dim=d,
        prefix=args.prefix,steps=args.steps,teacher_forced=True,checks=checks,layer_wall_samples=timings,
        sparse_vs_dense_relative_l2=sparse_dense,scope='pretrained layer0 Attention only; no MLP/full model/logits/generation quality claim',
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        timestamp_utc=datetime.now(timezone.utc).isoformat())
    out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('checks','tensor_hashes','layer_wall_samples','sparse_vs_dense_relative_l2')}))


if __name__=='__main__':main()
