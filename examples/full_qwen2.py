"""Experimental batch-1 full Qwen2 inference with the repository's KV cache.

Weights stay in the unchanged HF model. No sampling policy, scheduler, padding,
training, sliding window or dynamic active set. Graphs have fixed capacity.
"""
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func,flash_attn_with_kvcache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from moba_opt import PagedKVCache
from moba_opt.grouped_attention import opt_gqa_cuda
from moba_opt.cache_layout import positive_int
from moba_opt.ops import prepared_decode as step_ops
from moba_opt.ops import paged_decode as page_ops


class FullQwen2Session:
    def __init__(self,model,capacity,*,mode='dense',chunk=128,topk=4,
                 backend='eager',fused_qkv=False,page=16,dense_backend='fa2',dense_num_splits=0,
                 deterministic_merge=False,stable_routes=False,route_precision='upstream',
                 dense_layers=(),prefill_mode='match'):
        if mode not in ('dense','moba'):raise ValueError('mode')
        if backend not in ('eager','attention_graph','model_graph'):raise ValueError('backend')
        if model.config.model_type!='qwen2':raise ValueError('Qwen2 architecture required')
        if getattr(model.config,'use_sliding_window',False):raise ValueError('sliding window unsupported')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('eval model with gradients disabled required')
        capacity=positive_int(capacity,'capacity');chunk=positive_int(chunk,'chunk');topk=positive_int(topk,'topk')
        if topk>9:raise ValueError('topk must be <=9')
        self.model,self.capacity=model,capacity
        self.mode,self.chunk,self.topk,self.backend=mode,chunk,topk,backend
        if dense_backend not in ('fa2','triton'):raise ValueError('dense backend')
        if isinstance(dense_num_splits,bool) or not isinstance(dense_num_splits,int) or dense_num_splits<0:
            raise ValueError('dense_num_splits must be a nonnegative integer; 0 uses FA2 auto splitting')
        self.dense_backend=dense_backend
        self.dense_num_splits=dense_num_splits
        self.all_selected=mode=='moba' and (self.capacity+chunk-1)//chunk<=topk
        self.effective_dense=mode=='dense' or topk==1 or self.all_selected
        nlayer=len(model.model.layers)
        dense_layers=tuple(dense_layers)
        if len(set(dense_layers))!=len(dense_layers) or any(isinstance(i,bool) or not isinstance(i,int) or not 0<=i<nlayer for i in dense_layers):
            raise ValueError('dense_layers must be unique valid layer indices')
        if prefill_mode not in ('match','dense'):raise ValueError('prefill_mode')
        if route_precision not in ('upstream','fp32'):raise ValueError('route_precision')
        if not isinstance(deterministic_merge,bool) or not isinstance(stable_routes,bool):raise ValueError('stability flags must be bool')
        self.layer_dense=tuple(self.effective_dense or i in dense_layers for i in range(nlayer))
        self.prefill_mode=prefill_mode
        self.route_config=dict(chunk_size=chunk,topk=topk,deterministic_merge=deterministic_merge,
                               stable_routes=stable_routes,route_precision=route_precision)
        if backend=='attention_graph' and any(self.layer_dense) and dense_backend=='fa2':
            raise ValueError('FA2 dense uses eager or model_graph; attention_graph is for the Triton core')
        self.fused_qkv=fused_qkv
        self.profile=False
        self.device=next(model.parameters()).device
        if self.device.type!='cuda':raise ValueError('CUDA model required')
        if backend=='model_graph' and 'dynamic' in model.model.rotary_emb.rope_type:
            raise ValueError('dynamic RoPE is not supported by the fixed Graph path')
        self.stream=torch.cuda.current_stream(self.device).cuda_stream
        self.hq=model.config.num_attention_heads
        self.hk=model.config.num_key_value_heads
        self.dim=model.config.hidden_size//self.hq
        self.caches=[PagedKVCache(1,capacity,self.hk,self.dim,page_size=page,chunk_size=chunk,
                        dtype=next(model.parameters()).dtype,device=self.device) for _ in model.model.layers]
        self.plans=[];self.graph=None;self.closed=False;self.failed=False
        self.dense_views=[(c.k_cache.view(1,-1,self.hk,self.dim),c.v_cache.view(1,-1,self.hk,self.dim)) for c in self.caches]
        self.input=torch.zeros((1,1),device=self.device,dtype=torch.long)
        self.packed=[]
        if fused_qkv:
            for layer in model.model.layers:
                a=layer.self_attn
                self.packed.append((torch.cat([a.q_proj.weight,a.k_proj.weight,a.v_proj.weight],0),
                                    torch.cat([a.q_proj.bias,a.k_proj.bias,a.v_proj.bias],0)))

    def scope(self,name):
        return torch.profiler.record_function(name) if self.profile else nullcontext()

    def check(self):
        if self.closed or self.failed:raise ValueError('closed/failed session')
        if torch.cuda.current_stream(self.device).cuda_stream!=self.stream:raise ValueError('cross-stream session')

    def project(self,index,hidden,cos,sin):
        a=self.model.model.layers[index].self_attn
        b,t,_=hidden.shape
        if self.fused_qkv:
            qkv=F.linear(hidden,*self.packed[index])
            q,k,v=qkv.split((self.hq*self.dim,self.hk*self.dim,self.hk*self.dim),dim=-1)
        else:q,k,v=a.q_proj(hidden),a.k_proj(hidden),a.v_proj(hidden)
        q=q.view(b,t,self.hq,self.dim).transpose(1,2)
        k=k.view(b,t,self.hk,self.dim).transpose(1,2)
        v=v.view(b,t,self.hk,self.dim).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        return tuple(x.transpose(1,2).reshape(b*t,-1,self.dim).contiguous() for x in (q,k,v))

    @torch.no_grad()
    def prefill_hidden(self,ids):
        self.check()
        if self.plans or any(c.lengths[0] for c in self.caches):raise ValueError('fresh session required')
        if ids.dim()!=2 or ids.shape[0]!=1 or not 0<ids.shape[1]<self.capacity:
            raise ValueError('batch=1, nonempty prefix below capacity required')
        if ids.device!=self.device or ids.dtype!=torch.long:raise ValueError('CUDA int64 token IDs required')
        t=ids.shape[1]
        h=self.model.model.embed_tokens(ids)
        cos,sin=self.model.model.rotary_emb(h,torch.arange(t,device=self.device)[None])
        cu=torch.tensor([0,t],device=self.device,dtype=torch.int32)
        for i,layer in enumerate(self.model.model.layers):
            residual=h
            q,k,v=self.project(i,layer.input_layernorm(h),cos,sin)
            self.caches[i].append(k,v,sequence_ids=[0],cu_seqlens=[0,t])
            if self.layer_dense[i] or self.prefill_mode=='dense':out=flash_attn_func(q[None],k[None],v[None],causal=True)[0]
            else:out=opt_gqa_cuda(q,k,v,cu,t,self.route_config)
            h=residual+layer.self_attn.o_proj(out.reshape(1,t,-1))
            h=h+layer.mlp(layer.post_attention_layernorm(h))
        return self.model.model.norm(h)

    @torch.no_grad()
    def prepare(self):
        self.check()
        if self.plans:raise ValueError('already prepared')
        if not self.caches[0].lengths[0]:raise ValueError('prefill first')
        if any(self.layer_dense) and self.dense_backend=='fa2':
            for c,dense in zip(self.caches,self.layer_dense):
                if not dense:continue
                allocated=sum(x>=0 for x in c.page_table[0])
                if c.page_table[0][:allocated]!=tuple(range(allocated)) or c._layout.free_pages!=tuple(range(c._layout.num_pages-1,allocated-1,-1)):
                    raise ValueError('dense FA2 control requires monotonic single-request physical pages')
        self.plans=[c.prepare_decode([0],self.hq,mode='dense' if dense else self.mode,topk=self.topk,
                                    check_finite=False,stable_routes=self.route_config['stable_routes'])
                    for c,dense in zip(self.caches,self.layer_dense)]
        if self.backend=='attention_graph' and not (self.effective_dense and self.dense_backend=='fa2'):
            for p in self.plans:p.capture('step')
        elif self.backend=='model_graph':self.capture_model()

    def compute_step(self):
        with self.scope('model/embedding_rope'):
            h=self.model.model.embed_tokens(self.input)
            position=self.caches[0]._lengths[:1].to(torch.long).view(1,1)
            cos,sin=self.model.model.rotary_emb(h,position)
        for i,(layer,p) in enumerate(zip(self.model.model.layers,self.plans)):
            residual=h
            with self.scope('model/norm_qkv_rope'):
                q,k,v=self.project(i,layer.input_layernorm(h),cos,sin)
                p.q.copy_(q);p.k.copy_(k);p.v.copy_(v)
            with self.scope('model/cache_attention'):
                if self.layer_dense[i] and self.dense_backend=='fa2':
                    c=p.cache
                    step_ops.metadata(c._table,c._lengths,p.gpu_ids,p.positions,p.delta,True)
                    page_ops.append_tokens(p.k,p.v,c.k_cache,c.v_cache,c._table,p.gpu_ids,p.positions)
                    step_ops.step_means(c.k_cache,c._table,c._means,p.gpu_ids,p.positions,c.chunk_size)
                    dk,dv=self.dense_views[i]
                    out=flash_attn_with_kvcache(p.q[:,None],dk,dv,cache_seqlens=c._lengths,
                                              causal=True,num_splits=self.dense_num_splits)
                    p.output.copy_(out[:,0])
                elif self.backend=='attention_graph':p._run('step',True)
                else:p._compute(True)
            with self.scope('model/output_projection'):
                h=residual+layer.self_attn.o_proj(p.output.reshape(1,1,-1))
            with self.scope('model/mlp'):
                h=h+layer.mlp(layer.post_attention_layernorm(h))
        with self.scope('model/norm_lm_head'):
            self.logits=self.model.lm_head(self.model.model.norm(h))
        return self.logits

    @torch.no_grad()
    def capture_model(self):
        snapshots=[[t.clone() for t in (c._table,c._lengths,c._means,p.delta)] for c,p in zip(self.caches,self.plans)]
        counters=[p.delta_uploads for p in self.plans]
        stream=torch.cuda.Stream(device=self.device)
        try:
            for p in self.plans:p._upload(p._state.plan())
            stream.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.compute_step()
                    for c,snap in zip(self.caches,snapshots):c._lengths.copy_(snap[1])
            torch.cuda.current_stream(self.device).wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):self.compute_step()
            self.graph=graph
        finally:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            for c,p,snap,n in zip(self.caches,self.plans,snapshots,counters):
                for target,source in zip((c._table,c._lengths,c._means,p.delta),snap):target.copy_(source)
                p.delta_uploads=n

    @torch.no_grad()
    def step(self,token):
        self.check()
        if isinstance(token,bool) or not isinstance(token,int) or not 0<=token<self.model.config.vocab_size:
            raise ValueError('one valid CPU token ID required')
        if not self.plans:raise ValueError('prepare first')
        plans=[p._state.plan() for p in self.plans]
        if self.dense_backend=='fa2' and any(
            logical!=physical for plan,dense in zip(plans,self.layer_dense) if dense for _,logical,physical in plan.updates):
            raise ValueError('dense FA2 page layout changed')
        self.input.fill_(token)
        for p,plan in zip(self.plans,plans):p._upload(plan)
        try:
            if self.graph is not None:self.graph.replay()
            else:self.compute_step()
        except Exception:
            self.failed=True
            for p in self.plans:p.failed=True
            raise
        for p,plan in zip(self.plans,plans):p._state.commit(plan)
        return self.logits

    def close(self):
        if self.closed:return
        for p in self.plans:p.close()
        self.graph=None;self.closed=True

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
