"""Small Qwen2 LoRA prefill trainer. KV cache and CUDA Graph remain inference-only."""
import copy
from contextlib import contextmanager
from functools import partial
import hashlib
import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from flash_attn import flash_attn_func
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
import moba_opt.grouped_attention as grouped


class LoRALinear(nn.Module):
    def __init__(self,base,rank=8,alpha=16):
        super().__init__()
        if not isinstance(base,nn.Linear):raise ValueError('plain Linear base required')
        if isinstance(rank,bool) or not isinstance(rank,int) or rank<=0 or alpha<=0:raise ValueError('positive rank/alpha required')
        self.base=base.requires_grad_(False);self.rank=rank;self.scale=alpha/rank;self.enabled=True
        self.lora_a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.lora_b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a,a=math.sqrt(5))

    @property
    def weight(self):return self.base.weight
    @property
    def bias(self):return self.base.bias

    def student(self,x):
        update=F.linear(F.linear(x,self.lora_a.to(x.dtype)),self.lora_b.to(x.dtype))
        return self.base(x)+update*self.scale

    def forward(self,x):return self.student(x) if self.enabled else self.base(x)

    @torch.no_grad()
    def merged(self):
        result=copy.deepcopy(self.base)
        result.weight.copy_((self.base.weight.float()+self.scale*(self.lora_b@self.lora_a)).to(self.base.weight.dtype))
        return result.requires_grad_(False)


class Qwen2Distillation(nn.Module):
    def __init__(self,model,dense_layers,rank=8,alpha=16,chunk=128,topk=4):
        super().__init__();self.model=model.eval().requires_grad_(False)
        if model.config.model_type!='qwen2':raise ValueError('Qwen2 required')
        self.dense_layers=tuple(dense_layers);n=len(model.model.layers)
        if len(set(self.dense_layers))!=len(self.dense_layers) or any(not isinstance(i,int) or isinstance(i,bool) or not 0<=i<n for i in self.dense_layers):raise ValueError('invalid dense layers')
        self.hq=model.config.num_attention_heads;self.hk=model.config.num_key_value_heads
        self.dim=model.config.hidden_size//self.hq
        self.route=dict(chunk_size=chunk,topk=topk,deterministic_merge=True,stable_routes=True,route_precision='fp32')
        for layer in model.model.layers:
            for name in ('q_proj','k_proj','v_proj','o_proj'):
                setattr(layer.self_attn,name,LoRALinear(getattr(layer.self_attn,name),rank,alpha))
        self.model.eval()
        self.record_routes=False;self.route_trace={}

    def adapter_parameters(self):
        return [p for m in self.model.modules() if isinstance(m,LoRALinear) for p in (m.lora_a,m.lora_b)]

    def adapter_state(self):
        return {n:p.detach().cpu().contiguous().clone() for n,p in self.model.named_parameters() if n.endswith(('.lora_a','.lora_b'))}

    @torch.no_grad()
    def load_adapters(self,state):
        named={n:p for n,p in self.model.named_parameters() if n.endswith(('.lora_a','.lora_b'))}
        if set(state)!=set(named):raise ValueError('adapter key mismatch')
        for name,p in named.items():
            if state[name].shape!=p.shape or not bool(torch.isfinite(state[name]).all()):raise ValueError('invalid adapter tensor')
            p.copy_(state[name].to(p.device,p.dtype))

    @contextmanager
    def inference(self,enabled):
        modules=[m for m in self.model.modules() if isinstance(m,LoRALinear)]
        params=self.adapter_parameters();flags=[p.requires_grad for p in params];switches=[m.enabled for m in modules]
        try:
            for p in params:p.requires_grad_(False)
            for m in modules:m.enabled=enabled
            yield self.model
        finally:
            for m,flag in zip(modules,switches):m.enabled=flag
            for p,flag in zip(params,flags):p.requires_grad_(flag)

    @staticmethod
    def linear(module,x,teacher):
        return module.base(x) if teacher else module.student(x)

    def block(self,index,h,cos,sin,cu,teacher):
        layer=self.model.model.layers[index];a=layer.self_attn;b,t,_=h.shape;residual=h
        x=layer.input_layernorm(h)
        q=self.linear(a.q_proj,x,teacher).view(b,t,self.hq,self.dim).transpose(1,2)
        k=self.linear(a.k_proj,x,teacher).view(b,t,self.hk,self.dim).transpose(1,2)
        v=self.linear(a.v_proj,x,teacher).view(b,t,self.hk,self.dim).transpose(1,2)
        q,k=apply_rotary_pos_emb(q,k,cos,sin)
        q,k,v=[z.transpose(1,2).reshape(b*t,-1,self.dim).contiguous() for z in (q,k,v)]
        if teacher or index in self.dense_layers:out=flash_attn_func(q[None],k[None],v[None],causal=True)[0]
        else:
            previous=grouped.ROUTE_HOOK
            if self.record_routes:
                def trace(**kw):self.route_trace.setdefault(index,[]).append(hashlib.sha256(kw['selected'].detach().cpu().numpy().tobytes()).hexdigest())
                grouped.ROUTE_HOOK=trace
            try:out=grouped.opt_gqa_cuda(q,k,v,cu,t,self.route)
            finally:grouped.ROUTE_HOOK=previous
        h=residual+self.linear(a.o_proj,out.reshape(b,t,-1),teacher)
        return h+layer.mlp(layer.post_attention_layernorm(h))

    def forward(self,ids,*,teacher=False,checkpoint_blocks=True):
        if ids.ndim!=2 or ids.shape[0]!=1:raise ValueError('batch1 token IDs required')
        h=self.model.model.embed_tokens(ids);t=ids.shape[1]
        cos,sin=self.model.model.rotary_emb(h,torch.arange(t,device=ids.device)[None])
        cu=torch.tensor([0,t],device=ids.device,dtype=torch.int32)
        for i in range(len(self.model.model.layers)):
            fn=partial(self.block,i,teacher=teacher)
            if checkpoint_blocks and not teacher and torch.is_grad_enabled():
                h=checkpoint(fn,h,cos,sin,cu,use_reentrant=False,preserve_rng_state=False)
            else:h=fn(h,cos,sin,cu)
        return self.model.model.norm(h)

    @torch.no_grad()
    def teacher_logits(self,ids,positions):
        # no_grad, rather than inference tensors used as autograd loss targets.
        h=self(ids,teacher=True,checkpoint_blocks=False)
        return self.model.lm_head(h.index_select(1,positions)).float().detach()

    def student_logits(self,ids,positions,checkpoint_blocks=True):
        h=self(ids,teacher=False,checkpoint_blocks=checkpoint_blocks)
        return self.model.lm_head(h.index_select(1,positions)).float()

    def base_digest(self):
        adapters={id(p) for p in self.adapter_parameters()};sha=hashlib.sha256()
        for name,p in self.model.named_parameters():
            if id(p) in adapters:continue
            sha.update(name.encode());sha.update(str((p.shape,p.dtype)).encode())
            sha.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return sha.hexdigest()

    @torch.no_grad()
    def merged_model(self):
        result=copy.deepcopy(self.model)
        for layer in result.model.layers:
            for name in ('q_proj','k_proj','v_proj','o_proj'):
                setattr(layer.self_attn,name,getattr(layer.self_attn,name).merged())
        return result.eval().requires_grad_(False)


def distillation_loss(student,teacher,labels,temperature=2.,kl_weight=0.8):
    if student.shape!=teacher.shape:raise ValueError('teacher/student logit shapes differ')
    student=student.reshape(-1,student.shape[-1]);teacher=teacher.reshape_as(student);labels=labels.reshape(-1)
    a=F.log_softmax(student/temperature,-1);b=F.log_softmax(teacher.detach()/temperature,-1)
    kl=F.kl_div(a,b,log_target=True,reduction='batchmean')
    ce=F.cross_entropy(student,labels)
    return kl_weight*temperature**2*kl+(1-kl_weight)*ce,dict(kl=kl.detach(),ce=ce.detach())
