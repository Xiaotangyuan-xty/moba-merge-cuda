"""Validation-only precision ablations; does not change production inference.

Route replay deliberately mutates the test hook's selected tensor before expert
packing. It is single-threaded and only replays the same sequence/shape/layer.
"""
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from examples.qwen2_distillation import LoRALinear
import moba_opt.grouped_attention as grouped

MODES=('merged_master_bf16','merged_factor_bf16','merged_master_fp32','unmerged_lora_fp32')


def tensor_error(ref,out):
    a,b=ref.float(),out.float()
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):raise ValueError('nonfinite diagnostic tensor')
    return dict(max_abs=float((a-b).abs().max()),relative_l2=float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-12)))


class ProjectionAblation:
    def __init__(self,distillation):
        self.d=distillation
        self.modules={id(m):(n,m) for n,m in self.d.model.named_modules() if isinstance(m,LoRALinear)}
        self.weights={};self.weight_rows=[]
        with torch.no_grad():
            for key,(name,m) in self.modules.items():
                delta=m.scale*(m.lora_b@m.lora_a)
                wf=m.base.weight.float()+delta
                qf=m.base.weight.float()+m.scale*(m.lora_b.bfloat16().float()@m.lora_a.bfloat16().float())
                self.weights[key]=(wf,wf.bfloat16(),qf.bfloat16())
                actual=wf.bfloat16().float()-m.base.weight.float()
                nonzero=delta.ne(0)
                self.weight_rows.append(dict(projection=name,delta_norm=float(delta.norm()),
                    rounded_update_relative_l2=float((actual-delta).norm()/delta.norm().clamp_min(1e-12)),
                    nonzero_delta_entries=int(nonzero.sum()),rounded_to_base_entries=int((actual.eq(0)&nonzero).sum()),
                    master_factor_delta_relative_l2=float((delta-(qf-m.base.weight.float())).norm()/delta.norm().clamp_min(1e-12))))

    def project(self,module,x,mode):
        wf,wb,wq=self.weights[id(module)]
        if mode=='unmerged':return module.student(x)
        if mode=='merged_master_bf16':return F.linear(x,wb,module.base.bias)
        if mode=='merged_factor_bf16':return F.linear(x,wq,module.base.bias)
        if mode=='merged_master_fp32':
            bias=None if module.base.bias is None else module.base.bias.float()
            return F.linear(x.float(),wf,bias).to(x.dtype)
        if mode=='unmerged_lora_fp32':
            update=F.linear(F.linear(x.float(),module.lora_a),module.lora_b)*module.scale
            return (module.base(x).float()+update).to(x.dtype)
        raise ValueError('unknown projection ablation')

    @contextmanager
    def mode(self,mode,projection_rows=None):
        if 'linear' in self.d.__dict__:raise ValueError('nested projection override')
        def linear(module,x,teacher):
            if teacher:return module.base(x)
            out=self.project(module,x,mode)
            if projection_rows is not None:
                # The BF16 comparisons retain the original full GEMM shape.
                # FP32 algebra control intentionally samples <=32 input rows.
                positions=torch.linspace(0,x.shape[1]-1,min(32,x.shape[1]),device=x.device).long()
                ref=out.index_select(1,positions)
                name=self.modules[id(module)][0]
                for candidate in MODES:
                    other=self.project(module,x,candidate).index_select(1,positions)
                    projection_rows.append(dict(projection=name,mode=candidate,**tensor_error(ref,other)))
                xx=x.index_select(1,positions).float()
                bias=None if module.base.bias is None else module.base.bias.float()
                separated=F.linear(xx,module.base.weight.float(),bias)+module.scale*F.linear(F.linear(xx,module.lora_a),module.lora_b)
                merged=F.linear(xx,self.weights[id(module)][0],bias)
                control=tensor_error(separated,merged)
                if control['relative_l2']>1e-5:raise AssertionError('FP32 merge algebra control failed')
                projection_rows.append(dict(projection=name,mode='fp32_algebra_control',**control))
            return out
        self.d.linear=linear
        try:yield
        finally:del self.d.linear


class RouteTape:
    def __init__(self):self.layers={};self.rows=[]

    def hook(self,layer,replay=False):
        def hook(**kw):
            tensors={k:v.detach().cpu().clone() for k,v in kw.items()}
            if layer not in self.layers:
                if replay:raise ValueError('missing route tape layer')
                self.layers[layer]=tensors;return
            ref=self.layers[layer]
            if any(not torch.equal(ref[k],v) for k,v in tensors.items() if k!='selected'):raise ValueError('route tape metadata differs')
            a,b=ref['selected'],tensors['selected']
            if a.shape!=b.shape:raise ValueError('route tape shape differs')
            changed=(a!=b).any(0)
            self.rows.append(dict(layer=layer,replay=replay,changed_query_heads=int(changed.sum()),query_heads=changed.numel(),
                removed_edges=int((a&~b).sum()),reference_edges=int(a.sum())))
            if replay:kw['selected'].copy_(ref['selected'].to(kw['selected'].device))
        return hook


@torch.no_grad()
def diagnostic_forward(ablation,ids,positions,mode='unmerged',tape=None,replay=False,projection_rows=None):
    d=ablation.d;t=ids.shape[1];h=d.model.model.embed_tokens(ids)
    cos,sin=d.model.model.rotary_emb(h,torch.arange(t,device=ids.device)[None])
    cu=torch.tensor([0,t],device=ids.device,dtype=torch.int32)
    if d.record_routes:raise ValueError('disable training route trace during ablation')
    with ablation.mode(mode,projection_rows):
        for i in range(len(d.model.model.layers)):
            previous=grouped.ROUTE_HOOK
            grouped.ROUTE_HOOK=None if tape is None else tape.hook(i,replay)
            try:h=d.block(i,h,cos,sin,cu,teacher=False)
            finally:grouped.ROUTE_HOOK=previous
    return d.model.lm_head(d.model.model.norm(h).index_select(1,positions)).float()
