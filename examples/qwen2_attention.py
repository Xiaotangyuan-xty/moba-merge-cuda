"""Qwen2 projected/RoPE QKV adapter; no model weight downloads or scheduler."""
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb


def project_qkv(layer,hidden,cos,sin):
    b,t,_=hidden.shape
    d=layer.head_dim
    q=layer.q_proj(hidden).view(b,t,-1,d).transpose(1,2)
    k=layer.k_proj(hidden).view(b,t,-1,d).transpose(1,2)
    v=layer.v_proj(hidden).view(b,t,-1,d).transpose(1,2)
    q,k=apply_rotary_pos_emb(q,k,cos,sin)
    return tuple(x.transpose(1,2).reshape(b*t,-1,d).contiguous() for x in (q,k,v))


def output_projection(layer,attention_output):
    return layer.o_proj(attention_output.flatten(-2))


def decode_layer(layer,norm,hidden,cos,sin,cache_or_plan,*,graph=False):
    q,k,v=project_qkv(layer,norm(hidden),cos,sin)
    if hasattr(cache_or_plan,'step'):
        result=cache_or_plan.step(q,k,v,use_graph=graph)
    else:
        result=cache_or_plan.append_and_attend(q,k,v,sequence_ids=[0],topk=4)
    return output_projection(layer,result.output),result,q,k,v
