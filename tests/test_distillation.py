import json
from pathlib import Path
import pytest
from scripts.distill_protocol import training_plan,learning_rate,validation_score,selected_step
from scripts.multilen_protocol import windows
from scripts import run_shared_gpu_batch as guard

RECIPE=json.loads((Path(__file__).parents[1]/'configs/distill_pilot_20261103.json').read_text())


def test_schedule_has100_updates_exact_ratio_and_causal_labels():
    p=training_plan(2549174,RECIPE)
    assert len(p)==100 and sum(r['length']==8192 for r in p)==25
    assert sum(r['length'] for r in p)==358400
    assert p==training_plan(2549174,RECIPE)
    for r in p:
        assert len(set(r['positions']))==64
        assert min(r['positions'])>=0 and max(r['positions'])+1<r['length']
    assert learning_rate(10,RECIPE)==pytest.approx(1e-4)
    assert learning_rate(100,RECIPE)==pytest.approx(1e-5)


def test_new_test_cohort_and_validation_only_checkpoint_selection():
    rr=windows(RECIPE,'test');assert min(r['start'] for r in rr)==49152
    checkpoints=[]
    for step in RECIPE['training']['validate_updates']:
        rows=[dict(w,tail_delta_nll=0.1-step/1000,continuation_delta_nll=0.1-step/1000) for w in windows(RECIPE,'valid')]
        checkpoints.append(dict(step=step,rows=rows))
    assert selected_step(checkpoints,RECIPE)==100
    checkpoints[0]['rows'][0]['split']='test'
    with pytest.raises(ValueError):selected_step(checkpoints,RECIPE)


def test_gpu0_explicitly_selectable_gpu2_rejected(monkeypatch):
    monkeypatch.setattr(guard,'GPU_INDEX',0);monkeypatch.setattr(guard,'GPU_UUID','GPU-approved')
    def query(cmd,**kw):
        assert '--id=GPU-approved' in cmd
        return '0, GPU-approved, NVIDIA A40, 0, 46068\n' if any(x.startswith('--query-gpu') for x in cmd) else ''
    monkeypatch.setattr(guard.subprocess,'check_output',query)
    assert guard.inspect()[:2]==(0,set())
    monkeypatch.setattr(guard,'GPU_INDEX',2)
    with pytest.raises(RuntimeError):guard.inspect()


def test_kl_is_per_prediction_not_batch_sum():
    import torch
    from examples.qwen2_distillation import distillation_loss
    torch.manual_seed(3);s=torch.randn(1,64,128,requires_grad=True);t=torch.randn_like(s);y=torch.arange(64)[None]
    a,_=distillation_loss(s,t,y);b,_=distillation_loss(s[:,0:1],t[:,0:1],y[:,0:1])
    c,_=distillation_loss(s.expand(3,-1,-1),t.expand(3,-1,-1),y.expand(3,-1))
    torch.testing.assert_close(a,c)
    assert torch.isfinite(b)


def small_model():
    import torch
    from transformers import Qwen2Config,Qwen2ForCausalLM
    torch.manual_seed(701)
    cfg=Qwen2Config(vocab_size=128,hidden_size=64,intermediate_size=128,num_hidden_layers=2,
                   num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
    return Qwen2ForCausalLM(cfg).to('cuda',torch.bfloat16).eval().requires_grad_(False)


@pytest.mark.parametrize('checkpoint_blocks',[False,True])
def test_lora_zero_equivalence_and_backward(checkpoint_blocks):
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from examples.qwen2_distillation import Qwen2Distillation,distillation_loss
    from examples.full_qwen2 import FullQwen2Session
    m=small_model();ids=torch.arange(1,34,device='cuda')[None]
    with torch.no_grad(),FullQwen2Session(m,40,mode='moba',chunk=4,topk=2,dense_layers=(0,),
         deterministic_merge=True,stable_routes=True,route_precision='fp32') as s:
        expected=s.prefill_hidden(ids)
    d=Qwen2Distillation(m,(0,),rank=2,alpha=4,chunk=4,topk=2)
    positions=torch.tensor([5,16,30],device='cuda')
    with torch.no_grad():torch.testing.assert_close(d(ids,checkpoint_blocks=False),expected,atol=0,rtol=0)
    d.record_routes=True
    logits=d.student_logits(ids,positions,checkpoint_blocks)
    teacher=d.teacher_logits(ids,positions)
    loss,_=distillation_loss(logits,teacher,ids[:,positions+1]);loss.backward()
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    assert sum(float(p.grad.abs().sum()) for n,p in m.named_parameters() if n.endswith('lora_b'))>0
    assert all(len(set(v))==1 for v in d.route_trace.values())
    if checkpoint_blocks:assert all(len(v)>=2 for v in d.route_trace.values())


def test_teacher_immutable_and_merge_linear_semantics():
    import torch
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    from examples.qwen2_distillation import Qwen2Distillation
    m=small_model();d=Qwen2Distillation(m,(0,),rank=2,alpha=4,chunk=4,topk=2)
    ids=torch.arange(1,20,device='cuda')[None];p=torch.tensor([4,10,17],device='cuda')
    before=d.teacher_logits(ids,p).clone();digest=d.base_digest()
    with torch.no_grad():
        for name,param in m.named_parameters():
            if name.endswith('lora_b'):param.normal_(std=0.001)
    assert torch.equal(before,d.teacher_logits(ids,p)) and d.base_digest()==digest
    merged=d.merged_model();assert not any(x.requires_grad for x in merged.parameters())
    assert all(not name.endswith(('lora_a','lora_b')) for name,_ in merged.named_parameters())
