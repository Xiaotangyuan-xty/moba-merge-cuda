"""Controls for diagnostic projection substitution and exact route replay."""
import pytest
import torch
from examples.qwen2_distillation import Qwen2Distillation
from examples.qwen2_merge_diagnostic import ProjectionAblation,RouteTape,diagnostic_forward
import moba_opt.grouped_attention as grouped
from test_distillation import small_model

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


def fixture():
    d=Qwen2Distillation(small_model(),(0,),rank=2,alpha=4,chunk=4,topk=2)
    with torch.no_grad():
        for n,p in d.model.named_parameters():
            if n.endswith('lora_b'):p.normal_(std=.01)
    return d,torch.arange(1,34,device='cuda')[None],torch.tensor([5,16,30],device='cuda')


@torch.no_grad()
def test_diagnostic_baseline_and_current_export_match_existing_paths():
    from examples.full_qwen2 import FullQwen2Session
    d,ids,p=fixture();a=ProjectionAblation(d)
    ref=diagnostic_forward(a,ids,p)
    assert torch.equal(ref,d.student_logits(ids,p,False))
    merged=d.merged_model()
    with FullQwen2Session(merged,40,mode='moba',chunk=4,topk=2,dense_layers=(0,),
         deterministic_merge=True,stable_routes=True,route_precision='fp32') as s:
        out=merged.lm_head(s.prefill_hidden(ids).index_select(1,p)).float()
    assert torch.equal(out,diagnostic_forward(a,ids,p,'merged_master_bf16'))


@torch.no_grad()
def test_replay_preserves_original_output_and_records_all_sparse_layers():
    d,ids,p=fixture();a=ProjectionAblation(d);t=RouteTape()
    ref=diagnostic_forward(a,ids,p,tape=t)
    assert set(t.layers)=={1}
    out=diagnostic_forward(a,ids,p,tape=t,replay=True)
    assert torch.equal(ref,out)
    assert len(t.rows)==1 and t.rows[0]['changed_query_heads']==0
    assert grouped.ROUTE_HOOK is None and 'linear' not in d.__dict__


def test_route_replay_overrides_only_same_metadata_and_restores_global_on_error():
    t=RouteTape();s=torch.zeros(2,2,8,dtype=torch.bool,device='cuda');s[0,:,4:]=True
    kw=dict(cu_chunk=torch.tensor([0,4,8],device='cuda'),filtered_chunk_indices=torch.tensor([0,1],device='cuda'),
            chunk_to_batch=torch.zeros(2,device='cuda',dtype=torch.int32),selected=s)
    t.hook(1)(**kw);s.zero_();t.hook(1,True)(**kw)
    assert torch.equal(s.cpu(),t.layers[1]['selected']) and t.rows[-1]['removed_edges']==8
    kw['cu_chunk']=kw['cu_chunk']+1
    with pytest.raises(ValueError,match='metadata'):t.hook(1,True)(**kw)
    d,ids,p=fixture();a=ProjectionAblation(d)
    with pytest.raises(ValueError,match='unknown projection'):diagnostic_forward(a,ids,p,'bad')
    assert grouped.ROUTE_HOOK is None and 'linear' not in d.__dict__


@torch.no_grad()
def test_projection_oracle_and_zero_adapter_weight_rounding_controls():
    d,ids,p=fixture()
    for n,v in d.model.named_parameters():
        if n.endswith('lora_b'):v.zero_()
    a=ProjectionAblation(d);rows=[]
    ref=diagnostic_forward(a,ids,p,projection_rows=rows)
    assert all(r['nonzero_delta_entries']==0 and r['rounded_to_base_entries']==0 for r in a.weight_rows)
    assert all(r['relative_l2']<1e-5 for r in rows if r['mode']=='fp32_algebra_control')
    for mode in ('merged_master_bf16','merged_factor_bf16'):
        assert torch.equal(ref,diagnostic_forward(a,ids,p,mode))
