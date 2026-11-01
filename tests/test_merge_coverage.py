"""Expanded merge shapes and checked-API rejection; one authorized GPU only."""
import pytest
import torch
from moba_opt.ops.merge_cuda import merge_forward_cuda
from moba_opt.ops.merge_triton import merge_forward as merge_triton
from test_merge_triton import synth, check_vs_fp64

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("dim",range(8,257,8))
def test_masked_tail_dimensions(dim):
    so,sl,ro,rl,mp=synth([0,1,2,0,3,1,2],D=dim,seed=71)
    inp=(so.view(7,1,dim),sl.view(7,1),ro.view(-1,1,dim),rl,mp)
    for implementation in (merge_forward_cuda,merge_triton):
        out,lse=implementation(*inp,kmax=4)
        check_vs_fp64(out,lse,(so,sl,ro,rl,mp))


@pytest.mark.parametrize("bad",[-1,4])
def test_invalid_mapping_rejected_without_launching_bad_access(bad):
    so,sl,ro,rl,mp=synth([1,1,1,1])
    mp[0]=bad
    inp=(so.view(4,1,64),sl.view(4,1),ro.view(-1,1,64),rl,mp)
    for implementation in (merge_forward_cuda,merge_triton):
        with pytest.raises((RuntimeError,ValueError),match="mapping entries"):
            implementation(*inp,kmax=4)


def test_host_input_rejected():
    so,sl,ro,rl,mp=synth([1,1,1,1])
    for implementation in (merge_forward_cuda,merge_triton):
        with pytest.raises((RuntimeError,ValueError),match="all merge inputs"):
            implementation(so.view(4,1,64),sl.view(4,1).cpu(),ro.view(-1,1,64),rl,mp,kmax=4)


@pytest.mark.parametrize("bad_kind",["float","rank"])
def test_mapping_type_and_rank_are_not_silently_coerced(bad_kind):
    so,sl,ro,rl,mp=synth([1,1,1,1])
    bad=mp.float() if bad_kind=="float" else mp.reshape(2,2)
    for implementation in (merge_forward_cuda,merge_triton):
        with pytest.raises(ValueError,match="mapping must be a 1D"):
            implementation(so.view(4,1,64),sl.view(4,1),ro.view(-1,1,64),rl,bad,kmax=4)
