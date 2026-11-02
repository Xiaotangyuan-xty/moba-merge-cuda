"""Exclusive fixed-active-set decode with persistent buffers and optional Graph.

The strict default checks finite values synchronously. check_finite=False is
an explicit caller contract, matching FA's finite-input assumption. Structural
checks remain active. Outputs alias workspaces and are overwritten next call.
"""
import torch

from .cache_layout import IncrementalLayout, positive_int
from .kv_cache import DecodeResult, _ints, _mode
from .ops import paged_decode as attention
from .ops import prepared_decode as kernels


class PreparedDecode:
    def __init__(self,cache,sequence_ids,num_q_heads,*,topk=4,mode="moba",check_finite=True):
        with cache._lock,torch.cuda.device(cache.device):
            cache._check_stream()
            self.num_q_heads=positive_int(num_q_heads,"num_q_heads")
            if self.num_q_heads%cache.num_kv_heads:
                raise ValueError("Hq must be a multiple of Hkv")
            if not isinstance(check_finite,bool):raise ValueError("check_finite must be bool")
            self.dense=_mode(mode,topk)
            self.cache=cache
            self._state=IncrementalLayout(cache._layout,_ints(sequence_ids,"sequence_ids"))
            self.ids=self._state.ids
            b,h,d=len(self.ids),self.num_q_heads,cache.head_dim
            width=max(1,(cache._layout.max_seq_len-1)//cache.chunk_size)
            if b*h*width>=2**31:raise ValueError("route score grid exceeds int32 capacity")
            self.remote=0 if self.dense else min(topk-1,(cache._layout.max_seq_len-1)//cache.chunk_size)
            self.check_finite=check_finite
            self.closed=False
            self.failed=False
            self._graphs={}
            self.delta_uploads=0
            kw=dict(device=cache.device,dtype=cache.dtype)
            self.q=torch.zeros((b,h,d),**kw)
            self.k=torch.zeros((b,cache.num_kv_heads,d),**kw)
            self.v=torch.zeros_like(self.k)
            self.output=torch.empty_like(self.q)
            self.lse=torch.empty((b,h),device=cache.device,dtype=torch.float32)
            self.scores=torch.empty((b,h,width),device=cache.device,dtype=torch.float32)
            self.values=torch.empty((b,h,self.remote),device=cache.device,dtype=torch.float32)
            self.indices=torch.empty((b,h,self.remote),device=cache.device,dtype=torch.int64)
            self.selected=torch.empty((b,h,self.remote+1),device=cache.device,dtype=torch.int32)
            self.gpu_ids=torch.tensor(self.ids,device=cache.device,dtype=torch.int32)
            self.positions=torch.empty(b,device=cache.device,dtype=torch.int32)
            self.delta=torch.full((2,b),-1,device=cache.device,dtype=torch.int32)
            self.delta_host=torch.empty((2,b),dtype=torch.int32,pin_memory=True)
            table,lengths=cache._table.clone(),cache._lengths.clone()
            # Publish the exclusive lease only after all preparation allocations.
            cache._table,cache._lengths=table,lengths
            cache._prepared_owner=self

    @property
    def lengths(self):
        with self.cache._lock:return tuple(self._state.lengths)

    @property
    def page_table(self):
        with self.cache._lock:return tuple(map(tuple,self._state.tables))

    def _check(self):
        if self.closed or self.cache._prepared_owner is not self:
            raise ValueError("prepared decode is closed or no longer owns cache")
        if self.failed:raise RuntimeError("CUDA execution failed; discard prepared decode and cache")
        if torch.cuda.is_current_stream_capturing():raise ValueError("use capture(), not external capture")
        if torch.cuda.current_stream(self.cache.device).cuda_stream!=self.cache._stream:
            raise ValueError("cross-stream cache access is unsupported")

    def _input(self,tensor,target,name):
        if not isinstance(tensor,torch.Tensor) or tensor.shape!=target.shape:
            raise ValueError(f"{name} must have fixed shape {tuple(target.shape)}")
        if tensor.device!=target.device or tensor.dtype!=target.dtype or not tensor.is_contiguous():
            raise ValueError(f"{name} must match device/dtype and be contiguous")
        if tensor.requires_grad:raise ValueError("prepared decode is inference-only")
        self.cache._reject_storage_alias(tensor)
        workspaces={x.untyped_storage().data_ptr() for x in (self.q,self.k,self.v)}
        if tensor is not target and tensor.untyped_storage().data_ptr() in workspaces:
            raise ValueError("input aliases another prepared workspace; clone first")
        if self.check_finite and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains NaN/Inf")

    def _upload(self,plan):
        if not plan.updates:return
        self.delta_host.fill_(-1)
        rows={sid:i for i,sid in enumerate(self.ids)}
        for sid,logical,physical in plan.updates:
            row=rows[sid]
            self.delta_host[0,row]=logical
            self.delta_host[1,row]=physical
        # Synchronous tiny H2D prevents pinned-buffer reuse races. Only on new
        # pages; no CPU metadata transfer on ordinary within-page steps.
        self.delta.copy_(self.delta_host,non_blocking=False)
        self.delta_uploads+=1

    def _compute(self,append):
        c=self.cache
        kernels.metadata(c._table,c._lengths,self.gpu_ids,self.positions,self.delta,append)
        if append:
            attention.append_tokens(self.k,self.v,c.k_cache,c.v_cache,c._table,self.gpu_ids,self.positions)
            kernels.step_means(c.k_cache,c._table,c._means,self.gpu_ids,self.positions,c.chunk_size)
        if not self.dense:
            if self.remote:
                attention.route_scores(self.q,c._means,self.gpu_ids,self.positions,c.chunk_size,self.scores)
                torch.topk(self.scores,self.remote,dim=-1,sorted=False,out=(self.values,self.indices))
            kernels.selected(self.values,self.indices,self.positions,self.selected,c.chunk_size)
        attention.paged_attention(self.q,c.k_cache,c.v_cache,c._table,c._lengths,
            self.gpu_ids,self.positions,self.selected,self.output,self.lse,c.chunk_size,self.dense)

    def _run(self,kind,use_graph):
        try:
            if use_graph:self._graphs[kind].replay()
            else:self._compute(kind=="step")
        except Exception:
            self.failed=True
            raise

    def _result(self,positions):
        return DecodeResult(self.output,self.lse,None if self.dense else self.selected,self.ids,positions)

    @torch.no_grad()
    def attend(self,q,*,use_graph=False):
        with self.cache._lock,torch.cuda.device(self.cache.device):
            self._check()
            self._input(q,self.q,"q")
            pos=tuple(self._state.lengths[s]-1 for s in self.ids)
            if min(pos)<0:raise ValueError("query positions must refer to cached tokens")
            if use_graph and "attend" not in self._graphs:raise ValueError("capture('attend') first")
            if q is not self.q:self.q.copy_(q)
            self._run("attend",use_graph)
            return self._result(pos)

    @torch.no_grad()
    def step(self,q,k,v,*,use_graph=False):
        """One token per fixed active slot; validation/capacity errors are atomic."""
        with self.cache._lock,torch.cuda.device(self.cache.device):
            self._check()
            for tensor,target,name in ((q,self.q,"q"),(k,self.k,"k"),(v,self.v,"v")):
                self._input(tensor,target,name)
            if use_graph and "step" not in self._graphs:raise ValueError("capture('step') first")
            plan=self._state.plan()
            for tensor,target in ((q,self.q),(k,self.k),(v,self.v)):
                if tensor is not target:target.copy_(tensor)
            self._upload(plan)
            try:self._run("step",use_graph)
            except Exception:
                self.failed=True
                raise
            self._state.commit(plan)
            return self._result(plan.positions)

    @torch.no_grad()
    def capture(self,kind="step"):
        """Capture GPU compute only. Visible cache state is restored afterwards."""
        if kind not in ("step","attend"):raise ValueError("kind must be step or attend")
        c=self.cache
        with c._lock,torch.cuda.device(c.device):
            self._check()
            if kind in self._graphs:raise ValueError("graph already captured")
            plan=self._state.plan() if kind=="step" else None
            if kind=="attend" and any(self._state.lengths[s]==0 for s in self.ids):
                raise ValueError("attend capture requires nonempty cached sequences")
            snapshots=[t.clone() for t in (c._table,c._lengths,c._means,self.delta)]
            uploads=self.delta_uploads
            stream=torch.cuda.Stream(device=c.device)
            stream.wait_stream(torch.cuda.current_stream(c.device))
            try:
                if plan is not None:self._upload(plan)
                stream.wait_stream(torch.cuda.current_stream(c.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self._compute(kind=="step")
                        c._lengths.copy_(snapshots[1])
                torch.cuda.current_stream(c.device).wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):self._compute(kind=="step")
                self._graphs[kind]=graph
            finally:
                torch.cuda.current_stream(c.device).wait_stream(stream)
                for target,snapshot in zip((c._table,c._lengths,c._means,self.delta),snapshots):
                    target.copy_(snapshot)
                self.delta_uploads=uploads
            return self

    def close(self):
        if self.closed:return
        with self.cache._lock,torch.cuda.device(self.cache.device):
            if self.cache._prepared_owner is not self:raise ValueError("prepared ownership lost")
            if torch.cuda.current_stream(self.cache.device).cuda_stream!=self.cache._stream:
                raise ValueError("cross-stream cache access is unsupported")
            if self.failed:self.cache._invalid=True
            else:
                self._state.publish(self.cache._layout)
                self.cache._candidate_count=self.cache._count_candidates(self._state.lengths)
            self.cache._prepared_owner=None
            self._graphs.clear()
            self.closed=True

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
