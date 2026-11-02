"""Inference-only contiguous/paged KV cache and incremental MoBA attention.

Inputs are projected/position-encoded Q/K/V. The cache owns page allocation,
but is not a serving scheduler. Operations are serialized and use one CUDA
stream per cache; cross-stream access is rejected explicitly.
"""
from dataclasses import dataclass
import threading

import torch

from .cache_layout import PageAllocator, positive_int
from .ops import paged_decode as kernels


def _ints(values, name):
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu" or values.dtype not in (torch.int32, torch.int64) or values.dim() != 1:
            raise ValueError(f"{name} must be a Python sequence or CPU int32/int64 vector")
        values = values.tolist()
    if not isinstance(values, (tuple, list)):
        raise ValueError(f"{name} must be a Python sequence or CPU integer vector")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in values):
        raise ValueError(f"{name} must contain integers")
    return tuple(values)


def _mode(mode, topk):
    if mode not in ("moba", "dense"):
        raise ValueError("mode must be moba or dense")
    if positive_int(topk, "topk") > 9:
        raise ValueError("topk must be in [1,9]")
    return bool(mode == "dense" or topk == 1)


@dataclass(frozen=True)
class AppendResult:
    sequence_ids: tuple
    positions: tuple


@dataclass(frozen=True)
class DecodeResult:
    output: torch.Tensor
    lse: torch.Tensor
    selected_blocks: object  # [Nq,Hq,slots], current block first; None for dense
    sequence_ids: tuple
    positions: tuple


class PagedKVCache:
    def __init__(self, batch_size, max_seq_len, num_kv_heads, head_dim, *,
                 page_size=16, num_pages=None, chunk_size=128,
                 dtype=torch.bfloat16, device="cuda"):
        batch_size = positive_int(batch_size, "batch_size")
        max_seq_len = positive_int(max_seq_len, "max_seq_len")
        page_size = positive_int(page_size, "page_size")
        self.num_kv_heads = positive_int(num_kv_heads, "num_kv_heads")
        self.head_dim = positive_int(head_dim, "head_dim")
        if self.head_dim > 256:
            raise ValueError("decode head_dim must be in [1,256]")
        self.chunk_size = positive_int(chunk_size, "chunk_size")
        if self.chunk_size > 1024:
            raise ValueError("cache chunk_size must be <=1024")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("cache dtype must be BF16 or FP16")
        requested = torch.device(device)
        if requested.type != "cuda":
            raise ValueError("cache device must be CUDA")
        if num_pages is None:
            num_pages = batch_size * ((max_seq_len + page_size - 1) // page_size)
        self._layout = PageAllocator(batch_size, max_seq_len, page_size, num_pages, self.chunk_size)
        self.dtype = dtype
        self._lock = threading.RLock()
        with torch.cuda.device(requested):
            self.device = torch.device("cuda", torch.cuda.current_device())
            self._stream = torch.cuda.current_stream(self.device).cuda_stream
            shape = (self._layout.num_pages, page_size, self.num_kv_heads, self.head_dim)
            self.k_cache = torch.empty(shape, dtype=dtype, device=self.device)
            self.v_cache = torch.empty_like(self.k_cache)
            self._table = torch.full((batch_size, self._layout.table_width), -1,
                                     dtype=torch.int32, device=self.device)
            self._lengths = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
            max_chunks = (max_seq_len + self.chunk_size - 1) // self.chunk_size
            self._means = torch.empty((batch_size, max_chunks, self.num_kv_heads, self.head_dim),
                                      dtype=dtype, device=self.device)
            self._candidate_count = 0

    @property
    def lengths(self):
        return self._layout.lengths

    @property
    def page_table(self):
        """Immutable CPU snapshot; no caller can edit the internal page table."""
        return self._layout.tables

    def _check_stream(self):
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("CUDA Graph capture is unsupported for this cache API")
        if torch.cuda.current_stream(self.device).cuda_stream != self._stream:
            raise ValueError("cross-stream cache access is unsupported")

    def _tensor(self, tensor, name, *, expected_heads=None, expected_tokens=None):
        if not isinstance(tensor, torch.Tensor) or tensor.dim() != 3:
            raise ValueError(f"{name} must be [N,H,D]")
        if tensor.device != self.device or tensor.dtype != self.dtype:
            raise ValueError(f"{name} must match cache device and dtype")
        if tensor.requires_grad:
            raise ValueError("KV cache/decode is inference-only; detach Q/K/V")
        if tensor.shape[0] < 1 or tensor.shape[1] < 1 or tensor.shape[2] != self.head_dim:
            raise ValueError(f"{name} has an invalid shape")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} last dimension must be contiguous")
        if expected_heads is not None and tensor.shape[1] != expected_heads:
            raise ValueError(f"{name} KV head count differs from cache")
        if expected_tokens is not None and tensor.shape[0] != expected_tokens:
            raise ValueError(f"{name} token count differs")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains NaN/Inf")

    def _append_counts(self, token_count, sequence_ids, cu_seqlens):
        ids = self._layout.validate_ids(_ints(sequence_ids, "sequence_ids"), unique=True)
        cu = tuple(range(len(ids) + 1)) if cu_seqlens is None else _ints(cu_seqlens, "cu_seqlens")
        if len(cu) != len(ids) + 1 or cu[0] != 0 or cu[-1] != token_count:
            raise ValueError("append cu_seqlens must start at 0 and end at N")
        counts = tuple(b - a for a, b in zip(cu, cu[1:]))
        if any(n <= 0 for n in counts):
            raise ValueError("append sequences must be nonempty")
        return ids, counts

    def _count_candidates(self, lengths):
        return sum(max(0, (length + self.chunk_size - 1) // self.chunk_size - 1)
                   for length in lengths)

    def _gpu(self, data):
        return torch.tensor(data, dtype=torch.int32, device=self.device)

    def _metadata(self, *vectors):
        """One H2D copy; returned views keep the packed storage alive."""
        data, sizes = [], []
        for vector in vectors:
            sizes.append(len(vector))
            data.extend(vector)
        return self._gpu(data).split(sizes)

    def _reject_storage_alias(self, *tensors):
        pools = {self.k_cache.untyped_storage().data_ptr(), self.v_cache.untyped_storage().data_ptr()}
        if any(t.untyped_storage().data_ptr() in pools for t in tensors):
            raise ValueError("append inputs must not alias KV cache storage; clone first")

    @torch.no_grad()
    def append(self, k, v, *, sequence_ids, cu_seqlens=None):
        with self._lock, torch.cuda.device(self.device):
            self._check_stream()
            self._tensor(k, "k", expected_heads=self.num_kv_heads)
            self._tensor(v, "v", expected_heads=self.num_kv_heads, expected_tokens=k.shape[0])
            self._reject_storage_alias(k, v)
            ids, counts = self._append_counts(k.shape[0], sequence_ids, cu_seqlens)
            return self._append_validated(k, v, ids, counts)[0]

    def _append_validated(self, k, v, ids, counts):
        """Internal: caller holds lock/device/stream and validated all inputs."""
        plan = self._layout.plan_append(ids, counts)
        # All metadata allocated before writing invisible new slots. Candidate
        # IDs are now implicit per request; no global candidate list is copied.
        # Keep the large 2-D table on torch's native nested-sequence conversion
        # path. Flattening it in Python erased the H2D savings at large B/L.
        table = self._gpu(plan.tables)
        lengths, token_ids, positions, closed_ids, closed_chunks = self._metadata(
            plan.lengths, plan.token_sequences, plan.token_positions,
            [p[0] for p in plan.completed_chunks], [p[1] for p in plan.completed_chunks])
        candidate_count = self._count_candidates(plan.lengths)
        kernels.append_tokens(k.contiguous(), v.contiguous(), self.k_cache, self.v_cache,
                              table, token_ids, positions)
        kernels.update_closed_means(self.k_cache, table, self._means,
                                    closed_ids, closed_chunks, self.chunk_size)
        self._layout.commit(plan)
        self._table, self._lengths = table, lengths
        self._candidate_count = candidate_count
        return AppendResult(plan.token_sequences, plan.token_positions), token_ids, positions

    def _query_metadata(self, q, sequence_ids, positions):
        self._tensor(q, "q")
        if q.shape[1] % self.num_kv_heads:
            raise ValueError("Hq must be a multiple of Hkv")
        ids = self._layout.validate_ids(_ints(sequence_ids, "sequence_ids"))
        if len(ids) != q.shape[0]:
            raise ValueError("one sequence ID per query row required")
        pos = (tuple(self.lengths[sid] - 1 for sid in ids) if positions is None
               else _ints(positions, "positions"))
        if len(pos) != len(ids) or any(not 0 <= p < self.lengths[s] for s, p in zip(ids, pos)):
            raise ValueError("query positions must refer to cached tokens")
        if q.shape[0] * q.shape[1] >= 2**31:
            raise ValueError("query grid exceeds int32 capacity")
        return ids, pos

    def _route(self, q, ids, positions, topk, max_history):
        nq, hq, _ = q.shape
        local = positions.div(self.chunk_size, rounding_mode="floor")[:, None, None].expand(nq, hq, 1)
        # Preserve v1 result width, including -1 padding when other requests
        # have candidates but these queries have little/no legal history.
        remote = min(int(topk) - 1, self._candidate_count)
        if not remote:
            return local.contiguous()
        width = max(remote, max_history)
        if width * hq * nq >= 2**31:
            raise ValueError("route score grid exceeds int32 capacity")
        scores = torch.empty(nq, hq, width, dtype=torch.float32, device=self.device)
        kernels.route_scores(q, self._means, ids, positions, self.chunk_size, scores)
        values, indices = torch.topk(scores, remote, dim=-1, sorted=False)
        chunks = torch.where(values.isfinite(), indices.to(torch.int32), -1)
        return torch.cat((local, chunks), dim=-1).contiguous()

    @torch.no_grad()
    def attend(self, q, *, sequence_ids, positions=None, topk=4, mode="moba"):
        with self._lock, torch.cuda.device(self.device):
            self._check_stream()
            dense = _mode(mode, topk)
            ids, pos = self._query_metadata(q, sequence_ids, positions)
            return self._attend_validated(q, ids, pos, topk, dense)

    def _attend_validated(self, q, ids, pos, topk, dense, gpu_metadata=None):
        q = q.contiguous()
        gpu_ids, gpu_pos = self._metadata(ids, pos) if gpu_metadata is None else gpu_metadata
        selected = None if dense else self._route(q, gpu_ids, gpu_pos, topk,
                                                 max(pos) // self.chunk_size)
        out = torch.empty_like(q)
        lse = torch.empty(q.shape[:2], device=self.device, dtype=torch.float32)
        kernels.paged_attention(q, self.k_cache, self.v_cache, self._table, self._lengths,
                                gpu_ids, gpu_pos, selected if selected is not None else self._lengths,
                                out, lse, self.chunk_size, dense)
        return DecodeResult(out, lse, selected, ids, pos)

    @torch.no_grad()
    def append_and_attend(self, q, k, v, *, sequence_ids, cu_seqlens=None, topk=4, mode="moba"):
        with self._lock, torch.cuda.device(self.device):
            self._check_stream()
            dense = _mode(mode, topk)
            self._tensor(k, "k", expected_heads=self.num_kv_heads)
            self._tensor(v, "v", expected_heads=self.num_kv_heads, expected_tokens=k.shape[0])
            self._tensor(q, "q", expected_tokens=k.shape[0])
            self._reject_storage_alias(q, k, v)
            if q.shape[1] % self.num_kv_heads:
                raise ValueError("Hq must be a multiple of Hkv")
            if q.shape[0] * q.shape[1] >= 2**31:
                raise ValueError("query grid exceeds int32 capacity")
            ids, counts = self._append_counts(k.shape[0], sequence_ids, cu_seqlens)
            info, gpu_ids, gpu_pos = self._append_validated(k, v, ids, counts)
            return self._attend_validated(q, info.sequence_ids, info.positions, topk, dense,
                                          gpu_metadata=(gpu_ids, gpu_pos))

    @torch.no_grad()
    def reset(self, sequence_ids=None):
        with self._lock, torch.cuda.device(self.device):
            self._check_stream()
            ids = tuple(range(self._layout.batch_size)) if sequence_ids is None else _ints(sequence_ids, "sequence_ids")
            # Work on a temporary CPU layout so metadata allocation errors do not
            # publish a half-reset allocator.
            old = self._layout
            replacement = PageAllocator(old.batch_size, old.max_seq_len, old.page_size,
                                        old.num_pages, old.chunk_size)
            replacement.tables, replacement.lengths, replacement.free_pages = old.tables, old.lengths, old.free_pages
            replacement.version = old.version
            replacement.reset(ids)
            table, lengths = self._gpu(replacement.tables), self._gpu(replacement.lengths)
            candidate_count = self._count_candidates(replacement.lengths)
            self._layout, self._table, self._lengths = replacement, table, lengths
            self._candidate_count = candidate_count

    @torch.no_grad()
    def materialize(self, sequence_id):
        """Diagnostic/reference readback only; not called by attend()."""
        with self._lock, torch.cuda.device(self.device):
            self._check_stream()
            sid = self._layout.validate_ids([sequence_id])[0]
            length = self.lengths[sid]
            pos = torch.arange(length, device=self.device)
            pages = self._table[sid, pos.div(self._layout.page_size, rounding_mode="floor")]
            return (self.k_cache[pages.long(), pos % self._layout.page_size],
                    self.v_cache[pages.long(), pos % self._layout.page_size])


class ContiguousKVCache(PagedKVCache):
    """One contiguous physical slab per sequence, sharing the same semantics."""
    def __init__(self, batch_size, max_seq_len, num_kv_heads, head_dim, **kwargs):
        if "page_size" in kwargs or "num_pages" in kwargs:
            raise ValueError("ContiguousKVCache owns page_size/num_pages")
        super().__init__(batch_size, max_seq_len, num_kv_heads, head_dim,
                         page_size=max_seq_len, num_pages=batch_size, **kwargs)
