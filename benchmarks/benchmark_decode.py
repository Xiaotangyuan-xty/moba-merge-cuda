"""Paired cached-decode timing. Run only inside an approved GPU batch.

Primary comparison: frozen add0a49 cache vs current cache, identical inputs.
Fixed-route gather+FA measures attention only; dense FA is a distinct algorithm.
CUDA events span stream idle gaps and are NOT pure summed kernel time.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from unittest.mock import patch

import torch
from flash_attn import flash_attn_func, flash_attn_with_kvcache

from moba_opt import PagedKVCache
from moba_opt.ops import paged_decode
from ._decode_v1 import PagedKVCache as FrozenCache
from . import _decode_v1_kernels

ROOT = Path(__file__).resolve().parents[1]
BASELINE_SHA = "add0a49fdccfca52e5873fbdad0893e96777cf89"


def timed(fn):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    start.record()
    result = fn()
    end.record()
    end.synchronize()
    wall = (time.perf_counter() - t0) * 1000
    event = start.elapsed_time(end)
    del result
    return {"wall_ms": wall, "stream_ms": event}


def paired(functions, rounds, samples, warmup):
    for _ in range(warmup):
        for fn in functions.values():
            fn()
    torch.cuda.synchronize()
    records = []
    names = list(functions)
    for r in range(rounds):
        for s in range(samples):
            order = names if (r + s) % 2 == 0 else names[::-1]
            for name in order:
                records.append({"round": r, "sample": s, "impl": name,
                                "order": order, **timed(functions[name])})
    return records


def stat(records):
    result = {}
    for name in dict.fromkeys(x["impl"] for x in records):
        result[name] = {}
        for metric in ("wall_ms", "stream_ms"):
            by_round = [statistics.median(x[metric] for x in records
                        if x["impl"] == name and x["round"] == r)
                        for r in sorted({x["round"] for x in records})]
            result[name][metric] = statistics.median(by_round)
    return result


def fixed_route_gather_fa(cache, q, ids, selected):
    """Full selected blocks: direct GPU gather then FA, including gather cost.

    Harness checks full blocks outside timing. Each query-head becomes a
    one-head FA batch item. No padding sort, length creation or route readback.
    """
    nq, hq, d = q.shape
    c = cache.chunk_size
    token = selected[..., None].long() * c + torch.arange(c, device=q.device)
    token = token.flatten(2)
    page = cache._table[ids[:, None, None].long(), token // cache.k_cache.shape[1]]
    kh = (torch.arange(hq, device=q.device) // (hq // cache.num_kv_heads))[None, :, None]
    keys = cache.k_cache[page.long(), token % cache.k_cache.shape[1], kh]
    values = cache.v_cache[page.long(), token % cache.k_cache.shape[1], kh]
    out = flash_attn_func(q.reshape(nq*hq, 1, 1, d),
                         keys.reshape(nq*hq, -1, 1, d),
                         values.reshape(nq*hq, -1, 1, d),causal=False)
    return out.reshape_as(q)


def _routes(result):
    return torch.sort(result.selected_blocks, dim=-1).values


def validate_pair(a, b):
    assert torch.equal(_routes(a), _routes(b)), "route mismatch; timing aborted"
    tol=0.02 if a.output.dtype==torch.bfloat16 else 0.002
    torch.testing.assert_close(a.output, b.output, atol=tol, rtol=tol)
    torch.testing.assert_close(a.lse, b.lse, atol=5e-4, rtol=2e-5)


def fixture(cfg, extra, seed):
    b, length, hk, d = (cfg[k] for k in ("batch", "length", "kv_heads", "dim"))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[cfg["dtype"]]
    generator = torch.Generator(device="cuda").manual_seed(seed)
    def rand(shape):
        return torch.randn(shape, generator=generator, device="cuda", dtype=dtype)
    args = (b, length+extra, hk, d)
    kwargs = dict(page_size=cfg["page"], chunk_size=cfg["chunk"], dtype=dtype)
    caches = {"v1": FrozenCache(*args, **kwargs), "current": PagedKVCache(*args, **kwargs)}
    ids = list(range(b))
    # Bound setup workspace too: full input tensors plus isfinite temporaries
    # can otherwise exceed the shared-node ceiling before timing even begins.
    for sid in ids:
        for lo in range(0,length,2048):
            n=min(2048,length-lo)
            k,v=rand((n,hk,d)),rand((n,hk,d))
            for cache in caches.values():
                cache.append(k,v,sequence_ids=[sid],cu_seqlens=[0,n])
            del k,v
    q = rand((b, cfg["q_heads"], d))
    kn, vn = rand((b, hk, d)), rand((b, hk, d))
    return caches, q, kn, vn, ids


def profile(cache, q, ids, path):
    """Scoped benchmark-only labels; production dispatch is unchanged."""
    module = _decode_v1_kernels if isinstance(cache, FrozenCache) else paged_decode
    def labelled(fn, label):
        def call(*args, **kwargs):
            with torch.profiler.record_function(label):
                return fn(*args, **kwargs)
        return call
    with ExitStack() as stack:
        for name in ("_tensor", "_gpu", "_route"):
            stack.enter_context(patch.object(cache, name, labelled(getattr(cache, name), "decode/"+name)))
        stack.enter_context(patch.object(module, "paged_attention",
                            labelled(module.paged_attention, "decode/attention")))
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                              torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(3):
                with torch.profiler.record_function("decode/attend"):
                    cache.attend(q, sequence_ids=ids, topk=4)
            torch.cuda.synchronize()
    p.export_chrome_trace(str(path))
    labels = [{"name":x.key,"count":x.count,"cpu_total_us":x.cpu_time_total}
              for x in p.key_averages() if x.key.startswith("decode/") and x.cpu_time_total>0]
    kernels = {}
    for event in json.loads(path.read_text())["traceEvents"]:
        if event.get("cat")!="kernel":continue
        row=kernels.setdefault(event["name"],{"count":0,"total_us":0})
        row["count"]+=1;row["total_us"]+=event.get("dur",0)
    return {"labels_cpu":labels,"kernels":kernels,
            "note":"profiler is diagnostic; primary latency comes from uninstrumented paired runs"}


def measure(cfg, args, out_dir):
    caches, q, kn, vn, ids = fixture(cfg, 0, args.seed)
    outputs = {n: c.attend(q, sequence_ids=ids, topk=cfg["topk"]) for n,c in caches.items()}
    validate_pair(outputs["v1"], outputs["current"])
    funcs = {n: (lambda c=c: c.attend(q, sequence_ids=ids, topk=cfg["topk"]))
             for n,c in caches.items()}
    attend = paired(funcs, args.rounds, args.samples, args.warmup)
    current = caches["current"]
    gpu_ids = torch.tensor(ids, dtype=torch.int32, device="cuda")
    positions = torch.full_like(gpu_ids, cfg["length"]-1)
    selected = outputs["current"].selected_blocks
    assert bool((selected>=0).all()) and cfg["length"]%cfg["chunk"]==0
    stage_funcs = {"direct_paged": lambda: paged_decode.paged_attention(
        q, current.k_cache, current.v_cache, current._table, current._lengths,
        gpu_ids, positions, selected, torch.empty_like(q), torch.empty(q.shape[:2],device="cuda"),
        cfg["chunk"], False),
        "gather_fa": lambda: fixed_route_gather_fa(current,q,gpu_ids,selected)}
    gathered = stage_funcs["gather_fa"]()
    tol=0.02 if q.dtype==torch.bfloat16 else 0.002
    torch.testing.assert_close(gathered, outputs["current"].output, atol=tol, rtol=tol)
    attention = paired(stage_funcs, args.rounds, args.samples, args.warmup)
    # Equal-length fixture was allocated in sequence order. These views are
    # dense logical KV without materializing another full cache.
    assert cfg["length"] % cfg["page"] == 0
    dense_k = current.k_cache.view(cfg["batch"], cfg["length"],cfg["kv_heads"],cfg["dim"])
    dense_v = current.v_cache.view_as(dense_k)
    dense_lens = torch.full((cfg["batch"],),cfg["length"],device="cuda",dtype=torch.int32)
    dense_fn = lambda: flash_attn_with_kvcache(q[:,None],dense_k,dense_v,
                                             cache_seqlens=dense_lens,causal=True)
    torch.testing.assert_close(dense_fn()[:,0],current.attend(q,sequence_ids=ids,mode="dense").output,
                               atol=tol,rtol=tol)
    dense = paired({"dense_fa_distinct_algorithm":dense_fn},args.rounds,args.samples,args.warmup)
    profiles = {}
    cache = None
    if args.profile:
        for name,cache in caches.items():
            path = out_dir/(cfg["id"]+"_"+name+"_trace.json")
            profiles[name] = profile(cache,q,ids,path)
    del outputs, funcs, stage_funcs, gathered, selected, current, caches, cache
    # Explicitly release closures/views before allocating growing append caches.
    del dense_k, dense_v, dense_fn
    gc.collect(); torch.cuda.empty_cache()
    extra = args.warmup + args.rounds*args.samples + 2
    caches,q,kn,vn,ids = fixture(cfg,extra,args.seed)
    before = {n:c.append_and_attend(q,kn,vn,sequence_ids=ids,topk=cfg["topk"])
              for n,c in caches.items()}
    validate_pair(before["v1"],before["current"])
    append_funcs = {n:(lambda c=c:c.append_and_attend(q,kn,vn,sequence_ids=ids,topk=cfg["topk"]))
                    for n,c in caches.items()}
    append = paired(append_funcs,args.rounds,args.samples,args.warmup)
    validate_pair(caches["v1"].attend(q,sequence_ids=ids,topk=cfg["topk"]),
                  caches["current"].attend(q,sequence_ids=ids,topk=cfg["topk"]))
    return {"attend":attend,"append_and_attend":append,"fixed_route_attention":attention,
            "dense_reference":dense,"profiles":profiles,
            "summaries":{key:stat(value) for key,value in (
                ("attend",attend),("append_and_attend",append),
                ("fixed_route_attention",attention),("dense_reference",dense))}}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/decode_perf.json")
    p.add_argument("--ids",default="all")
    p.add_argument("--out",required=True)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--samples",type=int,default=10)
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--seed",type=int,default=401)
    p.add_argument("--profile",action="store_true")
    args=p.parse_args()
    if min(args.rounds,args.samples,args.warmup)<1:raise SystemExit("positive counts required")
    # Leave room for CUDA context and library allocations below the monitor's
    # device-wide limit. This bounds PyTorch's caching allocator independently.
    allocator_mib=min(6144,int(os.environ.get("MOBA_BATCH_LIMIT_MIB","8192"))-1024)
    if allocator_mib<=0:raise SystemExit("insufficient approved memory budget")
    torch.cuda.set_per_process_memory_fraction(allocator_mib*1024**2/
                                               torch.cuda.get_device_properties(0).total_memory)
    out=Path(args.out)
    if out.exists():raise SystemExit("Output exists; choose a new path")
    out.parent.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path(args.config).read_text())
    ids=None if args.ids=="all" else set(args.ids.split(","))
    if ids and ids-set(c["id"] for c in config):raise SystemExit("unknown config IDs")
    trace_dir=out.parent/(out.stem+"_traces")
    if args.profile:trace_dir.mkdir(exist_ok=False)
    for cfg in config:
        if ids and cfg["id"] not in ids:continue
        base={"config":cfg,"baseline_sha":BASELINE_SHA,"seed":args.seed,
              "rounds":args.rounds,"samples":args.samples,"warmup":args.warmup,
              "timestamp_utc":datetime.now(timezone.utc).isoformat(),
              "git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "working_tree_dirty":bool(subprocess.check_output(["git","status","--porcelain"],text=True).strip()),
            "config_sha256":hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
              "harness_version":2,
              "fixed_route_baseline":"full-block gather + flash_attn_func, no padding sort",
              "torch":torch.__version__,"gpu":torch.cuda.get_device_name(),
              "metric":"median of round medians; stream events include host idle gaps"}
        try:
            with torch.inference_mode():base.update(status="ok",**measure(cfg,args,trace_dir))
        except Exception as exc:
            base.update(status="failed",error=repr(exc))
            with out.open("a") as f:f.write(json.dumps(base)+"\n")
            raise
        with out.open("a") as f:f.write(json.dumps(base)+"\n")
        print(json.dumps({"id":cfg["id"],"summary":base["summaries"]}),flush=True)
        gc.collect();torch.cuda.empty_cache()


if __name__=="__main__":main()
