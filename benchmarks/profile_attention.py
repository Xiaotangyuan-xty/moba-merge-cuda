#!/usr/bin/env python
"""Staged profiling (plan 5.1) with the record_function-marked diagnostic copy.

Per config: warmup, then profile `iters` fwd+bwd iterations under torch.profiler.
Artifacts per config (results/raw/):
  - traces/<id>_<meta>_<ts>.trace.json.gz  (chrome trace, for Nsight-adjacent archive)
  - profile_<id>_<meta>_<ts>.json          (stage cuda/cpu times, sync counters, top kernels)

--metadata warm: same immutable cu_seqlens object every call (lru_cache hits).
--metadata cold: fresh cu_seqlens tensor object every iteration (metadata rebuilt).

Diagnostic profiling runs are NOT formal timing (plan 5.1): no synchronize per
stage, profiler only here, numbers never mixed with benchmark JSONL.
GPU-gated: run only with explicit approval after an nvidia-smi idle check.
"""

import argparse
import gzip
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moba_opt.diagnostic import moba_attn_varlen as moba_diag  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def load_config(path, config_id):
    doc = yaml.safe_load(Path(path).read_text())
    cfg = dict(doc.get("defaults", {}))
    cfg.update(doc["configs"][config_id])
    return cfg


def build_inputs(cfg, device, seed):
    total, heads, dim = cfg["total_tokens"], cfg["heads"], cfg["head_dim"]
    dtype = DTYPES[cfg["dtype"]]
    gen = torch.Generator().manual_seed(seed)

    def mk():
        return torch.randn((total, heads, dim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    q, k, v = mk(), mk(), mk()
    lens = cfg.get("seq_lens") or [total]
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device=device, dtype=torch.int32)
    g = torch.randn((total, heads, dim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    return q, k, v, cu, max(lens), g


def self_device_us(evt):
    return getattr(evt, "self_device_time_total", None) or getattr(evt, "self_cuda_time_total", 0)


def run_profile(cfg, device, iters, warmup, seed, metadata, out_dir, ts):
    q, k, v, cu, max_len, g = build_inputs(cfg, device, seed)
    for t in (q, k, v):
        t.requires_grad_(True)

    def one_iter(cu_arg):
        out = moba_diag(q, k, v, cu_arg, max_len, cfg["chunk_size"], cfg["topk"])
        torch.autograd.grad(out, (q, k, v), grad_outputs=g)

    def cu_for_iter():
        return cu if metadata == "warm" else cu.detach().clone()

    for _ in range(warmup):
        one_iter(cu)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            one_iter(cu_for_iter())
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) * 1e3

    trace_path = out_dir / "traces" / f"{cfg['id']}_{metadata}_{ts}.trace.json.gz"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    raw_trace = trace_path.with_suffix("")
    prof.export_chrome_trace(str(raw_trace))
    with open(raw_trace, "rb") as fi, gzip.open(trace_path, "wb") as fo:
        fo.writelines(fi)
    raw_trace.unlink()

    stages = {}
    syncs = {}
    kernels = []
    for evt in prof.key_averages():
        name = evt.key
        dev_us = self_device_us(evt)
        if name.startswith("stage"):
            stages[name] = {
                "count": evt.count,
                "cuda_total_us": dev_us,
                "cuda_per_iter_us": dev_us / iters,
                "cpu_total_us": evt.cpu_time_total,
                "cpu_per_iter_us": evt.cpu_time_total / iters,
            }
        elif any(s in name for s in ("cudaStreamSynchronize", "cudaMemcpy", "cudaMemset", "aten::nonzero", "Memcpy", "cudaStreamIsCapturing", "aten::item", "aten::_local_scalar_dense")):
            if evt.count or dev_us:
                syncs[name] = {"count": evt.count, "cpu_total_us": evt.cpu_time_total, "cuda_total_us": dev_us}
        if dev_us > 0 and evt.count > 0:
            kernels.append((name, dev_us, evt.count))
    kernels.sort(key=lambda x: -x[1])

    stage_cuda_sum = sum(s["cuda_per_iter_us"] for s in stages.values())
    summary = {
        "config_id": cfg["id"],
        "metadata_mode": metadata,
        "config": {k: cfg[k] for k in ("total_tokens", "heads", "head_dim", "chunk_size", "topk", "dtype")},
        "iters": iters,
        "warmup": warmup,
        "seed": seed,
        "wall_ms_per_iter": wall_ms / iters,
        "stage_cuda_sum_us_per_iter": stage_cuda_sum,
        "stages": dict(sorted(stages.items())),
        "sync_like_events": dict(sorted(syncs.items())),
        "top_kernels_by_self_device_us": [
            {"name": n, "self_device_us_total": d, "count": c} for n, d, c in kernels[:15]
        ],
        "trace_file": str(trace_path),
        "timestamp_utc": ts,
    }
    out_path = out_dir / f"profile_{cfg['id']}_{metadata}_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n== {cfg['id']} metadata={metadata} == wall/iter {wall_ms / iters:.3f} ms; stage cuda sum {stage_cuda_sum / 1e3:.3f} ms")
    for name, s in sorted(stages.items()):
        pct = 100.0 * s["cuda_per_iter_us"] / stage_cuda_sum if stage_cuda_sum else 0
        print(f"  {name:32s} cuda {s['cuda_per_iter_us'] / 1e3:8.3f} ms/iter ({pct:5.1f}%)  cpu {s['cpu_per_iter_us'] / 1e3:8.3f} ms  x{s['count']}")
    print("  sync-like:", {k: v["count"] for k, v in syncs.items()})
    print(f"  wrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--metadata", choices=["warm", "cold"], default="warm")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "raw"))
    ap.add_argument("--verify", action="store_true", help="first check diagnostic == official bitwise on this config")
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cid in args.ids.split(","):
        cfg = load_config(args.config, cid)
        cfg["id"] = cid
        if args.verify:
            from moba.moba_efficient import moba_attn_varlen as moba_ref

            q, k, v, cu, max_len, _ = build_inputs(cfg, device, args.seed)
            o_ref = moba_ref(q, k, v, cu, max_len, cfg["chunk_size"], cfg["topk"])
            o_diag = moba_diag(q, k, v, cu, max_len, cfg["chunk_size"], cfg["topk"])
            same = torch.equal(o_ref, o_diag)
            print(f"verify {cid}: diagnostic == official bitwise: {same}")
            if not same:
                diff = (o_ref.float() - o_diag.float()).abs().max().item()
                print(f"  max abs diff {diff}")
                sys.exit(2)
            del o_ref, o_diag, q, k, v, cu
            torch.cuda.empty_cache()
        run_profile(cfg, device, args.iters, args.warmup, args.seed, args.metadata, out_dir, ts)


if __name__ == "__main__":
    main()
