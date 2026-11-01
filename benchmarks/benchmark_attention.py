#!/usr/bin/env python
"""Paired timing harness (plan 8.1-8.4, 9.4): attention fwd and fwd+bwd,
layers 3/4 -- the primary performance evidence.

--impl accepts a comma list (e.g. official,opt_merge_cuda). Both impls run in
the same process on the SAME leaf tensors; sample order alternates AB/BA per
round parity and interleaves per sample (plan 8.2 item 5). autograd.grad is
used (no accumulation). The same immutable cu_seqlens object feeds every call
(warm metadata, plan 8.3), warmed per impl before timing.

Per training-mode sample:  t0 --fwd--> sync t1 --backward--> sync t2
  fwd_wall_ms / bwd_wall_ms / fwd_bwd_wall_ms from perf_counter;
  CUDA events wrap the fwd and bwd regions separately (stream time, may
  include idle gaps -- not "pure kernel time", plan 8.2).
Inference-mode forward (torch.no_grad) measured in separate samples.

Statistics: per-round medians, then the median across rounds (the same
statistic summarize_results.py uses). All device work runs under an explicit
torch.cuda.device context; sync/events/memory queries are device-qualified.

One JSONL record per (impl, config, round). OOM/interruption keep a status
record. GPU-gated: run only with explicit approval after an idle check.
"""

import argparse
import itertools
import json
import sys
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import moba_opt  # noqa: F401,E402  (registers impls)
from moba_opt.api import run_impl  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def load_config(path, config_id):
    doc = yaml.safe_load(Path(path).read_text())
    cfg = dict(doc.get("defaults", {}))
    cfg.update(doc["configs"][config_id])
    cfg.setdefault("metadata_mode", "warm_immutable_tensor")
    return cfg


def build_inputs(cfg, device, seed):
    total, heads, dim = cfg["total_tokens"], cfg["heads"], cfg["head_dim"]
    dtype = DTYPES[cfg["dtype"]]
    gen = torch.Generator().manual_seed(seed)

    kv_heads = cfg.get("kv_heads", heads)

    def mk(n_heads):
        return torch.randn((total, n_heads, dim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    q, k, v = mk(heads), mk(kv_heads), mk(kv_heads)
    lens = cfg.get("seq_lens") or [total]
    assert all(x > 0 for x in lens) and sum(lens) == total, f"bad seq_lens {lens} vs T={total}"
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device=device, dtype=torch.int32)
    g = torch.randn((total, heads, dim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    return q, k, v, cu, max(lens), g


def timed_sample(fwd, q, k, v, g, device):
    e0, e1, e2, e3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    e0.record()
    out = fwd()
    e1.record()
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    e2.record()
    torch.autograd.grad(out, (q, k, v), grad_outputs=g)
    e3.record()
    torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    del out
    return {
        "fwd_wall_ms": (t1 - t0) * 1e3,
        "bwd_wall_ms": (t2 - t1) * 1e3,
        "fwd_bwd_wall_ms": (t2 - t0) * 1e3,
        "fwd_cuda_event_ms": e0.elapsed_time(e1),
        "bwd_cuda_event_ms": e2.elapsed_time(e3),
    }


def measure(impls, cfg, device, rounds, samples, warmup, seed, metadata="warm"):
    with torch.cuda.device(device):
        q, k, v, cu, max_len, g = build_inputs(cfg, device, seed)
        for t in (q, k, v):
            t.requires_grad_(True)

        cu_holder = {"cu": cu}

        def fwd(impl):
            # warm: the same immutable cu object is reused everywhere -> warm cache.
            # cold: caller swaps in a fresh tensor object (same values) per call ->
            # metadata rebuilt every time (plan 8.3 dynamic-metadata scenario).
            return run_impl(impl, q, k, v, cu_holder["cu"], max_len, cfg)

        def maybe_cold():
            if metadata == "cold":
                cu_holder["cu"] = cu.clone()

        out = None
        for impl in impls:
            for _ in range(warmup):
                maybe_cold()
                out = fwd(impl)
                torch.autograd.grad(out, (q, k, v), grad_outputs=g)
        cu_holder["cu"] = cu
        del out  # drop the last warmup output before any memory measurement
        torch.cuda.synchronize(device)

        mem = {}
        for impl in impls:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
            out = fwd(impl)
            torch.autograd.grad(out, (q, k, v), grad_outputs=g)
            torch.cuda.synchronize(device)
            mem[impl] = {
                "allocated_before_bytes": before,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
            }
            del out
            torch.cuda.synchronize(device)

        labels = {impl: chr(ord("A") + i) for i, impl in enumerate(impls)}
        records = {impl: [] for impl in impls}
        for r in range(rounds):
            order = impls if r % 2 == 0 else list(reversed(impls))
            order_str = "".join(labels[i] for i in order)
            recs = {
                impl: {
                    "round_id": r,
                    "round_start_impl": labels[order[0]],
                    "order": order_str,
                    "fwd_wall_ms": [],
                    "bwd_wall_ms": [],
                    "fwd_bwd_wall_ms": [],
                    "fwd_cuda_event_ms": [],
                    "bwd_cuda_event_ms": [],
                    "fwd_nograd_wall_ms": [],
                }
                for impl in impls
            }
            for _ in range(samples):
                for impl in order:
                    maybe_cold()
                    s = timed_sample(lambda: fwd(impl), q, k, v, g, device)
                    for key, val in s.items():
                        recs[impl][key].append(val)
            with torch.no_grad():
                for _ in range(samples):
                    for impl in order:
                        maybe_cold()
                        torch.cuda.synchronize(device)
                        t0 = time.perf_counter()
                        fwd(impl)
                        torch.cuda.synchronize(device)
                        t1 = time.perf_counter()
                        recs[impl]["fwd_nograd_wall_ms"].append((t1 - t0) * 1e3)
            cu_holder["cu"] = cu
            for impl in impls:
                records[impl].append(recs[impl])
        return records, mem, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", required=True, help="comma separated config ids")
    ap.add_argument("--impl", default="official", help="comma separated impls, paired in order")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metadata", choices=["warm", "cold"], default="warm")
    args = ap.parse_args()

    impls = args.impl.split(",")
    device = torch.device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    upstream = (ROOT / "third_party" / "UPSTREAM_SHA.txt").read_text().strip()

    for cid in args.ids.split(","):
        cfg = load_config(args.config, cid)
        base = {
            "status": "ok",
            "config_id": cid,
            "upstream_commit": upstream,
            "candidate_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
            "working_tree_dirty": bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip()),
            "gpu": torch.cuda.get_device_name(device),
            "device_index": device.index if device.index is not None else torch.cuda.current_device(),
            "config": {**{k: cfg[k] for k in ("total_tokens", "heads", "head_dim", "chunk_size", "topk", "dtype")},
                       "kv_heads": cfg.get("kv_heads",cfg["heads"])},
            "seq_lens": cfg.get("seq_lens"),
            "batch": len(cfg.get("seq_lens") or [cfg["total_tokens"]]),
            "metadata_mode": "cold_fresh_object_per_call" if args.metadata == "cold" else cfg["metadata_mode"],
            "seed": args.seed,
            "rounds": args.rounds,
            "samples": args.samples,
            "warmup": args.warmup,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "route_match": None,
            "output_error": None,
            "gradient_error": None,
            "workspace_bytes": None,
        }
        try:
            records, mem, labels = measure(impls, cfg, device, args.rounds, args.samples, args.warmup, args.seed, metadata=args.metadata)
        except torch.cuda.OutOfMemoryError:
            line = dict(base)
            line["status"] = "oom"
            with out_path.open("a") as f:
                f.write(json.dumps(line) + "\n")
            print(f"{cid}: OOM (recorded)")
            torch.cuda.empty_cache()
            continue
        except KeyboardInterrupt:
            line = dict(base)
            line["status"] = "interrupted"
            with out_path.open("a") as f:
                f.write(json.dumps(line) + "\n")
            raise
        base["impl_labels"] = labels
        for impl in impls:
            for rec in records[impl]:
                line = dict(base)
                line.update(mem[impl])
                line.update(rec)
                line["impl"] = impl
                with out_path.open("a") as f:
                    f.write(json.dumps(line) + "\n")
        from statistics import median

        # same statistic as summarize_results.py: median of per-round medians
        meds = {}
        for impl in impls:
            meds[impl] = median(median(rec["fwd_bwd_wall_ms"]) for rec in records[impl])
        if len(impls) == 2:
            sp = meds[impls[0]] / meds[impls[1]]
            print(f"{cid}: {impls[0]} fwdbwd med={meds[impls[0]]:.3f} ms, {impls[1]} med={meds[impls[1]]:.3f} ms, paired speedup={sp:.3f}x")
        else:
            print(f"{cid}: {impls[0]} fwdbwd med={meds[impls[0]]:.3f} ms -> {out_path}")


if __name__ == "__main__":
    main()
