#!/usr/bin/env python
"""Paired timing harness (plan 8.1-8.4, 9.4): attention fwd and fwd+bwd,
layers 3/4 -- the primary performance evidence.

--impl accepts a comma list (e.g. official,opt_merge_atomic). Both impls run
in the same process on the SAME leaf tensors; sample order alternates AB/BA
per round parity and interleaves per sample (plan 8.2 item 5). autograd.grad
is used (no accumulation). The same immutable cu_seqlens object feeds every
call (warm metadata, plan 8.3), warmed per impl before timing.

Per training-mode sample:  t0 --fwd--> sync t1 --backward--> sync t2
  fwd_wall_ms / bwd_wall_ms / fwd_bwd_wall_ms from perf_counter;
  CUDA events wrap the fwd and bwd regions separately (stream time, may
  include idle gaps -- not "pure kernel time", plan 8.2).
Inference-mode forward (torch.no_grad) measured in separate samples.

One JSONL record per (impl, config, round). OOM/interruption keep a status
record. GPU-gated: run only with explicit approval after an idle check.
"""

import argparse
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

    def mk():
        return torch.randn((total, heads, dim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    q, k, v = mk(), mk(), mk()
    lens = cfg.get("seq_lens") or [total]
    assert all(x > 0 for x in lens) and sum(lens) == total, f"bad seq_lens {lens} vs T={total}"
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device=device, dtype=torch.int32)
    g = torch.randn((total, heads, dim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)
    return q, k, v, cu, max(lens), g


def timed_sample(fwd, q, k, v, g):
    e0, e1, e2, e3 = (torch.cuda.Event(enable_timing=True) for _ in range(4))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    e0.record()
    out = fwd()
    e1.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    e2.record()
    torch.autograd.grad(out, (q, k, v), grad_outputs=g)
    e3.record()
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    del out
    return {
        "fwd_wall_ms": (t1 - t0) * 1e3,
        "bwd_wall_ms": (t2 - t1) * 1e3,
        "fwd_bwd_wall_ms": (t2 - t0) * 1e3,
        "fwd_cuda_event_ms": e0.elapsed_time(e1),
        "bwd_cuda_event_ms": e2.elapsed_time(e3),
    }


def measure(impls, cfg, device, rounds, samples, warmup, seed):
    q, k, v, cu, max_len, g = build_inputs(cfg, device, seed)
    for t in (q, k, v):
        t.requires_grad_(True)

    def fwd(impl):
        # the same immutable cu object is reused everywhere -> warm metadata cache
        return run_impl(impl, q, k, v, cu, max_len, cfg)

    for impl in impls:
        for _ in range(warmup):
            out = fwd(impl)
            torch.autograd.grad(out, (q, k, v), grad_outputs=g)
    torch.cuda.synchronize()

    mem = {}
    for impl in impls:
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        out = fwd(impl)
        torch.autograd.grad(out, (q, k, v), grad_outputs=g)
        torch.cuda.synchronize()
        mem[impl] = {
            "allocated_before_bytes": before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
        del out
        torch.cuda.synchronize()

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
                s = timed_sample(lambda: fwd(impl), q, k, v, g)
                for key, val in s.items():
                    recs[impl][key].append(val)
        with torch.no_grad():
            for _ in range(samples):
                for impl in order:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fwd(impl)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    recs[impl]["fwd_nograd_wall_ms"].append((t1 - t0) * 1e3)
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
            "candidate_commit": None,
            "gpu": torch.cuda.get_device_name(device),
            "device_index": device.index if device.index is not None else torch.cuda.current_device(),
            "config": {k: cfg[k] for k in ("total_tokens", "heads", "head_dim", "chunk_size", "topk", "dtype")},
            "batch": cfg.get("batch", 1),
            "metadata_mode": cfg["metadata_mode"],
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
            records, mem, labels = measure(impls, cfg, device, args.rounds, args.samples, args.warmup, args.seed)
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

        meds = {}
        for impl in impls:
            flat = [x for rec in records[impl] for x in rec["fwd_bwd_wall_ms"]]
            meds[impl] = median(flat)
        if len(impls) == 2:
            sp = meds[impls[0]] / meds[impls[1]]
            print(f"{cid}: {impls[0]} fwdbwd med={meds[impls[0]]:.3f} ms, {impls[1]} med={meds[impls[1]]:.3f} ms, paired speedup={sp:.3f}x")
        else:
            print(f"{cid}: {impls[0]} fwdbwd med={meds[impls[0]]:.3f} ms -> {out_path}")


if __name__ == "__main__":
    main()
