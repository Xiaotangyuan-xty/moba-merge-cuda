#!/usr/bin/env python
"""Baseline / paired timing harness (plan sections 8.1-8.4, 9.4).

Per training-mode sample:  t0 --fwd--> sync t1 --backward--> sync t2
  fwd_wall_ms / bwd_wall_ms / fwd_bwd_wall_ms from perf_counter;
  CUDA events wrap the fwd and bwd regions separately (stream time, may
  include idle gaps -- not "pure kernel time", see plan 8.2).
Inference-mode forward (torch.no_grad) measured in separate samples.

One JSONL record per (impl, config, round) plus a per-config memory record.
GPU-gated: run only with explicit approval after an nvidia-smi idle check.
On external interruption / OOM the partial record is kept with a status flag.
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


def measure(impl, cfg, device, rounds, samples, warmup, seed):
    q, k, v, cu, max_len, g = build_inputs(cfg, device, seed)
    for t in (q, k, v):
        t.requires_grad_(True)

    def fwd():
        # the same immutable cu object is reused every call -> warm metadata cache
        return run_impl(impl, q, k, v, cu, max_len, cfg)

    for _ in range(warmup):
        out = fwd()
        torch.autograd.grad(out, (q, k, v), grad_outputs=g)
    torch.cuda.synchronize()

    # memory: dedicated single fwd+bwd after warmup (plan 8.5)
    torch.cuda.reset_peak_memory_stats()
    alloc_before = torch.cuda.memory_allocated()
    out = fwd()
    torch.autograd.grad(out, (q, k, v), grad_outputs=g)
    torch.cuda.synchronize()
    mem = {
        "allocated_before_bytes": alloc_before,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
    }
    del out
    torch.cuda.synchronize()

    records = []
    for r in range(rounds):
        rec = {
            "round_id": r,
            "fwd_wall_ms": [],
            "bwd_wall_ms": [],
            "fwd_bwd_wall_ms": [],
            "fwd_cuda_event_ms": [],
            "bwd_cuda_event_ms": [],
            "fwd_nograd_wall_ms": [],
        }
        for _ in range(samples):
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
            rec["fwd_wall_ms"].append((t1 - t0) * 1e3)
            rec["bwd_wall_ms"].append((t2 - t1) * 1e3)
            rec["fwd_bwd_wall_ms"].append((t2 - t0) * 1e3)
            rec["fwd_cuda_event_ms"].append(e0.elapsed_time(e1))
            rec["bwd_cuda_event_ms"].append(e2.elapsed_time(e3))
            del out
        with torch.no_grad():
            for _ in range(samples):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fwd()
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                rec["fwd_nograd_wall_ms"].append((t1 - t0) * 1e3)
        records.append(rec)
    return records, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", required=True, help="comma separated config ids")
    ap.add_argument("--impl", default="official")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device(args.device)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    upstream = (ROOT / "third_party" / "UPSTREAM_SHA.txt").read_text().strip()

    for cid in args.ids.split(","):
        cfg = load_config(args.config, cid)
        base = {
            "status": "ok",
            "config_id": cid,
            "impl": args.impl,
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
            "order": "official_only",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "route_match": None,
            "output_error": None,
            "gradient_error": None,
            "workspace_bytes": None,
        }
        try:
            records, mem = measure(args.impl, cfg, device, args.rounds, args.samples, args.warmup, args.seed)
        except torch.cuda.OutOfMemoryError:
            base["status"] = "oom"
            with out_path.open("a") as f:
                f.write(json.dumps(base) + "\n")
            print(f"{cid}: OOM (recorded)")
            torch.cuda.empty_cache()
            continue
        except KeyboardInterrupt:
            base["status"] = "interrupted"
            with out_path.open("a") as f:
                f.write(json.dumps(base) + "\n")
            raise
        base.update(mem)
        for rec in records:
            line = dict(base)
            line.update(rec)
            with out_path.open("a") as f:
                f.write(json.dumps(line) + "\n")
        med = sorted(records[0]["fwd_bwd_wall_ms"])[args.samples // 2]
        print(f"{cid}: round0 fwd+bwd median ~{med:.3f} ms -> {out_path}")


if __name__ == "__main__":
    main()
