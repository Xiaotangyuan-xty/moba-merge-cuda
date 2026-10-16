#!/usr/bin/env python
"""Export the official routing decision for later fixed-route reference checks.

Runs the diagnostic copy (bitwise-identical routing to the frozen baseline) once
with ROUTE_HOOK set, captures the selected block sets, and saves them with the
inputs via torch.save. Test-only tool (plan 7.1): performance runs never replay
fixed routes.

GPU-gated: run only with explicit approval.
"""

import argparse
import itertools
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import moba_opt.diagnostic as diag  # noqa: E402

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def export_route(cfg, device, seed):
    total, heads, dim = cfg["total_tokens"], cfg["heads"], cfg["head_dim"]
    dtype = DTYPES[cfg["dtype"]]
    gen = torch.Generator().manual_seed(seed)

    def mk():
        return torch.randn((total, heads, dim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    q, k, v = mk(), mk(), mk()
    lens = cfg.get("seq_lens") or [total]
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device=device, dtype=torch.int32)
    max_len = max(lens)

    captured = {}

    def hook(cu_chunk, filtered_chunk_indices, chunk_to_batch, selected):
        captured.update(
            cu_chunk=cu_chunk.detach().cpu(),
            filtered_chunk_indices=filtered_chunk_indices.detach().cpu(),
            chunk_to_batch=chunk_to_batch.detach().cpu(),
            selected=selected.detach().cpu(),
        )

    diag.ROUTE_HOOK = hook
    try:
        diag.moba_attn_varlen(q, k, v, cu, max_len, cfg["chunk_size"], cfg["topk"])
    finally:
        diag.ROUTE_HOOK = None
    if not captured:
        raise RuntimeError("route hook never fired (k=1 / fallback branch does not produce a sparse route)")

    route = {
        **captured,
        "cu_seqlens": cu.detach().cpu(),
        "chunk_size": int(cfg["chunk_size"]),
        "topk": int(cfg["topk"]),
        "max_seqlen": int(max_len),
    }
    payload = {
        "route": route,
        "q": q.detach().cpu(),
        "k": k.detach().cpu(),
        "v": v.detach().cpu(),
        "config": {kk: cfg[kk] for kk in ("total_tokens", "heads", "head_dim", "chunk_size", "topk", "dtype")},
        "seed": seed,
    }
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "raw" / "routes"))
    args = ap.parse_args()

    doc = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid in args.ids.split(","):
        cfg = dict(doc.get("defaults", {}))
        cfg.update(doc["configs"][cid])
        payload = export_route(cfg, torch.device(args.device), args.seed)
        out = out_dir / f"{cid}_seed{args.seed}.pt"
        torch.save(payload, out)
        f = payload["route"]["selected"].shape[0]
        m = int(payload["route"]["selected"].sum())
        print(f"{cid}: F={f} selected_rows={m} -> {out}")


if __name__ == "__main__":
    main()
