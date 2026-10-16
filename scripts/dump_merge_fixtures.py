#!/usr/bin/env python
"""Dump real merge-stage (stage8) inputs/outputs as fixtures for suboperator
tests (plan 7.2), via MERGE_HOOK on the diagnostic copy.

Fixture payload:
  inputs:   self_out [T,H,D], self_lse [T,H] fp32, remote_out [M,1,D],
            remote_lse [M] fp32, mapping [M] (row p = s*H + h)
  expected: output [T,H,D] (input dtype), mixed_lse [T,H] fp32
  config / seed for traceability
GPU-gated (one diagnostic forward per config on GPU); fixtures are CPU tensors.
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


def dump(cfg, device, seed):
    total, heads, dim = cfg["total_tokens"], cfg["heads"], cfg["head_dim"]
    dtype = DTYPES[cfg["dtype"]]
    gen = torch.Generator().manual_seed(seed)

    def mk():
        return torch.randn((total, heads, dim), generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    q, k, v = mk(), mk(), mk()
    lens = cfg.get("seq_lens") or [total]
    cu = torch.tensor([0] + list(itertools.accumulate(lens)), device=device, dtype=torch.int32)

    got = {}

    def hook(phase, tensors):
        got[phase] = {name: t.detach().to(device="cpu", copy=True) for name, t in tensors.items()}

    diag.MERGE_HOOK = hook
    try:
        diag.moba_attn_varlen(q, k, v, cu, max(lens), cfg["chunk_size"], cfg["topk"])
    finally:
        diag.MERGE_HOOK = None
    assert set(got) == {"inputs", "outputs"}, f"hook phases missing: {sorted(got)}"

    inputs = got["inputs"]
    inputs["remote_lse"] = inputs["remote_lse"].reshape(-1)  # [M,1] -> [M]
    return {
        "inputs": inputs,
        "expected": {"output": got["outputs"]["output"], "mixed_lse": got["outputs"]["mixed_lse"]},
        "config": {kk: cfg[kk] for kk in ("total_tokens", "heads", "head_dim", "chunk_size", "topk", "dtype")},
        "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "raw" / "merge_fixtures"))
    args = ap.parse_args()

    torch.cuda.set_device(torch.device(args.device))
    doc = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cid in args.ids.split(","):
        cfg = dict(doc.get("defaults", {}))
        cfg.update(doc["configs"][cid])
        payload = dump(cfg, torch.device(args.device), args.seed)
        out = out_dir / f"{cid}_seed{args.seed}.pt"
        torch.save(payload, out)
        m = payload["inputs"]["mapping"].numel()
        t, h, d = payload["inputs"]["self_out"].shape
        print(f"{cid}: T={t} H={h} D={d} M={m} -> {out}")


if __name__ == "__main__":
    main()
