#!/usr/bin/env python
"""V2 kernel-only / stage micro-benchmark for route A (plan 9.3 V2, 8.1 layers 1-2).

Variants per config:
  official_stage  exact replica of the upstream merge-stage op sequence
  triton_kernel   merge kernel only, prebuilt slots (fixed buffers)
  triton_stage    deterministic torch slots build + merge kernel

Fixed input buffers; CUDA events + perf_counter wall; warmup outside timing.
These numbers judge LOCAL potential only -- they are not full-path claims
(plan 8.1) and must not be reported as attention speedups.
GPU-gated: run only with explicit approval after an nvidia-smi idle check.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import torch
import triton
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from moba_opt.ops.merge_triton import (  # noqa: E402
    _build_slots_atomic_kernel,
    _merge_kernel,
    build_slots_torch,
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def official_merge(self_attn_out_sh, self_attn_lse_hs, moba_attn_out, moba_attn_lse_hs, moba_q_sh_indices):
    """Exact replica of the upstream MixedAttention.forward merge section."""
    self_attn_lse_sh = self_attn_lse_hs.t().contiguous()
    moba_attn_lse = moba_attn_lse_hs.t().contiguous()
    S, H, D = self_attn_out_sh.shape
    output = torch.zeros((S, H, D), device=self_attn_out_sh.device, dtype=torch.float32)
    output_2d = output.view(-1, D)
    max_lse_1d = self_attn_lse_sh.view(-1)
    max_lse_1d = max_lse_1d.index_reduce(
        0, moba_q_sh_indices, moba_attn_lse.view(-1), "amax"
    )
    self_attn_lse_sh = self_attn_lse_sh - max_lse_1d.view_as(self_attn_lse_sh)
    moba_attn_lse = (
        moba_attn_lse.view(-1)
        .sub(max_lse_1d.index_select(0, moba_q_sh_indices))
        .reshape_as(moba_attn_lse)
    )
    mixed_attn_se_sh = self_attn_lse_sh.exp()
    moba_attn_se = moba_attn_lse.exp()
    mixed_attn_se_sh.view(-1).index_add_(
        0, moba_q_sh_indices, moba_attn_se.view(-1)
    )
    mixed_attn_lse_sh = mixed_attn_se_sh.log()
    factor = (self_attn_lse_sh - mixed_attn_lse_sh).exp()
    self_attn_out_sh = self_attn_out_sh * factor.unsqueeze(-1)
    output_2d += self_attn_out_sh.reshape_as(output_2d)
    mixed_attn_lse = (
        mixed_attn_lse_sh.view(-1)
        .index_select(0, moba_q_sh_indices)
        .view_as(moba_attn_lse)
    )
    factor = (moba_attn_lse - mixed_attn_lse).exp()
    moba_attn_out = moba_attn_out * factor.unsqueeze(-1)
    raw_attn_out = moba_attn_out.view(-1, moba_attn_out.shape[-1])
    output_2d.index_add_(0, moba_q_sh_indices, raw_attn_out)
    output = output.to(self_attn_out_sh.dtype)
    mixed_attn_lse_sh = mixed_attn_lse_sh + max_lse_1d.view_as(mixed_attn_se_sh)
    return output, mixed_attn_lse_sh


def synth_merge_inputs(total, heads, dim, chunk, topk, dtype, device, seed=0):
    """Analytic merge-stage inputs; counts match the real packed layout
    (verified against the C01 fixture: M and count distribution identical)."""
    gen = torch.Generator().manual_seed(seed)
    nchunk = (total + chunk - 1) // chunk
    num_filtered = max(0, nchunk - 1)
    tk = min(topk - 1, num_filtered)
    blocks = torch.arange(total, device=device) // chunk
    counts_s = torch.clamp(blocks, max=tk)
    counts_p = counts_s.repeat_interleave(heads)
    mapping = torch.repeat_interleave(torch.arange(total * heads, device=device), counts_p)
    M = mapping.numel()

    def rnd(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale)

    return {
        "self_out": rnd(total, heads, dim).to(device=device, dtype=dtype),
        "self_lse_hs": rnd(heads, total, scale=3.0).to(device=device, dtype=torch.float32),
        "remote_out": rnd(M, 1, dim).to(device=device, dtype=dtype),
        "remote_lse_hs": rnd(1, M, scale=3.0).to(device=device, dtype=torch.float32),
        "mapping": mapping,
        "P": total * heads,
        "kmax": max(tk, 1),
        "M": M,
    }


def time_fn(fn, warmup, samples, rounds):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(rounds):
        evs, walls = [], []
        for _ in range(samples):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            e0.record()
            fn()
            e1.record()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            evs.append(e0.elapsed_time(e1))
            walls.append((t1 - t0) * 1e3)
        out.append({"event_ms_median": median(evs), "wall_ms_median": median(walls)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--samples", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--check-fixture", default=None, help="validate the official_merge replica against a fixture and exit")
    args = ap.parse_args()

    device = torch.device(args.device)

    if args.check_fixture:
        fx = torch.load(args.check_fixture, map_location="cpu", weights_only=False)
        inp = fx["inputs"]
        out, lse = official_merge(
            inp["self_out"].to(device),
            inp["self_lse"].to(device).t().contiguous(),
            inp["remote_out"].to(device),
            inp["remote_lse"].to(device).reshape(1, -1),
            inp["mapping"].to(device),
        )
        d_out = (out.float().cpu() - fx["expected"]["output"].float()).abs()
        d_lse = (lse.cpu() - fx["expected"]["mixed_lse"]).abs()
        print(f"replica vs fixture: out max_abs={d_out.max():.2e} mean={d_out.mean():.2e}; lse max_abs={d_lse.max():.2e}")
        assert d_out.max() < 1e-2 and d_lse.max() < 1e-4
        print("replica OK")
        return

    doc = yaml.safe_load(Path(args.config).read_text())
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for cid in args.ids.split(","):
        cfg = dict(doc.get("defaults", {}))
        cfg.update(doc["configs"][cid])
        total, heads, dim = cfg["total_tokens"], cfg["heads"], cfg["head_dim"]
        chunk, topk = cfg["chunk_size"], cfg["topk"]
        dtype = DTYPES[cfg["dtype"]]
        inp = synth_merge_inputs(total, heads, dim, chunk, topk, dtype, device, args.seed)
        P, D, M, kmax = inp["P"], dim, inp["M"], inp["kmax"]
        kpad = 1 << max(0, (kmax - 1)).bit_length()

        slots_t, counts_t = build_slots_torch(inp["mapping"], P, kmax, kpad=kpad)
        so = inp["self_out"].reshape(P, D).contiguous()
        sl = inp["self_lse_hs"].t().reshape(P).contiguous()
        ro = inp["remote_out"].reshape(M, D).contiguous()
        rl = inp["remote_lse_hs"].reshape(M).contiguous()
        out_buf = torch.empty((P, D), device=device, dtype=dtype)
        lse_buf = torch.empty((P,), device=device, dtype=torch.float32)
        slots_a = torch.empty((P, kpad), dtype=torch.int32, device=device)
        counts_a = torch.zeros(P, dtype=torch.int32, device=device)

        def run_official():
            official_merge(
                inp["self_out"], inp["self_lse_hs"], inp["remote_out"], inp["remote_lse_hs"], inp["mapping"]
            )

        def run_kernel():
            _merge_kernel[(P,)](so, sl, ro, rl, slots_t, counts_t, out_buf, lse_buf, D=D, KPAD=kpad, num_warps=4)

        def run_stage_torch():
            s, c = build_slots_torch(inp["mapping"], P, kmax, kpad=kpad)
            _merge_kernel[(P,)](so, sl, ro, rl, s, c, out_buf, lse_buf, D=D, KPAD=kpad, num_warps=4)

        def run_stage_atomic():
            counts_a.zero_()
            _build_slots_atomic_kernel[(triton.cdiv(M, 256),)](
                inp["mapping"], slots_a, counts_a, M, KPAD=kpad, BLOCK=256
            )
            _merge_kernel[(P,)](so, sl, ro, rl, slots_a, counts_a, out_buf, lse_buf, D=D, KPAD=kpad, num_warps=4)

        variants = {
            "official_stage": run_official,
            "triton_kernel": run_kernel,
            "triton_stage_torchmap": run_stage_torch,
            "triton_stage_atomic": run_stage_atomic,
        }

        for name, fn in variants.items():
            rounds = time_fn(fn, args.warmup, args.samples, args.rounds)
            for r, rec in enumerate(rounds):
                line = {
                    "config_id": cid,
                    "variant": name,
                    "round_id": r,
                    "config": {"total_tokens": total, "heads": heads, "head_dim": dim, "chunk_size": chunk, "topk": topk, "dtype": cfg["dtype"]},
                    "P": P,
                    "M": M,
                    "kmax": kmax,
                    "kpad": kpad,
                    "samples": args.samples,
                    "warmup": args.warmup,
                    "seed": args.seed,
                    "scope": "kernel_or_stage_only_not_full_path",
                    "timestamp_utc": ts,
                    **rec,
                }
                with out_path.open("a") as f:
                    f.write(json.dumps(line) + "\n")
        print(f"{cid}: P={P} M={M} kmax={kmax} -> {out_path}")


if __name__ == "__main__":
    main()
