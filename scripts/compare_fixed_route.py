#!/usr/bin/env python
"""Compare frozen official MoBA vs layer-3 fixed-route fp64 reference on
exported real routes (plan 7.1). The official path runs on GPU; the reference
is CPU/fp64 and shares no code with it.

Error metrics follow plan 7.4: max/mean abs, relative L2, NaN/Inf flag.
Writes results/raw/fixed_route_check_<ts>.json. GPU-gated.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "third_party" / "MoBA"))

from reference.fixed_route_attention import fixed_route_attention  # noqa: E402


def error_stats(a, b):
    a = a.detach().to(device="cpu", dtype=torch.float64)
    b = b.detach().to(device="cpu", dtype=torch.float64)
    d = (a - b).abs()
    return {
        "max_abs": d.max().item(),
        "mean_abs": d.mean().item(),
        "rel_l2": (d.norm() / b.norm().clamp_min(1e-12)).item(),
        "nan_or_inf": bool(~torch.isfinite(a).all()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True, help="glob of exported route .pt files")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from moba.moba_efficient import moba_attn_varlen

    device = torch.device(args.device)
    results = []
    for path in sorted(Path().glob(args.routes)):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = payload["config"]
        route = payload["route"]
        q = payload["q"].to(device)
        k = payload["k"].to(device)
        v = payload["v"].to(device)
        cu = route["cu_seqlens"].to(device=device, dtype=torch.int32)

        o_official = moba_attn_varlen(
            q, k, v, cu, route["max_seqlen"],
            moba_chunk_size=cfg["chunk_size"], moba_topk=cfg["topk"],
        )
        o_ref, lse_ref = fixed_route_attention(payload["q"], payload["k"], payload["v"], route["cu_seqlens"], route)

        rec = {
            "route_file": str(path),
            "config": cfg,
            "official_vs_reference": error_stats(o_official, o_ref),
            "note": "official is BF16 flash-attn + fp32 merge; reference is fp64 explicit mask. bf16 rounding dominates.",
        }
        results.append(rec)
        e = rec["official_vs_reference"]
        print(f"{Path(path).name}: max_abs={e['max_abs']:.3e} mean_abs={e['mean_abs']:.3e} rel_l2={e['rel_l2']:.3e} nan_inf={e['nan_or_inf']}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = ROOT / "results" / "raw" / f"fixed_route_check_{ts}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
