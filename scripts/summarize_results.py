#!/usr/bin/env python
"""Summarize paired benchmark JSONL into a markdown table + CSV (plan 9.4:
summaries are generated from raw records, never hand-copied).

Usage: python scripts/summarize_results.py results/raw/paired_core12_v3.jsonl [...]
Writes results/summaries/summary_<ts>.md and .csv next to the inputs.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("fwd_wall_ms", "bwd_wall_ms", "fwd_bwd_wall_ms", "fwd_nograd_wall_ms")


def load(paths):
    rows = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            key = (r["config_id"], r["impl"], r.get("metadata_mode"), Path(path).name)
            rows.setdefault(key, []).append(r)
    return rows


def main():
    rows = load(sys.argv[1:])
    configs = {}
    for (cid, impl, meta, fname), rs in rows.items():
        stats = {}
        for k in KEYS:
            stats[k] = median([median(r[k]) for r in rs])
        peak = rs[0].get("peak_allocated_bytes")
        configs.setdefault((cid, meta, fname), {})[impl] = {**stats, "peak_mib": (peak or 0) / 2**20}

    lines = []
    header = "| config | metadata | impl A fwd+bwd (ms) | impl B fwd+bwd (ms) | fwd speedup | bwd speedup | fwd+bwd speedup | fwd(nograd) speedup |"
    lines.append(header)
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    csv_lines = ["config_id,metadata,file,impl_a,impl_b,fwd_ms_a,fwdbwd_ms_a,fwdbwd_ms_b,sp_fwd,sp_bwd,sp_fwdbwd,sp_fwd_nograd,peak_mib_a,peak_mib_b"]
    speedups = []
    for (cid, meta, fname), impls in sorted(configs.items()):
        if len(impls) != 2:
            continue
        a, b = list(impls)
        sa, sb = impls[a], impls[b]
        sp = {k: sa[k] / sb[k] for k in KEYS}
        speedups.append(sp["fwd_bwd_wall_ms"])
        lines.append(
            f"| {cid} | {meta} | {sa['fwd_bwd_wall_ms']:.3f} ({a}) | {sb['fwd_bwd_wall_ms']:.3f} ({b}) "
            f"| {sp['fwd_wall_ms']:.3f} | {sp['bwd_wall_ms']:.3f} | **{sp['fwd_bwd_wall_ms']:.3f}** | {sp['fwd_nograd_wall_ms']:.3f} |"
        )
        csv_lines.append(
            f"{cid},{meta},{fname},{a},{b},{sa['fwd_wall_ms']:.4f},{sa['fwd_bwd_wall_ms']:.4f},{sb['fwd_bwd_wall_ms']:.4f},"
            f"{sp['fwd_wall_ms']:.4f},{sp['bwd_wall_ms']:.4f},{sp['fwd_bwd_wall_ms']:.4f},{sp['fwd_nograd_wall_ms']:.4f},"
            f"{sa['peak_mib']:.0f},{sb['peak_mib']:.0f}"
        )
    if speedups:
        geo = 1.0
        for s in speedups:
            geo *= s
        geo **= 1.0 / len(speedups)
        lines.append(f"\nfwd+bwd speedup geometric mean over {len(speedups)} paired rows: **{geo:.4f}x**; min {min(speedups):.3f}, max {max(speedups):.3f}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "results" / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"summary_{ts}.md"
    md.write_text("\n".join(lines) + "\n")
    (out_dir / f"summary_{ts}.csv").write_text("\n".join(csv_lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {md}")


if __name__ == "__main__":
    main()
