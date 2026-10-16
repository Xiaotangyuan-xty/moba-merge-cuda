import json
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
recs = [json.loads(line) for line in open(path) if line.strip()]
by = defaultdict(list)
for r in recs:
    by[r["config_id"]].append(r)
print(f"file: {path}")
print(f"total lines: {len(recs)}")
for cid in sorted(by):
    rs = by[cid]
    statuses = {r["status"] for r in rs}
    fwd = [statistics.median(r["fwd_wall_ms"]) for r in rs]
    bwd = [statistics.median(r["bwd_wall_ms"]) for r in rs]
    fb = [statistics.median(r["fwd_bwd_wall_ms"]) for r in rs]
    evf = [statistics.median(r["fwd_cuda_event_ms"]) for r in rs]
    ng = [statistics.median(r["fwd_nograd_wall_ms"]) for r in rs]
    peak = rs[0].get("peak_allocated_bytes")
    peak_mib = peak / 2**20 if peak else None
    print(cid, f"status={statuses}", f"rounds={len(rs)}")
    print(f"  fwd_wall     med={statistics.median(fwd):.3f} ms  round_spread={min(fwd):.3f}..{max(fwd):.3f}")
    print(f"  bwd_wall     med={statistics.median(bwd):.3f} ms  round_spread={min(bwd):.3f}..{max(bwd):.3f}")
    print(f"  fwd+bwd_wall med={statistics.median(fb):.3f} ms  round_spread={min(fb):.3f}..{max(fb):.3f}")
    print(f"  fwd_event    med={statistics.median(evf):.3f} ms")
    print(f"  fwd_nograd   med={statistics.median(ng):.3f} ms")
    print(f"  peak_mem     {peak_mib:.0f} MiB" if peak_mib else "  peak_mem     n/a")
