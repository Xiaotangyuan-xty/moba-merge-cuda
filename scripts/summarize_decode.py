#!/usr/bin/env python3
"""CPU-only decode report. Retain every configured row, including regressions."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics


def metrics(records, names, rounds, samples):
    expected={(r,s,n) for r in range(rounds) for s in range(samples) for n in names}
    actual=[(x["round"],x["sample"],x["impl"]) for x in records]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError("missing/duplicate/unexpected paired samples")
    result={}
    for name in names:
        medians={}
        for clock in ("wall_ms","stream_ms"):
            if any(not math.isfinite(x[clock]) or x[clock]<=0 for x in records):
                raise ValueError("non-positive/non-finite sample")
            medians[clock]=[statistics.median(x[clock] for x in records
                if x["impl"]==name and x["round"]==r) for r in range(rounds)]
        result[name]=medians
    return result


def summarize(path, configs):
    raw=[json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    by_id={x["config"]["id"]:x for x in raw}
    if len(raw)!=len(by_id) or set(by_id)!=set(configs):
        raise ValueError(f"{path}: expected all {len(configs)} configs exactly once")
    heads={x["git_head"] for x in raw}
    if len(heads)!=1:raise ValueError("mixed implementation commits")
    rows=[]
    for cid,cfg in configs.items():
        x=by_id[cid]
        if x["status"]!="ok" or x["config"]!=cfg or x["working_tree_dirty"]:
            raise ValueError(f"{cid}: failed/modified config or dirty source")
        row={"run":path.stem,**cfg,"git_head":x["git_head"],"seed":x["seed"]}
        for workload in ("attend","append_and_attend"):
            m=metrics(x[workload],["v1","current"],x["rounds"],x["samples"])
            for clock in ("wall_ms","stream_ms"):
                for name in ("v1","current"):
                    row[f"{workload}_{name}_{clock}"]=statistics.median(m[name][clock])
                ratios=[a/b for a,b in zip(m["v1"][clock],m["current"][clock])]
                row[f"{workload}_{clock}_speedup"]=statistics.median(ratios)
        a=metrics(x["fixed_route_attention"],["direct_paged","gather_fa"],x["rounds"],x["samples"])
        for name in a:
            row[name+"_wall_ms"]=statistics.median(a[name]["wall_ms"])
        d=metrics(x["dense_reference"],["dense_fa_distinct_algorithm"],x["rounds"],x["samples"])
        row["dense_fa_wall_ms"]=statistics.median(d["dense_fa_distinct_algorithm"]["wall_ms"])
        rows.append(row)
    aggregate={"input":path.name,"git_head":next(iter(heads)),"config_count":len(rows),
               "all_config_ids":list(configs),"workloads":{}}
    for workload in ("attend","append_and_attend"):
        values=[r[f"{workload}_wall_ms_speedup"] for r in rows]
        aggregate["workloads"][workload]={"min_speedup":min(values),"max_speedup":max(values),
            "geomean_speedup":math.exp(statistics.mean(math.log(v) for v in values)),
            "regression_ids":[r["id"] for r in rows if r[f"{workload}_wall_ms_speedup"]<1]}
    return rows,aggregate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("inputs",nargs="+")
    p.add_argument("--config",default="configs/decode_perf.json")
    p.add_argument("--out-dir",required=True)
    args=p.parse_args()
    configs={x["id"]:x for x in json.loads(Path(args.config).read_text())}
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    if any((out/n).exists() for n in ("all_rows.csv","summary.json")):
        raise SystemExit("summary already exists; preserve previous evidence")
    rows,runs=[],[]
    for path in map(Path,args.inputs):
        r,s=summarize(path,configs);rows.extend(r);runs.append(s)
    with (out/"all_rows.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={"statistic":"median across rounds of (v1 round median / current round median)",
             "primary_metric":"synchronized CPU wall latency, strict input validation enabled",
             "dense_reference":"different token set; not an equivalent-kernel speedup claim",
             "runs":runs}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
