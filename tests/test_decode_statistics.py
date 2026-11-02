"""Validate the scientific aggregation independently of any GPU timing."""
import json

import pytest

from scripts.summarize_decode import metrics, summarize


def samples(a,b,names=("v1","current")):
    return [dict(round=r,sample=s,impl=name,wall_ms=value,stream_ms=value)
            for r,pair in enumerate(zip(a,b)) for s in range(2)
            for name,value in zip(names,pair)]


def test_missing_or_duplicate_sample_is_not_silently_accepted():
    rows=samples([1,9],[1,3])
    for bad in (rows[:-1],rows+[rows[0]]):
        with pytest.raises(ValueError,match="samples"):metrics(bad,["v1","current"],2,2)


def test_round_ratios_and_regression_rows_are_preserved(tmp_path):
    configs={"A":{"id":"A"},"B":{"id":"B"}}
    raw=[]
    for cid,a,b in [("A",[1,9],[1,3]),("B",[0.5,0.5],[1,1])]:
        primary=samples(a,b)
        raw.append(dict(config=configs[cid],status="ok",git_head="tested",seed=1,
                        working_tree_dirty=False,rounds=2,samples=2,
                        attend=primary,append_and_attend=primary,
                        fixed_route_attention=samples([1,1],[1,1],("direct_paged","gather_fa")),
                        dense_reference=[dict(round=r,sample=s,impl="dense_fa_distinct_algorithm",
                                              wall_ms=1,stream_ms=1) for r in range(2) for s in range(2)]))
    path=tmp_path/"raw.jsonl"
    path.write_text("\n".join(json.dumps(x) for x in raw))
    rows,summary=summarize(path,configs)
    # Median of round speedups is 2; ratio of aggregate latencies would be 2.5.
    assert rows[0]["attend_wall_ms_speedup"]==2
    assert [r["id"] for r in rows]==["A","B"]
    assert summary["workloads"]["attend"]["regression_ids"]==["B"]
    assert summary["workloads"]["attend"]["geomean_speedup"]==pytest.approx(1)


def test_incomplete_configuration_matrix_is_rejected(tmp_path):
    path=tmp_path/"incomplete.jsonl";path.write_text("")
    with pytest.raises(ValueError,match="exactly once"):
        summarize(path,{"A":{"id":"A"}})
