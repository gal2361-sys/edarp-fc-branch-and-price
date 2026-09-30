"""
Labeling (DP) pricing prototype measurements -> results/dp_prototype_report.json.

    python dp_report.py

b25_0: the 5 fixture dual vectors (tests/fixtures/b25_0_duals.json), rule "subset" with
and without the completion bound (5M labels, 6 GB memory guard), against the MILP prove
call on the same duals (Threads=1). b50_0: the final duals of
results/root_b50_0_artfix600.json, "subset" with the bound, 20M labels.
This is the script that produced the committed report (run at commit 69bacad).
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict

import gurobipy as gp

from bp.pricing import PricingMILP
from bp.pricing_dp import LabelLimitExceeded, PricingDP
from edarp import load_instance
from run_compact import resolve_paths


def per_lap(counts: dict) -> dict:
    pl = defaultdict(int)
    for (lap, _station), c in counts.items():
        pl[lap] += c
    return dict(sorted(pl.items()))


def run(inst, pi, mu, prune_bound, max_labels, max_rss_gb):
    try:
        r = PricingDP(inst, dominance="subset", prune_bound=prune_bound,
                      max_labels=max_labels, max_rss_gb=max_rss_gb).solve(pi, mu)
        return dict(status="complete", rc=r.best_rc if r.columns else None,
                    ncols=len(r.columns), labels=r.labels_total,
                    per_lap=per_lap(r.labels_by_pos), max_group=r.max_group,
                    pruned=r.pruned_by_bound, time=r.time)
    except LabelLimitExceeded as e:
        return dict(status=f"ABORT: {e}", labels=e.total, per_lap=per_lap(e.counts),
                    time=e.elapsed)
    finally:
        gc.collect()


def load(name):
    req, circ = resolve_paths(name, None)
    return load_instance(req, circ, name=name)


def main() -> None:
    env = gp.Env(params={"OutputFlag": 0, "Threads": 1})
    inst = load("b25_0")
    milp = PricingMILP(inst, time_limit=600, pool_size=10, env=env,
                       restricted_k=None, early_stop=None)
    out = []
    with open("tests/fixtures/b25_0_duals.json") as f:
        duals = json.load(f)
    for d in duals:
        t0 = time.perf_counter()
        m = milp.solve(d["pi"], d["mu"], prove=True)
        row = {"iteration": d["iteration"], "milp_rc": m.best_rc if m.columns else None,
               "milp_time": time.perf_counter() - t0}
        for pb in (True, False):
            row["bound" if pb else "nobound"] = run(inst, d["pi"], d["mu"], pb, 5_000_000, 6.0)
        print(json.dumps(row), flush=True)
        out.append(row)
    with open("results/root_b50_0_artfix600.json") as f:
        r = json.load(f)
    inst50 = load("b50_0")
    b50 = {"pi_pos": sum(p > 0 for p in r["final_pi"]), "max_pi": max(r["final_pi"]),
           "mu": r["final_mu"]}
    b50["dp"] = run(inst50, r["final_pi"], r["final_mu"], True, 20_000_000, 6.0)
    print(json.dumps({"b50_0": b50}), flush=True)
    with open("results/dp_prototype_report.json", "w") as f:
        json.dump({"b25_0": out, "b50_0": b50}, f, indent=1)


if __name__ == "__main__":
    main()
