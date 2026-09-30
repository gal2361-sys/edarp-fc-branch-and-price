"""LNS pricing heuristic on b25_0."""

from __future__ import annotations

import json
import os

import pytest

from bp.column import column_from_published_plan, evaluate
from bp.pricing import RC_TOL, PricingMILP
from bp.pricing_lns import PricingLNS
from bp.published import parse_solution
from edarp import load_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def inst():
    return load_instance(
        os.path.join(ROOT, "data", "instances", "b25_0.txt"),
        os.path.join(ROOT, "data", "instances", "circuit_artificial_instances_0.txt"),
        name="b25_0",
    )


def test_i_published_seeds_columns_are_exact(inst):
    lns = PricingLNS(inst, time_limit=1.0)
    seeds = [column_from_published_plan(inst, laps) for laps in
             parse_solution(os.path.join(ROOT, "data", "solutions", "b25_0_best_solution.txt")).values()]
    pi, mu = [100.0] * inst.n, -3.0
    cols = lns.solve(pi, mu, seeds)
    assert cols
    rcs = []
    for c in cols:
        again = evaluate(inst, c.plan())            # independent re-evaluation
        assert again is not None and again == c
        assert again.cost == pytest.approx(c.cost, abs=1e-9)
        rc = again.cost - sum(pi[i] for i in again.served) - mu
        assert rc < -RC_TOL
        rcs.append(rc)
    assert rcs == sorted(rcs)
    assert len(set(cols)) == len(cols)


def test_i_branching_filter(inst):
    lns = PricingLNS(inst, time_limit=0.5)
    seeds = [lns.singleton(i) for i in range(inst.n)]
    pi = [100.0] * inst.n
    cols = lns.solve(pi, 0.0, seeds, together=[(8, 10)], apart=[(7, 9)])
    assert cols
    assert all(c.covers(8) == c.covers(10) for c in cols)
    assert not any(c.covers(7) and c.covers(9) for c in cols)


def test_ii_lns_finds_negative_when_milp_does(inst):
    with open(os.path.join(ROOT, "tests", "fixtures", "b25_0_duals.json")) as f:
        duals = json.load(f)
    milp = PricingMILP(inst, time_limit=120, pool_size=10, restricted_k=None, early_stop=None)
    lns = PricingLNS(inst, time_limit=2.0)
    seeds = [lns.singleton(i) for i in range(inst.n)]
    assert all(s is not None for s in seeds)
    rows = []
    for d in duals:
        exact = milp.solve(d["pi"], d["mu"], prove=True)
        heur = lns.solve(d["pi"], d["mu"], seeds)
        best_h = min((c.cost - sum(d["pi"][i] for i in c.served) - d["mu"] for c in heur),
                     default=None)
        rows.append((d["iteration"], exact.best_rc, best_h, len(heur)))
        if exact.columns:
            assert heur, f"iteration {d['iteration']}: MILP rc {exact.best_rc}, LNS found none"
    print("\n  it   milp_rc    lns_rc  n_lns")
    for it, a, b, k in rows:
        print(f"  {it:2d} {a:9.4f} {b if b is not None else float('nan'):9.4f} {k:5d}")
