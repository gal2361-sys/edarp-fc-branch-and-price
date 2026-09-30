"""Column generation: Lagrangian bound bookkeeping and the LNS time-limit restore."""

from __future__ import annotations

import json
import math
import os

import gurobipy as gp
import pytest

from bp.colgen import generate_columns
from bp.master import RestrictedMaster
from bp.pricing import PricingMILP
from bp.pricing_lns import PricingLNS
from edarp import load_instance, toy_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def env():
    e = gp.Env(params={"OutputFlag": 0, "Threads": 1})
    yield e
    e.dispose()


def b25_0():
    return load_instance(
        os.path.join(ROOT, "data", "instances", "b25_0.txt"),
        os.path.join(ROOT, "data", "instances", "circuit_artificial_instances_0.txt"),
        name="b25_0",
    )


def setup(inst, env, time_limit=30):
    pricing = PricingMILP(inst, time_limit=time_limit, pool_size=50, env=env)
    master = RestrictedMaster(inst, env=env)
    master.add_initial_columns(pricing)
    master.set_artificial_cost(master.singleton_cost_sum())
    return master, pricing


def test_lns_time_limit_restored_after_short_budget(env):
    """Regression: generate_columns used to leave lns.time_limit at whatever time was
    left at the end of its budget, crippling LNS for every later call."""
    inst = toy_instance(n=6, seed=1)
    master, pricing = setup(inst, env)
    lns = PricingLNS(inst, time_limit=0.5)
    generate_columns(inst, master, pricing, time_limit=0.2, lns=lns)   # cap < 0.5 applies
    assert lns.time_limit == 0.5
    cg = generate_columns(inst, master, pricing, time_limit=60, lns=lns)
    assert lns.time_limit == 0.5 and cg.converged


def test_lagrangian_bound_valid_on_fixture_duals(env):
    """For each b25_0 fixture dual vector, sum(pi) + V*mu + V*min(0, bound) built from
    a prove call's dual bound (ObjBound) is a lower bound on the optimum 17.45, and the
    dual bound never exceeds the best reduced cost found."""
    inst = b25_0()
    # single-threaded prove calls on these duals take up to about 80 s
    # (results/dp_prototype_report.json); 30 s made the test time out
    _, pricing = setup(inst, env, time_limit=300)
    with open(os.path.join(ROOT, "tests", "fixtures", "b25_0_duals.json")) as f:
        duals = json.load(f)
    for d in duals:
        res = pricing.solve(d["pi"], d["mu"], prove=True)
        assert res.status in ("optimal", "cutoff")
        if res.columns:
            assert res.bound <= res.best_rc + 1e-9
        lb = sum(d["pi"]) + inst.V * d["mu"] + inst.V * min(0.0, res.bound)
        assert lb <= 17.45 + 1e-6, (d["iteration"], lb)


def test_lagrangian_lb_is_max_over_calls(env):
    inst = b25_0()
    master, pricing = setup(inst, env)
    cg = generate_columns(inst, master, pricing, time_limit=90,
                          lns=PricingLNS(inst, time_limit=1.0))
    rows = [r for r in cg.log if math.isfinite(r["lagrangian_lb"])]
    assert rows
    best = max(rows, key=lambda r: r["lagrangian_lb"])
    assert cg.lagrangian_lb == pytest.approx(best["lagrangian_lb"], abs=1e-12)
    assert cg.lagrangian_lb_iteration == best["iteration"]
    assert cg.lagrangian_lb <= 17.45 + 1e-6
    for r in cg.log:
        if r["best_rc"] is not None and math.isfinite(r["pricing_bound"]):
            assert r["pricing_bound"] <= r["best_rc"] + 1e-9   # a dual bound, not an incumbent
    if cg.converged:
        assert cg.lagrangian_lb == pytest.approx(cg.lp_value, abs=1e-5)
