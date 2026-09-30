"""Exact column evaluation (bp.column.evaluate)."""

from __future__ import annotations

import os
import random
from dataclasses import replace

import pytest

from bp.column import LapPlan, evaluate, plan_from_model, plan_from_published
from bp.pricing import PricingMILP
from bp.published import parse_solution
from edarp import ALPHA1, ALPHA2, load_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def inst():
    return load_instance(
        os.path.join(ROOT, "data", "instances", "b25_0.txt"),
        os.path.join(ROOT, "data", "instances", "circuit_artificial_instances_0.txt"),
        name="b25_0",
    )


@pytest.fixture(scope="module")
def published(inst):
    sol = parse_solution(os.path.join(ROOT, "data", "solutions", "b25_0_best_solution.txt"))
    return [plan_from_published(inst, laps) for laps in sol.values()]


def test_i_published_plans(inst, published):
    cols = [evaluate(inst, p) for p in published]
    assert all(c is not None for c in cols)
    assert [c.n_laps for c in cols] == [5, 5, 4]
    assert [c.excess for c in cols] == pytest.approx([12, 22, 3], abs=1e-9)
    assert sum(c.cost for c in cols) == pytest.approx(17.45, abs=1e-9)


def test_ii_evaluate_never_cheaper_than_gurobi(inst):
    """evaluate() is exact, so it can only match or exceed the solver's cost."""
    rng = random.Random(0)
    pricer = PricingMILP(inst, time_limit=60, pool_size=10)
    m, v = pricer.model, pricer.vars
    checked = 0
    while checked < 20:
        pi = [rng.uniform(0, 8) for _ in range(inst.n)]
        res = pricer.solve(pi, 0.0, prove=True)
        assert res.status == "optimal"
        # re-solve with the same duals to read raw pool values (solve() cleans up)
        for i in range(inst.n):
            v["y"][i].Obj = pricer._y_base[i] - pi[i]
        m.ObjCon = 0.0
        m.optimize()
        for k in range(m.SolCount):
            m.Params.SolutionNumber = k
            w = sum(v["w"][l, 0].Xn for l in range(inst.L))
            gurobi_cost = ALPHA1 * w + ALPHA2 * sum(
                inst.q[i] * (v["tau"][i].Xn - inst.eps[i] * v["y"][i].Xn)
                for i in range(inst.n)
            )
            col = evaluate(inst, plan_from_model(inst, m, v, sol=k))
            if col is None or not col.served:
                continue
            assert col.cost >= gurobi_cost - 1e-6, (k, col.cost, gurobi_cost)
            checked += 1
            if checked == 20:
                break


def test_iii_deadline_violation_returns_none(inst, published):
    plan = published[0]
    assert 8 in plan[0].pickups   # request 8 is dropped at time 79 on lap 0
    # shrink d_8 just below that dropoff time
    tight = replace(inst)
    tight.d = list(inst.d)
    tight.d[8] = 78.0
    assert evaluate(tight, plan) is None


def test_other_violations_return_none(inst, published):
    plan = published[0]
    # missing the dropoff stop of request 8 (station 5) on lap 0
    bad = [replace(plan[0], stops=plan[0].stops - {5})] + plan[1:]
    assert evaluate(inst, bad) is None
    # charging without a depot stop
    bad = [replace(plan[0], stops=plan[0].stops - {0})] + plan[1:]
    assert evaluate(inst, bad) is None
    # battery: charging beyond beta_m
    bad = [replace(plan[0], charge=inst.beta_m + 1)] + plan[1:]
    assert evaluate(inst, bad) is None
    # a gap in the performed laps
    gap = list(plan)
    gap[1] = LapPlan(False, 0.0, frozenset(), frozenset())
    assert evaluate(inst, gap) is None
