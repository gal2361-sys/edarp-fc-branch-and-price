"""Regression test: the published b25_0 solution must be feasible in our model.

Molenbruch et al. (2023) publish their best solutions. We parse b25_0's, fix each
vehicle's stops, pickups, performed laps and charging in a single-vehicle copy of
the compact model, and check that it is feasible and reproduces the published
objective 17.45 = 0.85 * 14 laps + 0.15 * 37 excess. This pins down the circuit
orientation and the T big-M, which both broke this before.
"""

from __future__ import annotations

import os
from dataclasses import replace

import gurobipy as gp
import pytest
from gurobipy import GRB

from bp.column import column_from_published_plan
from bp.published import parse_solution
from edarp import ALPHA1, ALPHA2, build_compact, load_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "instances")
SOLS = os.path.join(ROOT, "data", "solutions")

def solve_vehicle(inst, laps: list[dict]):
    """Fix one vehicle's published plan in a V=1 compact model and solve it."""
    n = inst.n
    one = replace(inst, V=1)
    served = {node for lap in laps for node, *_ in lap["nodes"] if node < n}

    # laps the model cannot represent must be empty in the published solution
    for lap in laps:
        if lap["l"] >= one.L:
            assert not lap["nodes"] and lap["b"] == 0, f"lap {lap['l']} >= L={one.L}"
    laps = [lap for lap in laps if lap["l"] < one.L]

    env = gp.Env(params={"OutputFlag": 0})
    m, _ = build_compact(one, env=env)
    for i in range(n):
        if i not in served:
            for name in (f"assign_{i}", f"latest_{i}", f"earliest_{i}"):
                m.remove(m.getConstrByName(name))
            for l in range(one.L):
                m.getVarByName(f"u[{i},{l},0]").UB = 0.0
    m.update()

    def fix(name: str, val: float) -> None:
        v = m.getVarByName(name)
        v.LB = v.UB = val

    for lap in laps:
        l, b = lap["l"], lap["b"]
        stations = {st for _, _, _, st in lap["nodes"]}
        performed = bool(lap["nodes"]) or b > 0
        fix(f"w[{l},0]", 1.0 if performed else 0.0)
        fix(f"b[{l},0]", b)
        fix(f"x[0,{l},0]", 1.0 if b > 0 else 0.0)
        for s in range(1, one.S):
            fix(f"x[{s},{l},0]", 1.0 if s in stations else 0.0)
        pickups = {node for node, *_ in lap["nodes"] if node < n}
        for i in served:
            fix(f"u[{i},{l},0]", 1.0 if i in pickups else 0.0)

    m.optimize()
    return one, m, served


@pytest.fixture(scope="module")
def b25_0():
    inst = load_instance(
        os.path.join(DATA, "b25_0.txt"),
        os.path.join(DATA, "circuit_artificial_instances_0.txt"),
        name="b25_0",
    )
    sol = parse_solution(os.path.join(SOLS, "b25_0_best_solution.txt"))
    return inst, sol


def test_published_b25_0_solution_is_feasible_and_costs_17_45(b25_0):
    inst, sol = b25_0
    total_laps = 0
    total_excess = 0.0
    all_served: set[int] = set()

    for v, laps in sol.items():
        one, m, served = solve_vehicle(inst, laps)
        assert m.Status == GRB.OPTIMAL, f"vehicle {v}: status {m.Status}"
        assert not (served & all_served), f"vehicle {v} re-serves a request"
        all_served |= served
        total_laps += round(sum(m.getVarByName(f"w[{l},0]").X for l in range(one.L)))
        total_excess += sum(
            inst.q[i] * (m.getVarByName(f"tau[{i}]").X - inst.eps[i]) for i in served
        )

    assert all_served == set(range(inst.n))
    assert total_laps == 14
    assert total_excess == pytest.approx(37, abs=1e-6)
    assert ALPHA1 * total_laps + ALPHA2 * total_excess == pytest.approx(17.45, abs=1e-6)


def test_published_b25_0_columns_via_pricing_model(b25_0):
    """Same check through bp: the pricing model (pricing=True) and Column."""
    inst, sol = b25_0
    cols = [column_from_published_plan(inst, laps) for laps in sol.values()]
    assert sum(c.n_laps for c in cols) == 14
    assert sum(c.excess for c in cols) == pytest.approx(37, abs=1e-6)
    assert sum(c.cost for c in cols) == pytest.approx(17.45, abs=1e-6)
    served = [i for c in cols for i in c.served]
    assert sorted(served) == list(range(inst.n))
