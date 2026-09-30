"""Constructive full-fleet heuristic (bp/initial.py)."""

from __future__ import annotations

import pytest

from bp.column import evaluate
from bp.initial import initial_solution
from edarp import load_instance, solve_compact, toy_instance
from run_compact import PUBLISHED, resolve_paths


def check_cover(inst, sol):
    assert sol.covered and not sol.uncovered
    assert len(sol.columns) <= inst.V
    served = [i for c in sol.columns for i in c.served]
    assert sorted(served) == list(range(inst.n))                   # each exactly once
    for c in sol.columns:                                          # exact costs
        again = evaluate(inst, c.plan())
        assert again is not None and again.cost == pytest.approx(c.cost, abs=1e-9)
    assert sol.cost == pytest.approx(sum(c.cost for c in sol.columns), abs=1e-9)


@pytest.mark.parametrize("z", range(10))
def test_b25_cover_and_cost_at_least_published(z):
    name = f"b25_{z}"
    req, circ = resolve_paths(name, None)
    inst = load_instance(req, circ, name=name)
    sol = initial_solution(inst, time_per_vehicle=2.0, repair_time=30.0)
    check_cover(inst, sol)
    assert sol.cost >= PUBLISHED[name] - 1e-6


@pytest.mark.parametrize("n,seed", [
    (6, 0), (6, 1), (8, 3),
    # known limitation: the heuristic leaves request 6
    # uncovered although a cover exists (compact optimum 17.05); strict, so a fix shows
    pytest.param(10, 2, marks=pytest.mark.xfail(strict=True, reason="heuristic misses a cover")),
])
def test_toy_cover_and_cost_at_least_optimum(n, seed):
    inst = toy_instance(n=n, seed=seed)
    sol = initial_solution(inst, time_per_vehicle=1.0, repair_time=10.0)
    check_cover(inst, sol)
    _, compact = solve_compact(inst, time_limit=300, log=False)
    assert sol.cost >= compact["ub"] - 1e-6
