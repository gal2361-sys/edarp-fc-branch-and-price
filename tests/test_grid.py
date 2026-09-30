"""Objective grid (docs/GRID_ROUNDING.md) and the rounding helper."""

from __future__ import annotations

import os
from dataclasses import replace

import pytest

from bp.column import evaluate, plan_from_published
from bp.grid import grid_step, round_up
from bp.published import parse_solution
from edarp import load_instance
from run_compact import PUBLISHED, resolve_paths

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    req, circ = resolve_paths(name, None)
    return load_instance(req, circ, name=name)


@pytest.mark.parametrize("size", ["b25", "b50", "b75"])
def test_grid_step_of_all_artificial_instances(size):
    for z in range(10):
        inst = load(f"{size}_{z}")
        half = any(float(d) != int(d) for d in inst.d)
        assert grid_step(inst) == pytest.approx(0.025 if half else 0.05)


def test_published_optima_lie_on_the_grid():
    for name, value in PUBLISHED.items():
        step = grid_step(load(name))
        assert abs(value / step - round(value / step)) < 1e-9, name


def test_round_up():
    assert round_up(45.949996, 0.025) == pytest.approx(45.95)
    assert round_up(45.95, 0.025) == pytest.approx(45.95)          # float noise tolerated
    assert round_up(45.95 + 2e-5, 0.025) == pytest.approx(45.975)
    assert round_up(52.5000001, 0.025) == pytest.approx(52.5)
    assert round_up(3.21, None) == 3.21
    assert round_up(float("-inf"), 0.025) == float("-inf")


def test_feasible_column_can_be_off_grid():
    """The per-column claim is false: charging is continuous. Published b25_0 vehicle 0
    with its lap-1 charge raised from 8 to 8.3 stays feasible and leaves the grid."""
    inst = load("b25_0")
    laps = parse_solution(os.path.join(ROOT, "data", "solutions", "b25_0_best_solution.txt"))[0]
    plan = plan_from_published(inst, laps)
    assert plan[1].charge == 8.0
    plan[1] = replace(plan[1], charge=8.3)
    col = evaluate(inst, plan)
    assert col is not None
    assert col.cost == pytest.approx(6.185)
    assert abs(col.cost * 40 - round(col.cost * 40)) > 0.1       # off the 0.025 grid too


# --- empirical check on generated columns (step 2 of the v2 plan) --------------------

def _tree_pool(inst, time_limit):
    import gurobipy as gp
    from bp.initial import initial_solution
    from bp.master import RestrictedMaster
    from bp.pricing import PricingMILP
    from bp.pricing_lns import PricingLNS
    from bp.scorers import make_scorer
    from bp.tree import BranchAndPrice
    env = gp.Env(params={"OutputFlag": 0})
    pricing = PricingMILP(inst, time_limit=30, pool_size=50, env=env)
    master = RestrictedMaster(inst, env=env)
    master.add_initial_columns(pricing)
    master.set_artificial_cost(master.singleton_cost_sum())
    init = initial_solution(inst, time_per_vehicle=1.0, repair_time=10.0)
    if init.covered:
        for c in init.columns:
            master.add_column(c)
    bp_ = BranchAndPrice(inst, master, pricing, PricingLNS(inst, time_limit=1.0),
                         make_scorer("most_fractional"), time_limit=time_limit)
    if init.covered:
        bp_.offer(init.columns, "initial heuristic")
    tr = bp_.solve()
    return tr, master.columns


@pytest.mark.slow
@pytest.mark.parametrize("case", ["toy_6_0", "toy_6_1", "toy_10_2", "b25_0"])
def test_generated_columns_lie_on_the_grid(case):
    """Every column in the pool after a tree run has cost * 40 within 1e-6 of an
    integer (0.025 grid). Empirical: feasible off-grid columns exist in principle
    (test_feasible_column_can_be_off_grid); this checks whether pricing produces them."""
    from edarp import toy_instance
    if case.startswith("toy"):
        _, n, seed = case.split("_")
        inst = toy_instance(n=int(n), seed=int(seed))
    else:
        inst = load(case)
    tr, cols = _tree_pool(inst, time_limit=900)
    assert tr.status == "optimal"
    off = [c.cost for c in cols if abs(c.cost * 40 - round(c.cost * 40)) > 1e-6]
    assert not off, f"{len(off)} of {len(cols)} columns off the 0.025 grid, e.g. {off[:5]}"
