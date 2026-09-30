"""Restricted master on b25_0 seeded with the three published vehicle columns."""

from __future__ import annotations

import os

import pytest

from bp.column import column_from_published_plan
from bp.master import RestrictedMaster
from bp.published import parse_solution
from edarp import load_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def master_and_cols():
    inst = load_instance(
        os.path.join(ROOT, "data", "instances", "b25_0.txt"),
        os.path.join(ROOT, "data", "instances", "circuit_artificial_instances_0.txt"),
        name="b25_0",
    )
    sol = parse_solution(os.path.join(ROOT, "data", "solutions", "b25_0_best_solution.txt"))
    cols = [column_from_published_plan(inst, laps) for laps in sol.values()]
    master = RestrictedMaster(inst)  # artificial columns are built in
    for c in cols:
        assert master.add_column(c)
    return master, cols


def test_add_column_dedupes(master_and_cols):
    master, cols = master_and_cols
    assert not master.add_column(cols[0])
    assert len(master.columns) == 3


def test_lp_is_published_solution(master_and_cols):
    master, _ = master_and_cols
    sol = master.solve()
    assert sol.value == pytest.approx(17.45, abs=1e-6)
    assert sol.lambdas == pytest.approx([1.0, 1.0, 1.0], abs=1e-9)
    assert not sol.artificial_active
    assert sol.mu <= 1e-9


def test_integer_is_published_solution(master_and_cols):
    master, cols = master_and_cols
    ip = master.solve_integer(time_limit=60)
    assert ip.ub == pytest.approx(17.45, abs=1e-6)
    assert not ip.artificial_active
    assert set(ip.columns) == set(cols)


def test_integer_sees_columns_added_after_last_lp_solve(master_and_cols):
    """Regression: solve_integer copied the model before pending columns were flushed."""
    master, cols = master_and_cols
    from dataclasses import replace
    # a copy of column 0 with lower cost and a different charge, so it is a new column
    cheaper = replace(cols[0], cost=cols[0].cost - 1.0,
                      charge=tuple(c + 1.0 for c in cols[0].charge))
    assert master.add_column(cheaper)
    ip = master.solve_integer(time_limit=60)
    assert ip.ub == pytest.approx(16.45, abs=1e-6)
    assert cheaper in ip.columns


def test_singleton_sum_artificial_cost_is_large():
    """The artificial cost is the sum of the cheapest singleton costs.

    Note: with only singleton columns, V=3 of them fit under the fleet row and
    every other request is covered by its artificial, so every request is marginal
    against it and pi_i = artificial cost for all i is a valid (and here the
    chosen) LP dual. "Not all duals equal the artificial cost" therefore cannot be
    required at this LP; what must hold is that the cost does not cap the duals of
    the converged root (checked in the next test).
    """
    import gurobipy as gp
    from bp.pricing import PricingMILP
    inst = load_instance(
        os.path.join(ROOT, "data", "instances", "b25_0.txt"),
        os.path.join(ROOT, "data", "instances", "circuit_artificial_instances_0.txt"),
        name="b25_0",
    )
    env = gp.Env(params={"OutputFlag": 0})
    master = RestrictedMaster(inst, env=env)
    master.add_initial_columns(PricingMILP(inst, time_limit=60, env=env))
    assert master.columns and all(len(c.served) == 1 for c in master.columns)
    art = master.singleton_cost_sum()
    assert art > 10 * max(c.cost for c in master.columns)
    master.set_artificial_cost(art)
    sol = master.solve()
    assert max(sol.pi) <= art + 1e-6


def test_root_converges_after_escalation_from_tiny_artificial_cost():
    """Start with an artificial cost far too low to let any real column in: the
    stall escalation must raise it until the root converges at 17.45 with no
    artificial column active."""
    import gurobipy as gp
    from bp.colgen import generate_columns
    from bp.pricing import PricingMILP
    from bp.pricing_lns import PricingLNS
    inst = load_instance(
        os.path.join(ROOT, "data", "instances", "b25_0.txt"),
        os.path.join(ROOT, "data", "instances", "circuit_artificial_instances_0.txt"),
        name="b25_0",
    )
    env = gp.Env(params={"OutputFlag": 0})
    pricing = PricingMILP(inst, time_limit=30, pool_size=50, env=env)
    master = RestrictedMaster(inst, env=env)
    master.add_initial_columns(pricing)
    master.set_artificial_cost(0.01)
    cg = generate_columns(inst, master, pricing, time_limit=600,
                          lns=PricingLNS(inst, time_limit=2.0))
    assert cg.artificial_escalations >= 1
    assert master.artificial_cost > 0.01
    assert cg.converged and not cg.artificial_active
    assert cg.lp_value == pytest.approx(17.45, abs=1e-6)
    # the converged duals sit strictly below the artificial cost, i.e. it caps nothing
    assert max(cg.final_pi) < master.artificial_cost - 1e-6
