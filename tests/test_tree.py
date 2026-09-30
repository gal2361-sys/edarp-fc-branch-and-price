"""Branch-and-price tree on toy instances."""

from __future__ import annotations

import math

import gurobipy as gp
import pytest

from bp.master import RestrictedMaster
from bp.pricing import PricingMILP
from bp.pricing_lns import PricingLNS
from bp.scorers import make_scorer
from bp.tree import BranchAndPrice, violates
from edarp import solve_compact, toy_instance

BRANCHING_TOY = (6, 1)      # n, seed: root bound 8.75 < optimum 8.8, needs branching


@pytest.fixture(scope="module")
def env():
    e = gp.Env(params={"OutputFlag": 0, "Threads": 1})
    yield e
    e.dispose()


def make_tree(inst, env, scorer="most_fractional", time_limit=300):
    pricing = PricingMILP(inst, time_limit=30, pool_size=20, env=env)
    master = RestrictedMaster(inst, env=env)
    master.add_initial_columns(pricing)
    master.set_artificial_cost(master.singleton_cost_sum())
    return BranchAndPrice(inst, master, pricing, PricingLNS(inst, time_limit=0.2),
                          make_scorer(scorer), time_limit=time_limit)


@pytest.mark.parametrize("n,seed", [(6, 0), (6, 1), (6, 2), (10, 2)])
def test_toy_closes_to_compact_optimum(n, seed, env):
    """(6, 1) and (10, 2) need branching (root bound below the optimum)."""
    inst = toy_instance(n=n, seed=seed)
    _, compact = solve_compact(inst, time_limit=300, log=False)
    assert compact["status"] == "optimal"
    tr = make_tree(inst, env).solve()
    assert tr.status == "optimal"
    assert tr.ub == pytest.approx(compact["ub"], abs=1e-6)
    assert tr.lb == pytest.approx(tr.ub, abs=1e-9)
    if (n, seed) in (BRANCHING_TOY, (10, 2)):
        assert tr.nodes_processed > 1 and tr.root_lb < tr.ub - 1e-6


def test_children_respect_branching(env):
    inst = toy_instance(*BRANCHING_TOY)
    bp_ = make_tree(inst, env)
    root = bp_.new_node((), (), -math.inf, 0)
    children = bp_.process(root)
    assert len(children) == 2
    for child in children:
        assert len(child.together) + len(child.apart) == 1
        # allowed pool: every column with ub > 0 respects the child's decisions
        before = [v.UB for v in bp_.master.vars]
        blocked = bp_.allow(child)
        assert blocked
        for col, var in zip(bp_.master.columns, bp_.master.vars):
            if var.UB > 0:
                assert not violates(col, child.together, child.apart)
        bp_.restore(blocked)
        assert [v.UB for v in bp_.master.vars] == before
        # columns generated at the child respect them too
        n0 = len(bp_.master.columns)
        bp_.process(child)
        for col in bp_.master.columns[n0:]:
            for i, j in child.together:
                assert (i in col.served) == (j in col.served)
            for i, j in child.apart:
                assert not (i in col.served and j in col.served)


def test_parent_lb_and_monotone_bound_log(env):
    inst = toy_instance(*BRANCHING_TOY)
    bp_ = make_tree(inst, env)
    tr = bp_.solve()
    by_id = {r["id"]: r for r in tr.node_log}
    kids = [r for r in tr.node_log if r.get("parent") is not None]
    assert kids
    for r in kids:
        assert r["parent_lb"] == by_id[r["parent"]]["lb"]
    times = [t for t, _, _ in tr.bound_log]
    assert times == sorted(times)
    root_end = by_id[0]["time"]
    assert any(t < root_end for t in times)       # bounds logged during the root too
    lbs = [lb for _, lb, _ in tr.bound_log]
    assert all(b >= a - 1e-9 for a, b in zip(lbs, lbs[1:]))
    ubs = [ub for _, _, ub in tr.bound_log]
    assert all(b <= a + 1e-9 for a, b in zip(ubs, ubs[1:]))
    assert tr.bound_log[-1][1] == pytest.approx(tr.bound_log[-1][2])


def test_node_at_or_above_incumbent_is_never_processed(env, monkeypatch):
    inst = toy_instance(*BRANCHING_TOY)
    bp_ = make_tree(inst, env)
    bp_.incumbent = 5.0
    processed = []
    monkeypatch.setattr(bp_, "process", lambda node: processed.append(node) or [])
    start = [bp_.new_node(((0, 1),), (), 6.0, 1), bp_.new_node((), ((0, 1),), 5.0, 1)]
    tr = bp_.solve(start)
    assert processed == []
    assert tr.nodes_pruned == 2 and tr.nodes_processed == 0 and tr.status == "optimal"


def test_initial_incumbent_does_not_fake_a_lower_bound(env):
    """Regression: an incumbent offered before solve() was logged with LB = UB because
    the open list was still empty."""
    from bp.initial import initial_solution
    inst = toy_instance(*BRANCHING_TOY)
    bp_ = make_tree(inst, env)
    init = initial_solution(inst, time_per_vehicle=0.5, repair_time=5.0)
    assert init.covered
    for c in init.columns:
        bp_.master.add_column(c)
    bp_.offer(init.columns, "initial heuristic")
    bp_.log_initial_incumbent()
    tr = bp_.solve()
    assert tr.bound_log[0][1] == -math.inf and tr.bound_log[0][2] == pytest.approx(init.cost)
    lbs = [lb for _, lb, _ in tr.bound_log]
    assert all(b >= a - 1e-9 for a, b in zip(lbs, lbs[1:]))
    assert tr.status == "optimal"
