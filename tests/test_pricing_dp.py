"""Exactness of the labeling pricing prototype against the MILP prove call."""

from __future__ import annotations

import json
import os
import random

import gurobipy as gp
import pytest

from bp.column import evaluate, plan_from_published
from bp.pricing import PricingMILP
from bp.pricing_dp import LabelLimitExceeded, PricingDP
from bp.published import parse_solution
from edarp import load_instance, toy_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = [("none", False), ("same_sets", False), ("subset", False), ("subset", True)]


@pytest.fixture(scope="module")
def env():
    e = gp.Env(params={"OutputFlag": 0, "Threads": 1})
    yield e
    e.dispose()


def load_b25_0():
    return load_instance(
        os.path.join(ROOT, "data", "instances", "b25_0.txt"),
        os.path.join(ROOT, "data", "instances", "circuit_artificial_instances_0.txt"),
        name="b25_0",
    )


def milp_negative_optimum(milp, pi, mu):
    """Minimum reduced cost if negative, else None (the prove call uses a cutoff)."""
    res = milp.solve(pi, mu, prove=True)
    assert res.status in ("optimal", "cutoff")
    return res.best_rc if res.columns else None


def assert_same_optimum(dp, milp_rc):
    if milp_rc is None:
        assert not dp.columns
    else:
        assert dp.columns
        assert dp.best_rc == pytest.approx(milp_rc, abs=1e-6)


def check_columns(inst, res, pi, mu):
    """(d) every returned plan evaluates feasible with the same cost and rc."""
    for col, rc in zip(res.columns, res.reduced_costs):
        again = evaluate(inst, col.plan())
        assert again is not None and again == col
        assert again.cost == pytest.approx(col.cost, abs=1e-9)
        assert again.cost - sum(pi[i] for i in again.served) - mu == pytest.approx(rc, abs=1e-6)


def toy_cases():
    rng = random.Random(11)
    for seed in range(10):
        inst = toy_instance(n=6, seed=seed)
        pi = [rng.uniform(0, rng.choice([1.0, 3.0, 6.0])) for _ in range(inst.n)]
        mu = -rng.uniform(0, 1.0)
        yield seed, inst, pi, mu


@pytest.mark.parametrize("dominance,prune_bound", CONFIGS)
def test_a_toy_matches_milp(dominance, prune_bound, env):
    """Integer charging (the default): every rule equals the MILP, which charges
    continuously."""
    for seed, inst, pi, mu in toy_cases():
        milp = PricingMILP(inst, time_limit=300, pool_size=10, env=env,
                           restricted_k=None, early_stop=None)
        dp = PricingDP(inst, dominance=dominance, prune_bound=prune_bound).solve(pi, mu)
        assert_same_optimum(dp, milp_negative_optimum(milp, pi, mu))
        check_columns(inst, dp, pi, mu)


def test_a_jit_charging_is_not_exact(env):
    """Counterexample to just-in-time charging (charge to k * beta_d only): on toy
    seed 0 the optimum charges 11 before lap 0 (idle time before the first pickup)
    and 1 before lap 1. Charging to multiples of beta_d loses that; kept off."""
    seed, inst, pi, mu = next(toy_cases())
    milp = PricingMILP(inst, time_limit=300, pool_size=10, env=env,
                       restricted_k=None, early_stop=None)
    opt = milp_negative_optimum(milp, pi, mu)
    jit = PricingDP(inst, dominance="none", prune_bound=False, jit_charging=True).solve(pi, mu)
    exact = PricingDP(inst, dominance="none", prune_bound=False).solve(pi, mu)
    assert exact.best_rc == pytest.approx(opt, abs=1e-6)
    assert jit.best_rc > opt + 0.1
    assert exact.best.charge[:2] == (11.0, 1.0)


def _fixture_duals():
    with open(os.path.join(ROOT, "tests", "fixtures", "b25_0_duals.json")) as f:
        return json.load(f)


@pytest.mark.parametrize("k", [
    0,                                          # iteration 2: about 1 minute
    *[pytest.param(k, marks=pytest.mark.slow) for k in (1, 2, 3, 4)],  # 5-10 min each
])
def test_b_b25_0_fixture_duals_match_milp(k, env):
    """Subset dominance + completion bound equals the MILP on b25_0."""
    inst = load_b25_0()
    d = _fixture_duals()[k]
    milp = PricingMILP(inst, time_limit=600, pool_size=10, env=env,
                       restricted_k=None, early_stop=None)
    dp = PricingDP(inst, dominance="subset", prune_bound=True,
                   max_labels=5_000_000).solve(d["pi"], d["mu"])
    assert_same_optimum(dp, milp_negative_optimum(milp, d["pi"], d["mu"]))
    check_columns(inst, dp, d["pi"], d["mu"])


def test_c_published_charges_are_whole_laps():
    """The published b25_0 plans charge only beta_d or 2 * beta_d, always from an
    empty battery to a multiple of beta_d."""
    inst = load_b25_0()
    sol = parse_solution(os.path.join(ROOT, "data", "solutions", "b25_0_best_solution.txt"))
    for laps in sol.values():
        battery = 0.0
        for lap in plan_from_published(inst, laps):
            if not lap.performed:
                continue
            assert lap.charge in (0.0, inst.beta_d, 2 * inst.beta_d)
            battery += lap.charge
            assert battery % inst.beta_d == 0
            battery -= inst.beta_d


def test_label_limit_aborts_cleanly():
    inst = toy_instance(n=6, seed=0)
    with pytest.raises(LabelLimitExceeded) as e:
        PricingDP(inst, dominance="none", prune_bound=False,
                  max_labels=1000).solve([5.0] * inst.n, 0.0)
    assert e.value.total > 1000 and e.value.pos is not None
