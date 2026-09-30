"""PricingMILP sanity tests on b25_0."""

from __future__ import annotations

import os

import pytest

from bp.pricing import RC_TOL, PricingMILP
from edarp import ALPHA1, ALPHA2, load_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "instances")

# vehicle 0 of the published b25_0 solution: 5 laps, excess 12
VEH0 = frozenset({8, 10, 7, 21, 19, 9, 14, 22})
VEH0_COST = ALPHA1 * 5 + ALPHA2 * 12  # 6.05


@pytest.fixture(scope="module")
def inst():
    return load_instance(
        os.path.join(DATA, "b25_0.txt"),
        os.path.join(DATA, "circuit_artificial_instances_0.txt"),
        name="b25_0",
    )


@pytest.fixture(scope="module")
def pricer(inst):
    return PricingMILP(inst, time_limit=300, pool_size=10)


def veh0_pi(n: int) -> list[float]:
    return [100.0 if i in VEH0 else 0.0 for i in range(n)]


def test_a_zero_duals_give_empty_plan(inst, pricer):
    res = pricer.solve([0.0] * inst.n, 0.0, prove=True)
    # prove calls use Cutoff = -RC_TOL: "no plan below the cutoff" is the proof,
    # and the empty plan (reduced cost 0) is cut off rather than returned
    assert res.status == "cutoff"
    assert res.columns == []
    assert res.bound == pytest.approx(-RC_TOL)
    assert res.proven_no_negative
    assert res.pruned == inst.n          # every dual is 0, so every request is pruned


def test_b_high_dual_forces_request(inst, pricer):
    pi = [0.0] * inst.n
    pi[8] = 1000.0
    res = pricer.solve(pi, 0.0, prove=True)
    assert res.status == "optimal"
    assert res.columns and res.columns[0].covers(8)
    assert res.reduced_costs == sorted(res.reduced_costs)
    assert all(rc < -RC_TOL for rc in res.reduced_costs)


def test_c_published_vehicle_0(inst, pricer):
    res = pricer.solve(veh0_pi(inst.n), 0.0, prove=True)
    print(f"\n[test c] pricing wall-clock {res.time:.2f}s, status {res.status}, "
          f"{len(res.columns)} columns, best rc {res.best_rc:.4f}")
    assert res.status == "optimal"
    exact = [c for c in res.columns if c.served == VEH0]
    assert exact, f"no column serves exactly VEH0; got {[sorted(c.served) for c in res.columns]}"
    best = min(exact, key=lambda c: c.cost)
    print(f"[test c] best exact-set column: {best.n_laps} laps, excess {best.excess:g}, "
          f"cost {best.cost:.4f} (published {VEH0_COST:.2f})")
    assert best.cost <= VEH0_COST + 1e-6


def test_d_branching_hooks(inst, pricer):
    pi = veh0_pi(inst.n)
    n_constrs = pricer.model.NumConstrs

    res = pricer.solve(pi, 0.0, together=[(8, 10)], prove=True)
    assert res.status == "optimal" and res.columns
    assert all(c.covers(8) == c.covers(10) for c in res.columns)
    assert pricer.model.NumConstrs == n_constrs

    res = pricer.solve(pi, 0.0, apart=[(8, 10)], prove=True)
    assert res.status == "optimal" and res.columns
    assert not any(c.covers(8) and c.covers(10) for c in res.columns)
    assert pricer.model.NumConstrs == n_constrs


def test_d_together_hook_is_binding(inst, pricer):
    """Only request 8 is worth serving; together(8,10) must drag 10 along."""
    pi = [0.0] * inst.n
    pi[8] = 1000.0
    free = pricer.solve(pi, 0.0, prove=True)
    assert any(not c.covers(10) for c in free.columns)  # the hook is not vacuous
    res = pricer.solve(pi, 0.0, together=[(8, 10)], prove=True)
    assert res.status == "optimal" and res.columns
    assert all(c.covers(8) and c.covers(10) for c in res.columns)


def test_restricted_pricing_marks_and_restores(inst):
    """Restricted call: columns negative w.r.t. the full duals, never a proof, and
    every u bound (including B3/B4 zeros) restored afterwards."""
    p = PricingMILP(inst, time_limit=60, pool_size=10, restricted_k=5, early_stop=None)
    u = p.vars["u"]
    before = {k: v.UB for k, v in u.items()}
    assert any(ub == 0.0 for ub in before.values())   # B3/B4 bounds exist
    res = p.solve(veh0_pi(inst.n), 0.0)
    assert res.kind == "restricted" and res.status == "restricted"
    assert res.bound == float("-inf") and not res.proven_no_negative
    assert res.columns and all(rc < -RC_TOL for rc in res.reduced_costs)
    top5 = set(sorted(range(inst.n), key=lambda i: (-veh0_pi(inst.n)[i], i))[:5])
    assert all(c.served <= top5 for c in res.columns)
    assert {k: v.UB for k, v in u.items()} == before
    assert p.model.Params.MIPFocus == 0 and p.model.Params.TimeLimit == 60


def test_early_stop_is_never_a_proof(inst):
    p = PricingMILP(inst, time_limit=60, pool_size=10, restricted_k=None, early_stop=1)
    res = p.solve(veh0_pi(inst.n), 0.0)
    assert res.kind == "exact"
    assert not res.proven_no_negative
    assert res.bound <= res.best_rc + 1e-9          # MIP bound at termination stays valid
    proof = p.solve([0.0] * inst.n, 0.0, prove=True)
    assert proof.kind == "prove" and proof.proven_no_negative


def test_dual_pruning_is_exact_and_restores_bounds(inst):
    """Pruning pi_i <= 0 requests must not change the optimum, and must restore
    every u bound (B3/B4 zeros included)."""
    import random
    rng = random.Random(3)
    on = PricingMILP(inst, time_limit=120, pool_size=10, restricted_k=None, early_stop=None)
    off = PricingMILP(inst, time_limit=120, pool_size=10, restricted_k=None, early_stop=None,
                      dual_pruning=False)
    before = {k: v.UB for k, v in on.vars["u"].items()}
    for _ in range(3):
        pi = [rng.choice([0.0, -1.0, rng.uniform(0, 8)]) for _ in range(inst.n)]
        a, b = on.solve(pi, 0.0, prove=True), off.solve(pi, 0.0, prove=True)
        assert a.pruned > 0 and b.pruned == 0
        ra = a.best_rc if a.columns else 0.0
        rb = b.best_rc if b.columns else 0.0
        assert ra == pytest.approx(rb, abs=1e-6)
        assert all(not (c.served & {i for i in range(inst.n) if pi[i] <= 0}) for c in a.columns)
    assert {k: v.UB for k, v in on.vars["u"].items()} == before


def test_dual_pruning_keeps_together_pairs(inst):
    pi = [0.0] * inst.n
    pi[8] = 1000.0                     # 10 has pi = 0 but is tied to 8
    p = PricingMILP(inst, time_limit=120, pool_size=10, restricted_k=None, early_stop=None)
    res = p.solve(pi, 0.0, together=[(8, 10)], prove=True)
    assert res.columns and all(c.covers(8) and c.covers(10) for c in res.columns)
    assert res.pruned == inst.n - 2


def test_prove_trajectory_and_params_restored(inst):
    p = PricingMILP(inst, time_limit=60, pool_size=10)
    res = p.solve(veh0_pi(inst.n), 0.0, prove=True)
    assert res.kind == "prove" and res.bound_trajectory
    assert p.model.Params.MIPFocus == 0 and p.model.Params.Cutoff > 1e90
    assert p.model.Params.TimeLimit == 60
