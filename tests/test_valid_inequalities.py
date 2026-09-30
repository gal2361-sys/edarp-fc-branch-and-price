"""Four checks on the Electronic Companion B valid inequalities (b25_0).

a. flag off: compact-model fingerprints unchanged
b. each family alone: the published b25_0 plans stay feasible, at 14 laps,
   excess 37, 17.45 (per vehicle in the pricing model; B2 in the V=3 compact
   model under some vehicle permutation)
c. all families together: the same checks
d. pricing optimum unchanged on 10 seeded random dual vectors
"""

from __future__ import annotations

import itertools
import os
import random
import time
from dataclasses import replace

import gurobipy as gp
import pytest
from gurobipy import GRB

from bp.published import parse_solution
from edarp import ALPHA1, ALPHA2, VI_FAMILIES, build_compact, load_instance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "instances")

FINGERPRINTS = {  # compact model, flag off, recorded before the families existed
    "b25_0": -0x6B6BF06,
    "b25_1": 0x75435A54,
    "b50_0": -0x329EF4AB,
    "b75_3": -0x2AF4C956,
}
PRICING_FAMILIES = sorted(VI_FAMILIES - {"B2"}, key=lambda f: int(f[1:]))


def load(name: str):
    z = name.split("_")[1]
    return load_instance(
        os.path.join(DATA, f"{name}.txt"),
        os.path.join(DATA, f"circuit_artificial_instances_{z}.txt"),
        name=name,
    )


@pytest.fixture(scope="module")
def env():
    e = gp.Env(params={"OutputFlag": 0})
    yield e
    e.dispose()


@pytest.fixture(scope="module")
def inst():
    return load("b25_0")


@pytest.fixture(scope="module")
def published():
    sol = parse_solution(os.path.join(ROOT, "data", "solutions", "b25_0_best_solution.txt"))
    return [sol[v] for v in sorted(sol)]


def fix_plan(inst, v_, laps, v: int) -> None:
    """Fix one published vehicle plan onto vehicle index v of a built model."""
    n, L = inst.n, inst.L
    by_lap = {lap["l"]: lap for lap in laps}
    for l, lap in by_lap.items():
        assert l < L or (not lap["nodes"] and lap["b"] == 0)
    for l in range(L):
        lap = by_lap.get(l, {"b": 0.0, "nodes": []})
        stations = {st for _, _, _, st in lap["nodes"]}
        picks = {node for node, *_ in lap["nodes"] if node < n}
        vals = [(v_["w"][l, v], 1.0 if (lap["nodes"] or lap["b"] > 0) else 0.0),
                (v_["b"][l, v], lap["b"]),
                (v_["x"][0, l, v], 1.0 if lap["b"] > 0 else 0.0)]
        vals += [(v_["x"][s, l, v], 1.0 if s in stations else 0.0) for s in range(1, inst.S)]
        vals += [(v_["u"][i, l, v], 1.0 if i in picks else 0.0) for i in range(n)]
        for var, val in vals:
            if val > var.UB + 1e-9:
                raise AssertionError(f"{var.VarName} = {val} exceeds its bound {var.UB}")
            var.LB = var.UB = val


def per_vehicle_check(inst, published, fams, env):
    """Pricing model per vehicle; returns (laps, excess) or raises naming the family."""
    one = replace(inst, V=1)
    laps_total, excess_total = 0, 0.0
    for k, laps in enumerate(published):
        m, v_ = build_compact(one, env=env, pricing=True, valid_inequalities=fams)
        try:
            fix_plan(one, v_, laps, 0)
        except AssertionError as e:
            raise AssertionError(f"families {sorted(fams)}: vehicle {k}: {e}") from None
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            m.computeIIS()
            iis = [c.ConstrName for c in m.getConstrs() if c.IISConstr]
            raise AssertionError(f"families {sorted(fams)}: vehicle {k} infeasible; IIS {iis[:10]}")
        served = [i for i in range(one.n) if v_["y"][i].X > 0.5]
        laps_total += round(sum(v_["w"][l, 0].X for l in range(one.L)))
        excess_total += sum(one.q[i] * (v_["tau"][i].X - one.eps[i]) for i in served)
        m.dispose()
    return laps_total, excess_total


def compact_permutation_check(inst, published, fams, env):
    """V=3 compact model with fams; returns the objective of the first feasible
    vehicle permutation of the published plans, or None if none is feasible."""
    for perm in itertools.permutations(range(inst.V)):
        m, v_ = build_compact(inst, env=env, valid_inequalities=fams)
        try:
            for k, laps in enumerate(published):
                fix_plan(inst, v_, laps, perm[k])
        except AssertionError:
            m.dispose()
            continue
        m.optimize()
        if m.Status == GRB.OPTIMAL:
            obj = m.ObjVal
            m.dispose()
            return perm, obj
        m.dispose()
    return None


# --- a --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(FINGERPRINTS))
def test_a_fingerprint_unchanged_with_flag_off(name, env):
    m, _ = build_compact(load(name), env=env)
    assert m.Fingerprint == FINGERPRINTS[name]


# --- b --------------------------------------------------------------------------

@pytest.mark.parametrize("family", PRICING_FAMILIES)
def test_b_family_alone_keeps_published_plans(family, inst, published, env):
    laps, excess = per_vehicle_check(inst, published, {family}, env)
    assert laps == 14, family
    assert excess == pytest.approx(37, abs=1e-6), family
    assert ALPHA1 * laps + ALPHA2 * excess == pytest.approx(17.45, abs=1e-6), family


def test_b_family_B2_some_permutation_feasible(inst, published, env):
    found = compact_permutation_check(inst, published, {"B2"}, env)
    assert found is not None, "B2: no vehicle permutation of the published plans is feasible"
    assert found[1] == pytest.approx(17.45, abs=1e-6)


# --- c --------------------------------------------------------------------------

def test_c_all_families_per_vehicle(inst, published, env):
    laps, excess = per_vehicle_check(inst, published, True, env)
    assert (laps, excess) == (14, pytest.approx(37, abs=1e-6))


def test_c_all_families_compact(inst, published, env):
    found = compact_permutation_check(inst, published, True, env)
    assert found is not None, "all families: no vehicle permutation is feasible"
    assert found[1] == pytest.approx(17.45, abs=1e-6)


def test_b2_rejected_in_pricing(inst):
    with pytest.raises(ValueError):
        build_compact(replace(inst, V=1), pricing=True, valid_inequalities={"B2"})


# --- d --------------------------------------------------------------------------

def _pricing_models(inst, env):
    one = replace(inst, V=1)
    models = {}
    for flag in (False, True):
        m, v_ = build_compact(one, env=env, pricing=True, valid_inequalities=flag)
        m.Params.IntFeasTol = 1e-9
        m.Params.MIPGap = 0.0
        m.Params.MIPGapAbs = 1e-8
        models[flag] = (m, v_)
    return one, models


def _compare(one, models, scales, seed, count):
    rng = random.Random(seed)
    rows = []
    for k in range(count):
        scale = rng.choice(scales)
        pi = [rng.uniform(0, scale) for _ in range(one.n)]
        objs, times = {}, {}
        for flag, (m, v_) in models.items():
            for i in range(one.n):
                v_["y"][i].Obj = -ALPHA2 * one.q[i] * one.eps[i] - pi[i]
            t0 = time.perf_counter()
            m.optimize()
            times[flag] = time.perf_counter() - t0
            assert m.Status == GRB.OPTIMAL
            objs[flag] = m.ObjVal
        rows.append((k, scale, objs[False], objs[True], times[False], times[True]))
        assert objs[True] == pytest.approx(objs[False], abs=1e-6), (k, objs)
    return rows


def _show(rows):
    print("\n  k scale    rc_off     rc_on   t_off  t_on")
    for r in rows:
        print(f"  {r[0]} {r[1]:5.1f} {r[2]:9.4f} {r[3]:9.4f} {r[4]:6.2f} {r[5]:5.2f}")


def test_d_pricing_optimum_unchanged(inst, env):
    one, models = _pricing_models(inst, env)
    _show(_compare(one, models, [2.0, 5.0, 10.0], seed=20260923, count=10))


def test_d_pricing_optimum_unchanged_near_convergence(inst, env):
    """Small duals: optimum at or just below zero, the regime where pricing must
    prove that no negative column exists."""
    one, models = _pricing_models(inst, env)
    rows = _compare(one, models, [0.3, 0.6, 1.0], seed=7, count=5)
    _show(rows)
    assert any(r[2] > -1.0 for r in rows)
