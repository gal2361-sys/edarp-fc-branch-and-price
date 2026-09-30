"""A column is one vehicle's complete plan over the horizon.

Column costs come only from `evaluate`, an exact forward pass over a discrete plan.
Solver values are rounded before evaluate is called, never inside it, so a column's
cost never inherits Gurobi's integrality or big-M tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp

from edarp import ALPHA1, ALPHA2, Instance

_ROUND = 6    # decimals kept for charge durations
TOL = 1e-6    # slack on continuous checks (battery, deadlines) after that rounding


@dataclass(frozen=True)
class LapPlan:
    performed: bool
    charge: float                 # charging duration b_l before the lap
    stops: frozenset[int]         # stations stopped at, 0 = depot
    pickups: frozenset[int]       # requests picked up on this lap


@dataclass(frozen=True, eq=False)
class Column:
    """One vehicle's plan. Per-lap tuples cover the performed laps 0..n_laps-1.

    stops[l] holds the stations stopped at on lap l. Station 0 is included only
    when the vehicle charges before lap l (charge[l] > 0): a depot stop without
    charging changes nothing, so it is dropped to keep equal plans equal.
    Equality and hash use (served, n_laps, stops, pickups, charge).
    """

    served: frozenset[int]
    n_laps: int
    stops: tuple[frozenset[int], ...]
    pickups: tuple[frozenset[int], ...]
    charge: tuple[float, ...]
    cost: float
    excess: float

    def _key(self) -> tuple:
        return (self.served, self.n_laps, self.stops, self.pickups, self.charge)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Column) and self._key() == other._key()

    def __hash__(self) -> int:
        # frozensets/tuples of ints and floats hash deterministically across runs
        return hash(self._key())

    def covers(self, i: int) -> bool:
        return i in self.served

    def plan(self) -> list[LapPlan]:
        return [
            LapPlan(True, self.charge[l], self.stops[l], self.pickups[l])
            for l in range(self.n_laps)
        ]

    @classmethod
    def from_model(
        cls, inst: Instance, model: gp.Model, vars: dict, sol: int | None = None,
        reason: list[str] | None = None,
    ) -> "Column | None":
        """Round a solved V=1 model's plan, then evaluate it exactly.

        sol=k reads pool solution k instead of the best. Returns None if the
        rounded plan is infeasible (the solver's values relied on tolerances).
        """
        return evaluate(inst, plan_from_model(inst, model, vars, sol), reason=reason)


def _round_charge(b: float) -> float:
    c = round(max(b, 0.0), _ROUND)
    return 0.0 if c == 0 else c


def plan_from_model(
    inst: Instance, model: gp.Model, vars: dict, sol: int | None = None
) -> list[LapPlan]:
    """Rounded discrete plan of a solved V=1 model (all L laps)."""
    if sol is not None:
        model.Params.SolutionNumber = sol
        attr = "Xn"
    else:
        attr = "X"

    def vals(td):
        return dict(zip(td.keys(), model.getAttr(attr, list(td.values()))))

    x, u, w, b = vals(vars["x"]), vals(vars["u"]), vals(vars["w"]), vals(vars["b"])
    plan = []
    for l in range(inst.L):
        performed = w[l, 0] > 0.5
        plan.append(
            LapPlan(
                performed=performed,
                charge=_round_charge(b[l, 0]) if performed else 0.0,
                stops=frozenset(s for s in range(inst.S) if x[s, l, 0] > 0.5),
                pickups=frozenset(i for i in range(inst.n) if u[i, l, 0] > 0.5),
            )
        )
    return plan


def evaluate(
    inst: Instance, plan: list[LapPlan], reason: list[str] | None = None
) -> Column | None:
    """Exact schedule of a discrete single-vehicle plan, or None if infeasible.

    Earliest-time forward pass, which is optimal for a fixed plan: every timing
    constraint is a lower bound except the deadlines d_i, so waiting never helps.
      - T = 0 at the depot on lap 0. Before each performed lap, charge b_l at the
        depot (requires a depot stop and zero load); the depot has no service time.
      - Stations 1..S-1 in order: arrival = previous departure + t[s]; dropoffs
        happen at arrival (that is tau_i); service start = max(arrival, a_i of the
        pickups here); a stop adds sigma. Lap end returns to the depot via t[0].
      - Request i picked up on lap l is dropped on lap l + I(i).
    Checks: laps consecutive and <= L; stops at every pickup and dropoff station;
    load <= C; battery in [0, beta_m] (B_0 = 0); tau_i <= d_i.
    If `reason` is a list, the name of the first violated check is appended to it.
    """

    def fail(why: str):
        if reason is not None:
            reason.append(why)
        return None

    S, n, L = inst.S, inst.n, inst.L
    t, sigma = inst.t, inst.sigma
    if len(plan) > L:
        if any(p.performed or p.stops or p.pickups or p.charge > 0 for p in plan[L:]):
            return fail("laps_beyond_L")
        plan = plan[:L]

    n_laps = 0
    for p in plan:
        if p.performed:
            n_laps += 1
        else:
            break
    for p in plan[n_laps:]:
        if p.performed or p.pickups or p.charge > 0:
            return fail("nonconsecutive_laps")   # nothing happens on idle laps
    laps = plan[:n_laps]

    # dropoffs per (lap, station)
    drop_at: dict[tuple[int, int], list[int]] = {}
    served: set[int] = set()
    for l, p in enumerate(laps):
        for i in p.pickups:
            if i in served:
                return fail("duplicate_pickup")
            served.add(i)
            if inst.pickup_station[i] not in p.stops:
                return fail("missing_pickup_stop")
            ld = l + inst.I[i]
            if ld >= n_laps:
                return fail("drop_lap_not_performed")
            if inst.drop_station[i] not in laps[ld].stops:
                return fail("missing_drop_stop")
            drop_at.setdefault((ld, inst.drop_station[i]), []).append(i)
    board_at: dict[tuple[int, int], list[int]] = {}
    for l, p in enumerate(laps):
        for i in p.pickups:
            board_at.setdefault((l, inst.pickup_station[i]), []).append(i)

    time = 0.0
    load = 0
    battery = 0.0
    tau: dict[int, float] = {}
    stops_out, charge_out = [], []
    for l, p in enumerate(laps):
        if l > 0:
            time += t[0]                               # back to the depot
        depot_stop = 0 in p.stops
        if depot_stop and load != 0:
            return fail("depot_stop_not_empty")        # (13)
        if p.charge > 0:
            if not depot_stop:
                return fail("charge_without_depot_stop")  # (8)
            if battery + inst.beta_c * p.charge > inst.beta_m + TOL:
                return fail("battery_above_max")       # (9)
            battery += inst.beta_c * p.charge
            time += p.charge
        battery -= inst.beta_d                         # (6) discharge over the lap
        if battery < -TOL:
            return fail("battery_below_zero")         # (6)
        for s in range(1, S):
            time += t[s]                               # arrival at s
            for i in drop_at.get((l, s), ()):
                tau[i] = time
                if time > inst.d[i] + TOL:
                    return fail("deadline")            # (18)
                load -= inst.q[i]
            if s in p.stops:
                picks = board_at.get((l, s), ())
                for i in picks:
                    time = max(time, inst.a[i])        # (16)
                    load += inst.q[i]
                if load > inst.C:
                    return fail("capacity")            # (12)
                time += sigma
        stops_out.append(frozenset(s for s in p.stops if s != 0 or p.charge > 0))
        charge_out.append(p.charge)

    excess = sum(inst.q[i] * (tau[i] - inst.eps[i]) for i in served)
    return Column(
        served=frozenset(served),
        n_laps=n_laps,
        stops=tuple(stops_out),
        pickups=tuple(p.pickups for p in laps),
        charge=tuple(charge_out),
        cost=ALPHA1 * n_laps + ALPHA2 * excess,
        excess=excess,
    )


def lap_plans_from_pickups(inst: Instance, plan) -> list[LapPlan]:
    """Discrete plan from (pickups, charge) per performed lap; stops are derived:
    the pickup stations of the lap's pickups, the dropoff stations of requests
    dropped on the lap (same-lap pickups and depot-crossing pickups of the
    previous lap), and the depot when charge > 0."""
    out = []
    for l, (picks, charge) in enumerate(plan):
        stops = {inst.pickup_station[i] for i in picks}
        stops |= {inst.drop_station[i] for i in picks if inst.I[i] == 0}
        if l > 0:
            stops |= {inst.drop_station[i] for i in plan[l - 1][0] if inst.I[i] == 1}
        if charge > 0:
            stops.add(0)
        out.append(LapPlan(True, charge, frozenset(stops), frozenset(picks)))
    return out


def plan_from_published(inst: Instance, laps: list[dict]) -> list[LapPlan]:
    """Discrete plan of one vehicle of a parsed published solution (bp.published)."""
    n = inst.n
    by_lap = {lap["l"]: lap for lap in laps}
    n_rows = max([inst.L] + [l + 1 for l in by_lap])
    plan = []
    for l in range(n_rows):
        lap = by_lap.get(l, {"b": 0.0, "nodes": []})
        stations = {st for _, _, _, st in lap["nodes"]}
        if lap["b"] > 0:
            stations.add(0)
        plan.append(
            LapPlan(
                performed=bool(lap["nodes"]) or lap["b"] > 0,
                charge=_round_charge(lap["b"]),
                stops=frozenset(stations),
                pickups=frozenset(node for node, *_ in lap["nodes"] if node < n),
            )
        )
    return plan


def column_from_published_plan(inst: Instance, laps: list[dict], env=None) -> Column:
    """Evaluate one vehicle of a parsed published solution. Raises if infeasible."""
    col = evaluate(inst, plan_from_published(inst, laps))
    if col is None:
        raise ValueError("published plan is infeasible under evaluate()")
    return col
