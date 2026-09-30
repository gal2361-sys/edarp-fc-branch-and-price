"""LNS pricing heuristic (never a proof; the MILP stages stay the exact fallback).

A plan is a tuple of laps, each (pickups: frozenset[int], charge: float). Stops are
derived: the pickup stations of the lap's pickups, the dropoff stations of requests
dropped on the lap (same-lap pickups, and depot-crossing pickups of the previous
lap), and the depot when charge > 0. Feasibility and cost always come from
bp.column.evaluate(); this module never computes a cost itself.

Search per call, within `time_limit` seconds over all seeds:
  1. first-improvement local search on reduced cost from each seed, with moves:
     remove a request; insert an unserved request (into an existing lap, or into
     appended laps); move a request to another lap; shift one unit of charge
     between consecutive laps; drop an empty lap;
  2. ruin-and-recreate on the local optima: remove 2-4 random requests, then
     greedily reinsert by best reduced-cost gain; keep the result if not worse.
Requests with pi_i <= 0 are never inserted (they cannot lower the reduced cost).
Every evaluated plan with reduced cost < -RC_TOL is collected; results are
filtered by the branching decisions and returned best first.
"""

from __future__ import annotations

import random
import time

from bp.column import Column, LapPlan, evaluate, lap_plans_from_pickups
from bp.pricing import RC_TOL
from edarp import Instance

Plan = tuple  # tuple[tuple[frozenset[int], float], ...]


class PricingLNS:
    def __init__(self, inst: Instance, time_limit: float = 2.0, pool_size: int = 50,
                 seed: int = 0, max_seeds: int = 20) -> None:
        self.inst = inst
        self.time_limit = time_limit
        self.pool_size = pool_size
        self.max_seeds = max_seeds
        self.rng = random.Random(seed)
        self._cache: dict[Plan, Column | None] = {}
        self.stats = {"calls": 0, "time": 0.0, "evaluations": 0, "hits": 0}

    # ---- plan <-> column -------------------------------------------------------

    def to_lap_plans(self, plan: Plan) -> list[LapPlan]:
        return lap_plans_from_pickups(self.inst, plan)

    def evaluate(self, plan: Plan) -> Column | None:
        col = self._cache.get(plan, False)
        if col is False:
            self.stats["evaluations"] += 1
            col = evaluate(self.inst, self.to_lap_plans(plan))
            if len(self._cache) > 500_000:
                self._cache.clear()
            self._cache[plan] = col
        return col

    @staticmethod
    def from_column(col: Column) -> Plan:
        return tuple((frozenset(p), c) for p, c in zip(col.pickups, col.charge))

    def singleton(self, i: int) -> Column | None:
        """Cheapest feasible one-request plan, for seeding."""
        bd, bm = self.inst.beta_d, self.inst.beta_m
        if self.inst.I[i] == 0:
            options = [((frozenset({i}), bd),)]
        else:  # crossing: cannot charge between the two laps (passenger on board)
            options = [((frozenset({i}), min(bm, 2 * bd)), (frozenset(), 0.0))]
        cols = [c for c in (self.evaluate(p) for p in options) if c is not None]
        return min(cols, key=lambda c: c.cost) if cols else None

    # ---- neighbourhood ---------------------------------------------------------

    def _normalize(self, plan: Plan) -> Plan:
        """Drop trailing laps with nothing on them."""
        inst = self.inst
        laps = list(plan)
        while laps:
            picks, _ = laps[-1]
            carried = len(laps) > 1 and any(inst.I[i] == 1 for i in laps[-2][0])
            if picks or carried:
                break
            laps.pop()
        return tuple(laps)

    def _insertions(self, plan: Plan, j: int):
        inst, bd, bm = self.inst, self.inst.beta_d, self.inst.beta_m
        n = len(plan)
        for l in range(n):
            picks, ch = plan[l]
            new = list(plan)
            new[l] = (picks | {j}, ch)
            if inst.I[j] == 1 and l == n - 1:
                # dropoff lap does not exist yet: append it. The vehicle carries j
                # across the depot, so the new lap cannot charge; also try moving
                # beta_d of charge onto lap l instead (evaluate decides).
                yield tuple(new) + ((frozenset(), bd),)
                new2 = list(new)
                new2[l] = (picks | {j}, min(bm, ch + bd))
                yield tuple(new2) + ((frozenset(), 0.0),)
            else:
                yield tuple(new)
        # a fresh lap (or two, if j crosses the depot) at the end
        if inst.I[j] == 0:
            yield plan + ((frozenset({j}), bd),)
        else:
            yield plan + ((frozenset({j}), min(bm, 2 * bd)), (frozenset(), 0.0))

    def _neighbours(self, plan: Plan, candidates: list[int]):
        inst = self.inst
        n = len(plan)
        served = {i for picks, _ in plan for i in picks}
        moves = []
        for l, (picks, ch) in enumerate(plan):
            for i in picks:
                moves.append(("remove", l, i))
                for l2 in range(n):
                    if l2 != l and not (inst.I[i] == 1 and l2 == n - 1):
                        moves.append(("move", l, i, l2))
        for j in candidates:
            if j not in served:
                moves.append(("insert", j))
        for l in range(n - 1):
            moves.append(("shift", l, 1))
            moves.append(("shift", l, -1))
        for l in range(n):
            moves.append(("droplap", l))
        self.rng.shuffle(moves)
        for mv in moves:
            kind = mv[0]
            if kind == "remove":
                _, l, i = mv
                new = list(plan)
                new[l] = (plan[l][0] - {i}, plan[l][1])
                yield self._normalize(tuple(new))
            elif kind == "move":
                _, l, i, l2 = mv
                new = list(plan)
                new[l] = (plan[l][0] - {i}, plan[l][1])
                new[l2] = (plan[l2][0] | {i}, plan[l2][1])
                yield self._normalize(tuple(new))
            elif kind == "insert":
                yield from self._insertions(plan, mv[1])
            elif kind == "shift":
                _, l, d = mv
                a, b = plan[l][1] - d, plan[l + 1][1] + d
                if a >= 0 and b >= 0:
                    new = list(plan)
                    new[l] = (plan[l][0], a)
                    new[l + 1] = (plan[l + 1][0], b)
                    yield tuple(new)
            elif kind == "droplap":
                l = mv[1]
                carried = l > 0 and any(inst.I[i] == 1 for i in plan[l - 1][0])
                if plan[l][0] or carried:
                    continue
                new = list(plan[:l]) + list(plan[l + 1:])
                if l < len(new):
                    new[l] = (new[l][0], min(inst.beta_m, new[l][1] + plan[l][1]))
                yield self._normalize(tuple(new))

    # ---- search ----------------------------------------------------------------

    def solve(self, pi: list[float], mu: float, seeds: list[Column],
              together=(), apart=()) -> list[Column]:
        t0 = time.perf_counter()
        deadline = t0 + self.time_limit
        inst = self.inst
        candidates = [i for i in range(inst.n) if pi[i] > 0]
        found: dict[Column, float] = {}

        def rc_of(plan: Plan) -> float | None:
            col = self.evaluate(plan)
            if col is None:
                return None
            rc = col.cost - sum(pi[i] for i in col.served) - mu
            if rc < -RC_TOL:
                found[col] = rc
            return rc

        def local_search(plan: Plan, rc: float) -> tuple[Plan, float]:
            improved = True
            while improved and time.perf_counter() < deadline:
                improved = False
                for nb in self._neighbours(plan, candidates):
                    if time.perf_counter() >= deadline:
                        break
                    r = rc_of(nb)
                    if r is not None and r < rc - 1e-9:
                        plan, rc, improved = nb, r, True
                        break
            return plan, rc

        def recreate(plan: Plan, rc: float) -> tuple[Plan, float]:
            while time.perf_counter() < deadline:
                served = {i for picks, _ in plan for i in picks}
                best = None
                for j in candidates:
                    if j in served:
                        continue
                    for nb in self._insertions(plan, j):
                        r = rc_of(nb)
                        if r is not None and (best is None or r < best[1]):
                            best = (nb, r)
                    if time.perf_counter() >= deadline:
                        break
                if best is None or best[1] >= rc - 1e-9:
                    break
                plan, rc = best
            return plan, rc

        # seeds: best reduced cost first, capped
        scored = []
        for col in {c for c in seeds}:
            plan = self._normalize(self.from_column(col))
            r = rc_of(plan)
            if r is not None:
                scored.append((r, plan))
        scored.sort(key=lambda t: t[0])
        scored = scored[: self.max_seeds]

        optima = []
        for r, plan in scored:
            if time.perf_counter() >= deadline:
                break
            optima.append(local_search(plan, r))

        while optima and time.perf_counter() < deadline:
            k = self.rng.randrange(len(optima))
            plan, rc = optima[k]
            served = [i for picks, _ in plan for i in picks]
            if not served:
                continue
            ruined = list(plan)
            for i in self.rng.sample(served, min(len(served), self.rng.randint(2, 4))):
                ruined = [(p - {i}, c) for p, c in ruined]
            ruined = self._normalize(tuple(ruined))
            r0 = rc_of(ruined) if ruined else None
            if r0 is None:
                continue
            new, r_new = recreate(ruined, r0)
            if r_new <= rc + 1e-9:
                optima[k] = (new, r_new)

        def ok(col: Column) -> bool:
            s = col.served
            return all((i in s) == (j in s) for i, j in together) and not any(
                i in s and j in s for i, j in apart
            )

        out = sorted(((rc, c) for c, rc in found.items() if ok(c)), key=lambda t: t[0])
        cols = [c for _, c in out[: self.pool_size]]
        self.stats["calls"] += 1
        self.stats["time"] += time.perf_counter() - t0
        if cols:
            self.stats["hits"] += 1
        return cols
