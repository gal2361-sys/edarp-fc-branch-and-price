"""Constructive full-fleet heuristic: at most V columns covering every request.

1. Sequential: for vehicle k = 1..V, PricingLNS with pi_i = M (large, equal) for every
   unserved request and 0 otherwise, mu = 0, seeded with the unserved singletons. The
   reduced cost is cost - M * (requests served), so the search maximises the number of
   unserved requests served (then the cost). Requests with pi_i = 0 are never inserted,
   so the plans are disjoint. Keep the best plan; mark its requests served.
2. Repair: insert each uncovered request at its cheapest feasible position in any of
   the V plans (including an empty one), re-evaluated exactly.
   A request that fits nowhere is placed by an ejection: remove one request r from a
   plan, insert j there, and reinsert r in another plan (cheapest feasible combination).
3. If requests remain: ruin-and-recreate over the whole fleet for a bounded time:
   remove 2-4 random requests from one plan and greedily reinsert every uncovered
   request (with ejections). A move is accepted if it leaves no more requests
   uncovered (plateau moves allowed regardless of cost, to escape); the best state
   seen (fewest uncovered, then cost) is returned.
Every cost comes from bp.column.evaluate(); the returned total is exact.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from bp.column import Column
from bp.pricing_lns import PricingLNS
from edarp import Instance


@dataclass
class InitialSolution:
    columns: list[Column]              # non-empty plans, at most V
    covered: bool                      # every request served exactly once
    uncovered: list[int]
    cost: float                        # exact total cost of `columns`
    time: float
    stages: dict = field(default_factory=dict)   # uncovered count after each stage


def initial_solution(inst: Instance, time_per_vehicle: float = 5.0,
                     repair_time: float = 60.0, seed: int = 0,
                     big_m: float = 1000.0) -> InitialSolution:
    t0 = time.perf_counter()
    V, n = inst.V, inst.n
    rng = random.Random(seed)
    lns = PricingLNS(inst, time_limit=time_per_vehicle, pool_size=10, seed=seed)
    plans: list[tuple] = [() for _ in range(V)]
    served: set[int] = set()
    stages = {}

    # 1. sequential
    for k in range(V):
        unserved = [i for i in range(n) if i not in served]
        if not unserved:
            break
        pi = [big_m if i not in served else 0.0 for i in range(n)]
        seeds = [c for c in (lns.singleton(i) for i in unserved) if c is not None]
        lns.rng = random.Random(seed + k)
        cols = lns.solve(pi, 0.0, seeds)
        if not cols:
            continue
        best = cols[0]
        plans[k] = lns._normalize(lns.from_column(best))
        served |= best.served
    stages["after_sequential"] = n - len(served)

    def cost_of(plan):
        if not plan:
            return 0.0
        col = lns.evaluate(plan)
        return None if col is None else col.cost

    def best_insertion(plans, j):
        """(delta, k, new_plan) of the cheapest feasible insertion of j, or None."""
        best = None
        for k, plan in enumerate(plans):
            base = cost_of(plan)
            for cand in lns._insertions(plan, j):
                c = cost_of(cand)
                if c is not None and (best is None or c - base < best[0]):
                    best = (c - base, k, cand)
        return best

    def ejection(plans, j):
        """Place j by ejecting one request r from a plan k and reinserting r in a plan
        k2 != k. Returns the new plans or None."""
        best = None
        for k, plan in enumerate(plans):
            for r in [i for picks, _ in plan for i in picks]:
                shrunk = lns._normalize(tuple((p - {r}, c) for p, c in plan))
                if shrunk and cost_of(shrunk) is None:
                    continue
                trial = list(plans)
                trial[k] = shrunk
                ins_j = best_insertion(trial, j)
                if ins_j is None:
                    continue
                trial[ins_j[1]] = ins_j[2]
                others = [p if kk != k else () for kk, p in enumerate(trial)]
                ins_r = best_insertion(others, r)       # r may not return to plan k
                if ins_r is None or ins_r[1] == k:
                    continue
                trial[ins_r[1]] = ins_r[2]
                tot = sum(cost_of(p) for p in trial)
                if best is None or tot < best[0]:
                    best = (tot, trial)
        return None if best is None else best[1]

    def greedy(plans, uncovered):
        plans, uncovered = list(plans), set(uncovered)
        while uncovered:
            options = [(ins, j) for j in uncovered if (ins := best_insertion(plans, j))]
            if options:
                (delta, k, cand), j = min(options, key=lambda o: o[0][0])
                plans[k] = cand
                uncovered.discard(j)
                continue
            for j in sorted(uncovered):
                new = ejection(plans, j)
                if new is not None:
                    plans = new
                    uncovered.discard(j)
                    break
            else:
                break
        return plans, uncovered

    # 2. repair by cheapest insertion
    plans, uncovered = greedy(plans, {i for i in range(n) if i not in served})
    stages["after_repair"] = len(uncovered)

    # 3. ruin-and-recreate over the fleet
    deadline = time.perf_counter() + repair_time
    total = lambda ps: sum(cost_of(p) for p in ps)
    best_plans, best_unc = list(plans), set(uncovered)
    while uncovered and time.perf_counter() < deadline:
        nonempty = [k for k, p in enumerate(plans) if p]
        if not nonempty:
            break
        k = rng.choice(nonempty)
        reqs = [i for picks, _ in plans[k] for i in picks]
        drop = set(rng.sample(reqs, min(len(reqs), rng.randint(2, 4))))
        trial = list(plans)
        trial[k] = lns._normalize(tuple((p - drop, c) for p, c in plans[k]))
        if trial[k] and cost_of(trial[k]) is None:
            continue
        trial, left = greedy(trial, uncovered | drop)
        if len(left) <= len(uncovered):             # plateau moves allowed
            plans, uncovered = trial, left
            if (len(left), total(trial)) < (len(best_unc), total(best_plans)):
                best_plans, best_unc = list(trial), set(left)
    if (len(best_unc), total(best_plans)) < (len(uncovered), total(plans)):
        plans, uncovered = best_plans, best_unc
    stages["after_ruin_recreate"] = len(uncovered)

    columns = [lns.evaluate(p) for p in plans if p]
    assert all(c is not None for c in columns)
    covered_set = [i for c in columns for i in c.served]
    assert len(covered_set) == len(set(covered_set)), "a request is served twice"
    return InitialSolution(
        columns=columns, covered=not uncovered, uncovered=sorted(uncovered),
        cost=sum(c.cost for c in columns), time=time.perf_counter() - t0, stages=stages,
    )
