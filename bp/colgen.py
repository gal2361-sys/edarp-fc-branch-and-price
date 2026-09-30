"""Column generation at a node.

The LP value is a valid node lower bound only when `converged` is True: a
prove=True pricing call (no restricted pricing, no early stop) on the TRUE LP duals
proved no column has reduced cost < -RC_TOL (a bound from truncated column
generation is never reported), and no artificial
column is active. Stages per iteration, each only if the previous added nothing:
LNS heuristic -> restricted MILP -> exact MILP with early stop; when the exact call
adds nothing, the next call is a single long prove call (Cutoff = -RC_TOL,
MIPFocus = 3). If that call times out without a column, column generation stops
and reports not converged; the Lagrangian bound uses its final bound.

The Lagrangian bound at any duals (pi, mu <= 0) that pricing was run on is

    LB(pi, mu) = sum_i pi_i + V * mu + V * min(0, pricing_bound(pi, mu))

and is valid at every iteration: any solution of the full master (real columns
only) has sum lambda <= V and every reduced cost >= pricing_bound, Gurobi's proven
lower bound on the minimum reduced cost, even under a time limit. For the true LP
duals, sum pi + V mu is the LP value. It bounds the full master LP, so it holds
regardless of artificial columns in the restricted master.

Stabilization (flag `stabilize`): Wentges smoothing. Pricing runs first on
pi_s = alpha * pi_c + (1 - alpha) * pi_lp (mu likewise). Pool columns with negative
reduced cost w.r.t. the TRUE duals are added. If there are none (mis-pricing),
pricing re-runs on the true duals before anything is concluded. The centre pi_c
starts at the first LP duals and moves to the duals at which the Lagrangian bound
last improved (pi_s after a smoothed call, pi_lp after a true one).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from bp.master import ART_COST_CAP, RestrictedMaster
from bp.pricing import RC_TOL, PricingMILP, PricingResult
from bp.pricing_lns import PricingLNS
from edarp import Instance


@dataclass
class ColGenResult:
    lp_value: float
    converged: bool               # prove call proved no negative column, no artificial
    pricing_proven: bool          # prove call proved no negative column
    artificial_active: bool
    stalled: bool                 # prove call optimal but its columns were all known
    prove_timed_out: bool         # prove call hit its limit without a column
    lagrangian_lb: float          # max over all pricing calls of the Lagrangian bound
    lagrangian_lb_iteration: int | None   # master iteration at which that max was reached
    iterations: int               # master LP solves
    pricing_calls: int
    mispricings: int              # smoothed calls that yielded no true-negative column
    artificial_escalations: int
    time_master: float
    time_pricing: float
    stages: dict = field(default_factory=dict)       # calls / hits per stage
    final_pi: list = field(default_factory=list)
    final_mu: float = 0.0
    positive_columns: list = field(default_factory=list)  # [(served, lambda)]
    prove_trajectories: list = field(default_factory=list)
    escalation_events: list = field(default_factory=list)
    log: list[dict] = field(default_factory=list)


def generate_columns(
    inst: Instance,
    master: RestrictedMaster,
    pricing: PricingMILP,
    time_limit: float,
    log: list[dict] | None = None,
    together=(),
    apart=(),
    verbose: bool = False,
    stabilize: bool = False,
    alpha: float = 0.5,
    artificial_escalation: float = 10.0,
    stall_iterations: int = 5,
    lns: PricingLNS | None = None,
    prove_time_limit: float | None = 1200.0,
) -> ColGenResult:
    log = [] if log is None else log
    V = inst.V
    t_start = time.perf_counter()
    base_pricing_limit = pricing.model.Params.TimeLimit
    base_prove_limit = pricing.prove_time_limit
    best_lb = float("-inf")
    best_lb_it = None
    lns_limit = lns.time_limit if lns is not None else None   # restored on exit
    t_master = t_pricing = 0.0
    prev_pi = None
    center: tuple[list[float], float] | None = None
    proven = stalled = prove_timed_out = False
    sol = None
    it = calls = mispricings = escalations = 0
    stages = {k: {"calls": 0, "hits": 0, "time": 0.0}
              for k in ("lns", "restricted", "exact", "prove", "smoothed")}
    trajectories = []
    prev_cols: list = []
    escalation_events: list[dict] = []
    streak = 0                 # consecutive iterations in which no stage added a column
    iter_added = None          # columns added in the previous iteration

    def escalate(why: str) -> bool:
        """Raise the artificial cost (x artificial_escalation, capped). Returns False
        if already at the cap."""
        nonlocal escalations
        if master.artificial_cost >= ART_COST_CAP:
            return False
        new = min(ART_COST_CAP, master.artificial_cost * artificial_escalation)
        master.set_artificial_cost(new)
        escalations += 1
        escalation_events.append({"iteration": it, "reason": why, "artificial_cost": new,
                                  "elapsed": time.perf_counter() - t_start})
        if verbose:
            print(f"   escalate artificial cost to {new:g} ({why})", flush=True)
        return True

    def left() -> float:
        return time_limit - (time.perf_counter() - t_start)

    def price(pi, mu, prove=False):
        nonlocal t_pricing, calls
        pricing.model.Params.TimeLimit = max(1e-3, min(base_pricing_limit, left()))
        if prove:
            cap = left() if prove_time_limit is None else min(left(), prove_time_limit)
            pricing.prove_time_limit = max(1e-3, cap)
        res = pricing.solve(pi, mu, together=together, apart=apart, prove=prove)
        t_pricing += res.time
        calls += 1
        return res

    def record(duals, res, pi, mu, added, t_m, dpi):
        nonlocal best_lb, best_lb_it, center
        # res.bound is the pricing MILP's proven dual bound (ObjBound, capped at the
        # cutoff on prove calls), never the best solution found; LNS and restricted
        # calls carry -inf and cannot raise the bound
        lb = sum(pi) + V * mu + V * min(0.0, res.bound)
        if lb > best_lb + 1e-9:
            best_lb, best_lb_it = lb, it
            center = (list(pi), mu)
        st = stages["smoothed" if duals == "smoothed" else res.kind]
        st["calls"] += 1
        st["time"] += res.time
        if added:
            st["hits"] += 1
        row = {
            "iteration": it,
            "duals": duals,
            "kind": res.kind,
            "lp_value": sol.value,
            "best_rc": res.best_rc,
            "pricing_bound": res.bound,
            "pricing_status": res.status,
            "columns_added": added,
            "total_columns": len(master.columns),
            "time_master": t_m,
            "time_pricing": res.time,
            "dual_change_l1": dpi,
            "lagrangian_lb": lb,
            "artificial_active": sol.artificial_active,
            "pruned": res.pruned,
            "elapsed": time.perf_counter() - t_start,
        }
        log.append(row)
        if verbose:
            rc = res.best_rc if res.best_rc is not None else float("nan")
            print(
                f"it {it:4d} {duals[:4]} {res.kind[:5]:5s} lp {sol.value:11.4f}  rc {rc:11.4f}"
                f"  bnd {res.bound:11.4f}  +{added:3d} cols {len(master.columns):5d}"
                f"  LB {lb:9.4f}  tP {res.time:6.2f}"
                f"  dpi {dpi if dpi is not None else float('nan'):9.2f}",
                flush=True,
            )

    prove_next = False
    try:
        while left() > 0:
            if iter_added is not None:
                streak = streak + 1 if iter_added == 0 else 0
                if (streak >= stall_iterations and sol.artificial_active
                        and escalate(f"{streak} iterations without a new column")):
                    streak = 0
                    prove_next = False
            iter_added = 0
            it += 1
            sol = master.solve()
            t_master += sol.time
            pi_lp, mu_lp = sol.pi, sol.mu
            dpi = None if prev_pi is None else sum(abs(a - b) for a, b in zip(pi_lp, prev_pi))
            prev_pi = pi_lp
            if center is None:
                center = (list(pi_lp), mu_lp)
            t_m = sol.time

            if stabilize and not prove_next and left() > 0:
                pi_c, mu_c = center
                pi_s = [alpha * c + (1 - alpha) * p for c, p in zip(pi_c, pi_lp)]
                mu_s = alpha * mu_c + (1 - alpha) * mu_lp
                res = price(pi_s, mu_s)
                true_neg = [
                    c for c in res.pool
                    if c.cost - sum(pi_lp[i] for i in c.served) - mu_lp < -RC_TOL
                ]
                added = sum(master.add_column(c) for c in true_neg)
                iter_added += added
                record("smoothed", res, pi_s, mu_s, added, t_m, dpi)
                t_m = 0.0
                if added > 0:
                    prev_cols = true_neg
                    continue
                mispricings += 1

            if lns is not None and not prove_next and left() > 0:
                seeds = [c for c, lam in zip(master.columns, sol.lambdas) if lam > 1e-9]
                t0 = time.perf_counter()
                lns.time_limit = min(lns_limit, max(0.0, left()))   # per call, from the caller's value
                cols = lns.solve(pi_lp, mu_lp, seeds + prev_cols, together, apart)
                rcs = [c.cost - sum(pi_lp[i] for i in c.served) - mu_lp for c in cols]
                el = time.perf_counter() - t0
                t_pricing += el
                calls += 1
                added = sum(master.add_column(c) for c in cols)
                iter_added += added
                res_l = PricingResult(
                    columns=cols, reduced_costs=rcs, best_rc=min(rcs) if rcs else None,
                    bound=float("-inf"), status="lns", time=el, pool=cols, kind="lns",
                )
                record("true", res_l, pi_lp, mu_lp, added, t_m, dpi)
                t_m = 0.0
                if added > 0:
                    prev_cols = cols
                    continue

            if left() <= 0:
                break
            res = price(pi_lp, mu_lp, prove=prove_next)
            added = sum(master.add_column(c) for c in res.columns)
            iter_added += added
            record("true", res, pi_lp, mu_lp, added, t_m, dpi)
            if res.columns:
                prev_cols = res.columns
            # stall with an artificial column active: the capped duals may be what
            # stops pricing, so raise the artificial cost before proving anything
            if (added == 0 and res.kind in ("exact", "prove") and sol.artificial_active
                    and not res.proven_no_negative
                    and escalate(f"{res.kind} call found nothing")):
                prove_next = False
                continue
            if res.kind == "prove":
                trajectories.append({"iteration": it, "status": res.status,
                                     "trajectory": res.bound_trajectory})
                if res.proven_no_negative:
                    if sol.artificial_active and escalate("proof with artificial active"):
                        # LP prefers an artificial even though pricing is done: raise
                        # its cost and keep going; the LP value is not yet a bound
                        prove_next = False
                        continue
                    proven = True
                    break
                if added == 0:
                    if res.status == "optimal":
                        stalled = True        # columns found but all already known
                    else:
                        prove_timed_out = True
                    break                     # one prove call, no retries
                prove_next = False
                continue
            prove_next = added == 0           # exact stage added nothing: prove next
    finally:
        pricing.model.Params.TimeLimit = base_pricing_limit
        if lns is not None:
            lns.time_limit = lns_limit
        pricing.prove_time_limit = base_prove_limit

    if sol is None:
        sol = master.solve()
        t_master += sol.time
    art = sol.artificial_active
    return ColGenResult(
        lp_value=sol.value,
        converged=proven and not art,
        pricing_proven=proven,
        artificial_active=art,
        stalled=stalled,
        prove_timed_out=prove_timed_out,
        lagrangian_lb=best_lb,
        lagrangian_lb_iteration=best_lb_it,
        iterations=it,
        pricing_calls=calls,
        mispricings=mispricings,
        artificial_escalations=escalations,
        time_master=t_master,
        time_pricing=t_pricing,
        stages=stages,
        final_pi=list(sol.pi),
        final_mu=sol.mu,
        positive_columns=[
            (sorted(c.served), lam) for c, lam in zip(master.columns, sol.lambdas) if lam > 1e-9
        ],
        prove_trajectories=trajectories,
        escalation_events=escalation_events,
        log=log,
    )
