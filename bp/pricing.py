"""Pricing: single-vehicle MILP with a dual-adjusted objective.

Minimizes  cost(plan) - sum_{i served} pi_i - mu  over single-vehicle plans that
respect the branching decisions. A negative minimum gives new columns. A node
lower bound may only be recorded after a call returns proven_no_negative: status
optimal (or cutoff) and MIP bound >= -tol.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace

import gurobipy as gp
from gurobipy import GRB

from bp.column import Column
from edarp import ALPHA2, Instance, build_compact

RC_TOL = 1e-6


@dataclass
class PricingResult:
    columns: list[Column]            # exact reduced cost < -RC_TOL, sorted ascending
    reduced_costs: list[float]       # parallel to columns
    best_rc: float | None            # best exact reduced cost over evaluated pool columns
    bound: float                     # MIP lower bound on the minimum reduced cost
    status: str                      # "optimal", "time_limit", or Gurobi status code
    time: float                      # wall-clock of this call, seconds
    pool: list[Column] = field(default_factory=list)  # every evaluated pool column
    kind: str = "exact"              # "lns", "restricted", "exact" or "prove"
    pruned: int = 0                  # requests excluded because pi_i <= 0
    bound_trajectory: list = field(default_factory=list)  # prove calls: [(time, bound)]

    @property
    def proven_no_negative(self) -> bool:
        """True only if pricing proved that no column has reduced cost < -RC_TOL.

        Requires a complete exact solve: a restricted call has bound -inf and an
        early-stopped call has status "early_stop", so neither can qualify.
        """
        return self.status in ("optimal", "cutoff") and self.bound >= -RC_TOL


@dataclass
class PricingMILP:
    inst: Instance
    time_limit: float = 30.0
    pool_size: int = 10
    int_feas_tol: float = 1e-9   # tight: IntFeasTol * BIG_T leaks into tau via (17)
    # Electronic Companion B families (B2 excluded in pricing); on by default after
    # tests/test_valid_inequalities.py checks a-d passed
    valid_inequalities: bool | frozenset = True
    # heuristic pricing: first solve over the k requests with the largest duals
    restricted_k: int | None = 20
    restricted_time_limit: float = 5.0
    # exact call stops after this many improving solutions with reduced cost < -RC_TOL
    early_stop: int | None = 5
    # exact pruning: requests with pi_i <= 0 cannot lower the reduced cost
    dual_pruning: bool = True
    # prove calls: Cutoff = -RC_TOL, MIPFocus = 3, this time limit (None: TimeLimit)
    prove_time_limit: float | None = None
    trajectory_interval: float = 5.0
    env: gp.Env | None = None
    stats: dict = field(
        default_factory=lambda: {
            "calls": 0, "time": 0.0, "times": [], "dropped": 0,
            "dropped_reasons": {},
            "restricted": 0, "restricted_hits": 0, "time_restricted": 0.0,
            "exact": 0, "time_exact": 0.0, "prove": 0, "time_prove": 0.0,
            "early_stopped": 0, "pruned": 0, "prove_hits": 0, "exact_hits": 0,
        }
    )

    def __post_init__(self) -> None:
        self.one = replace(self.inst, V=1)
        if self.env is None:
            self.env = gp.Env(params={"OutputFlag": 0})
        self.model, self.vars = build_compact(
            self.one, env=self.env, pricing=True, valid_inequalities=self.valid_inequalities
        )
        y = self.vars["y"]
        # coefficient of y_i in the column cost; pi_i is subtracted from it per call
        self._y_base = [-ALPHA2 * self.one.q[i] * self.one.eps[i] for i in range(self.one.n)]
        for i in range(self.one.n):
            y[i].Obj = self._y_base[i]
        self.model.Params.PoolSearchMode = 2
        self.model.Params.PoolSolutions = self.pool_size
        self.model.Params.TimeLimit = self.time_limit
        self.model.Params.IntFeasTol = self.int_feas_tol
        self.model.update()

    def _collect(self, pi, mu) -> dict:
        """Evaluate every pool solution exactly; reduced cost w.r.t. the full pi."""
        m = self.model
        best: dict[Column, float] = {}
        for k in range(m.SolCount):
            reason: list[str] = []
            col = Column.from_model(self.one, m, self.vars, sol=k, reason=reason)
            if col is None:
                # rounded plan infeasible: the solver relied on tolerances
                self.stats["dropped"] += 1
                r = reason[0] if reason else "unknown"
                self.stats["dropped_reasons"][r] = self.stats["dropped_reasons"].get(r, 0) + 1
                continue
            rc = col.cost - sum(pi[i] for i in col.served) - mu
            if col not in best or rc < best[col]:
                best[col] = rc
        return best

    def _restricted(self, pi, mu) -> dict:
        """Heuristic solve over the k requests with the largest duals."""
        m, u, n, L = self.model, self.vars["u"], self.one.n, self.one.L
        keep = set(sorted(range(n), key=lambda i: (-pi[i], i))[: self.restricted_k])
        saved = []
        for i in range(n):
            if i in keep:
                continue
            for l in range(L):
                var = u[i, l, 0]
                saved.append((var, var.UB))   # keep B3/B4 bounds: restore, never reset to 1
                var.UB = 0.0
        old_limit, old_focus = m.Params.TimeLimit, m.Params.MIPFocus
        m.Params.TimeLimit = min(self.restricted_time_limit, old_limit)
        m.Params.MIPFocus = 1
        try:
            m.optimize()
            return self._collect(pi, mu)
        finally:
            for var, ub in saved:
                var.UB = ub
            m.Params.TimeLimit = old_limit
            m.Params.MIPFocus = old_focus
            m.update()

    def solve(
        self,
        pi: list[float],
        mu: float,
        together: tuple | list = (),
        apart: tuple | list = (),
        prove: bool = False,
    ) -> PricingResult:
        """Price one dual vector.

        prove=False: restricted pricing first (if enabled); if it finds a negative
        column, return those (kind "restricted", bound -inf). Otherwise an exact
        solve, stopped early after `early_stop` negative solutions (bound = MIP
        bound at termination, still valid). prove=True: one exact solve with
        neither acceleration, the only kind that may prove convergence.
        """
        t0 = time.perf_counter()
        m, y, u, n, L = self.model, self.vars["y"], self.vars["u"], self.one.n, self.one.L
        assert len(pi) == n
        for i in range(n):
            y[i].Obj = self._y_base[i] - pi[i]
        m.ObjCon = -mu

        branch = []
        for i, j in together:
            branch.append(m.addConstr(y[i] == y[j], name=f"br_together_{i}_{j}"))
        for i, j in apart:
            branch.append(m.addConstr(y[i] + y[j] <= 1, name=f"br_apart_{i}_{j}"))

        # exact dual pruning. Serving i adds cost >= 0 (laps, excess) and gains pi_i,
        # so pi_i <= 0 never lowers the reduced cost. Not valid for a request in a
        # "together" pair (it would also exclude its partner), so those are kept.
        pruned_saved = []
        if self.dual_pruning:
            tied = {k for pair in together for k in pair}
            for i in range(n):
                if pi[i] <= 0 and i not in tied:
                    for l in range(L):
                        pruned_saved.append((u[i, l, 0], u[i, l, 0].UB))  # keeps B3/B4
                        u[i, l, 0].UB = 0.0
        n_pruned = len(pruned_saved) // L if L else 0
        self.stats["pruned"] += n_pruned

        kind = "prove" if prove else "exact"
        trajectory: list = []
        try:
            best = None
            if not prove and self.restricted_k is not None and self.restricted_k < n:
                tr = time.perf_counter()
                cand = self._restricted(pi, mu)
                self.stats["restricted"] += 1
                self.stats["time_restricted"] += time.perf_counter() - tr
                if any(rc < -RC_TOL for rc in cand.values()):
                    self.stats["restricted_hits"] += 1
                    best, kind = cand, "restricted"
                    status, bound = "restricted", float("-inf")
            if best is None:
                te = time.perf_counter()
                stopped = False
                old = (m.Params.TimeLimit, m.Params.MIPFocus, m.Params.Cutoff)
                if prove:
                    if self.prove_time_limit is not None:
                        m.Params.TimeLimit = self.prove_time_limit
                    m.Params.MIPFocus = 3
                    m.Params.Cutoff = -RC_TOL
                    last = [-1e9]

                    def cb(model, where):
                        if where == GRB.Callback.MIP:
                            rt = model.cbGet(GRB.Callback.RUNTIME)
                            if rt - last[0] >= self.trajectory_interval:
                                last[0] = rt
                                trajectory.append((rt, model.cbGet(GRB.Callback.MIP_OBJBND)))
                elif self.early_stop:
                    target = self.early_stop
                    count = [0]

                    def cb(model, where):
                        nonlocal stopped
                        if where == GRB.Callback.MIPSOL:
                            if model.cbGet(GRB.Callback.MIPSOL_OBJ) < -RC_TOL:
                                count[0] += 1
                                if count[0] >= target:
                                    stopped = True
                                    model.terminate()
                else:
                    cb = None
                try:
                    m.optimize(cb) if cb is not None else m.optimize()
                finally:
                    m.Params.TimeLimit, m.Params.MIPFocus, m.Params.Cutoff = old
                if stopped and m.Status == GRB.INTERRUPTED:
                    status = "early_stop"
                    self.stats["early_stopped"] += 1
                else:
                    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
                              GRB.CUTOFF: "cutoff"}.get(m.Status, str(m.Status))
                best = self._collect(pi, mu)
                if status == "cutoff":
                    bound = -RC_TOL        # proven: no plan has reduced cost < cutoff
                elif prove:
                    # nodes at or above the cutoff were pruned, so the valid bound is
                    # the smaller of the open-node bound and the cutoff
                    bound = min(m.ObjBound, -RC_TOL)
                else:
                    bound = m.ObjBound
                if prove:
                    trajectory.append((m.Runtime, bound))
                self.stats[kind] += 1
                self.stats[f"time_{kind}"] += time.perf_counter() - te
                if any(rc < -RC_TOL for rc in best.values()):
                    self.stats[f"{kind}_hits"] += 1
        finally:
            for var, ub in pruned_saved:
                var.UB = ub
            for c in branch:
                m.remove(c)
            m.update()

        neg = sorted(((rc, c) for c, rc in best.items() if rc < -RC_TOL), key=lambda t: t[0])
        elapsed = time.perf_counter() - t0
        self.stats["calls"] += 1
        self.stats["time"] += elapsed
        self.stats["times"].append(elapsed)
        return PricingResult(
            columns=[c for _, c in neg],
            reduced_costs=[rc for rc, _ in neg],
            best_rc=min(best.values()) if best else None,
            bound=bound,
            status=status,
            time=elapsed,
            pool=list(best),
            kind=kind,
            pruned=n_pruned,
            bound_trajectory=trajectory,
        )
