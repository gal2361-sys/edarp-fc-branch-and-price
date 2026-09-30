"""Restricted master LP over single-vehicle columns.

    min  sum_p c_p lambda_p + ART_COST * sum_i a_i
    s.t. sum_{p covers i} lambda_p + a_i == 1     for every request i   (cover_i)
         sum_p lambda_p                  <= V                          (fleet)
         lambda, a >= 0

a_i is an artificial column serving only request i. It keeps the LP feasible
from the first iteration. It is not a vehicle, so it is not in the fleet row.
A master result with any a_i > ART_TOL is not a valid relaxation value of the
real problem; callers must check `artificial_active`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from bp.column import Column
from edarp import Instance

ART_COST = 1e4
ART_TOL = 1e-9
ART_COST_CAP = 1e6


@dataclass
class MasterSolution:
    value: float
    pi: list[float]                 # duals of the covering rows
    mu: float                       # dual of the fleet row, <= 0
    lambdas: list[float]            # parallel to master.columns
    artificial: list[float]         # value of a_i per request
    time: float

    @property
    def artificial_active(self) -> bool:
        return any(a > ART_TOL for a in self.artificial)


@dataclass
class IntegerSolution:
    ub: float | None                # None if no solution or an artificial is used
    columns: list[Column]
    status: str
    artificial_active: bool
    time: float


class RestrictedMaster:
    def __init__(
        self, inst: Instance, env: gp.Env | None = None, artificial_cost: float = ART_COST
    ) -> None:
        self.inst = inst
        self.artificial_cost = artificial_cost
        self.env = env or gp.Env(params={"OutputFlag": 0})
        m = gp.Model(f"rmp_{inst.name}", env=self.env)
        m.ModelSense = GRB.MINIMIZE
        self.art = m.addVars(inst.n, lb=0.0, obj=artificial_cost, name="art")
        self.cover = [
            m.addConstr(self.art[i] == 1, name=f"cover_{i}") for i in range(inst.n)
        ]
        self.fleet = m.addConstr(gp.LinExpr() <= inst.V, name="fleet")
        m.Params.Method = 0  # primal simplex: new columns keep the basis primal feasible
        m.update()
        self.model = m
        self.columns: list[Column] = []
        self.vars: list[gp.Var] = []
        self._index: dict[Column, int] = {}

    def set_artificial_cost(self, cost: float) -> None:
        self.artificial_cost = cost
        for i in range(self.inst.n):
            self.art[i].Obj = cost
        self.model.update()

    def add_column(self, col: Column) -> bool:
        """Add a column; returns False if an equal column is already present."""
        if col in self._index:
            return False
        rows = [self.cover[i] for i in sorted(col.served)] + [self.fleet]
        v = self.model.addVar(
            lb=0.0, obj=col.cost, column=gp.Column([1.0] * len(rows), rows),
            name=f"lam_{len(self.columns)}",
        )
        self._index[col] = len(self.columns)
        self.columns.append(col)
        self.vars.append(v)
        return True

    def solve(self) -> MasterSolution:
        t0 = time.perf_counter()
        m = self.model
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"master LP status {m.Status}")
        return MasterSolution(
            value=m.ObjVal,
            pi=[c.Pi for c in self.cover],
            mu=self.fleet.Pi,
            lambdas=[v.X for v in self.vars],
            artificial=[self.art[i].X for i in range(self.inst.n)],
            time=time.perf_counter() - t0,
        )

    def solve_integer(self, time_limit: float = 60.0) -> IntegerSolution:
        """Primal heuristic: the same master with binary lambda over current columns."""
        t0 = time.perf_counter()
        self.model.update()  # flush columns added since the last solve()
        ip = self.model.copy()
        for v in ip.getVars():
            v.VType = GRB.BINARY
        ip.Params.TimeLimit = time_limit
        ip.Params.Method = -1
        ip.optimize()
        status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time_limit",
                  GRB.INFEASIBLE: "infeasible"}.get(ip.Status, str(ip.Status))
        chosen: list[Column] = []
        ub, art_active = None, False
        if ip.SolCount > 0:
            vals = ip.getAttr("X", ip.getVars())
            by_name = dict(zip((v.VarName for v in ip.getVars()), vals))
            art_active = any(by_name[f"art[{i}]"] > 0.5 for i in range(self.inst.n))
            chosen = [c for c, v in zip(self.columns, self.vars) if by_name[v.VarName] > 0.5]
            if not art_active:
                ub = sum(c.cost for c in chosen)
        ip.dispose()
        return IntegerSolution(ub, chosen, status, art_active, time.perf_counter() - t0)

    def singleton_cost_sum(self) -> float:
        """Sum over requests of the cost of their cheapest singleton column.

        Uses the master's columns serving exactly {i}; a request without one gets
        its cheapest one-request plan from evaluate() (via PricingLNS.singleton).
        No real LP solution can justify a dual above this, so it is the default
        artificial cost.
        """
        best: dict[int, float] = {}
        for c in self.columns:
            if len(c.served) == 1:
                (i,) = c.served
                best[i] = min(best.get(i, float("inf")), c.cost)
        missing = [i for i in range(self.inst.n) if i not in best]
        if missing:
            from bp.pricing_lns import PricingLNS
            lns = PricingLNS(self.inst)
            for i in missing:
                col = lns.singleton(i)
                if col is None:
                    raise RuntimeError(f"request {i} has no feasible singleton plan")
                best[i] = col.cost
        return sum(best.values())

    def add_initial_columns(self, pricing) -> int:
        """Real singleton-driven columns: pricing with pi_i = 1e4 for one request."""
        added = 0
        for i in range(self.inst.n):
            pi = [0.0] * self.inst.n
            pi[i] = 1e4
            res = pricing.solve(pi, 0.0)
            added += sum(self.add_column(c) for c in res.columns)
        return added
