"""Branch-and-price over Ryan-Foster pairs.

Nodes carry the branching decisions on their path: `together` pairs (a column must
serve both or neither) and `apart` pairs (no column serves both). At a node:
  a. master columns violating a decision get lambda ub = 0 (restored afterwards;
     other nodes may need them);
  b. column generation with the decisions forwarded to LNS and MILP pricing;
  c. the node LB is the LP value only if pricing proved convergence with no
     artificial column active; otherwise the Lagrangian bound,
     otherwise the parent's bound. A child's feasible set is a subset of its
     parent's, so max(bound, parent_lb) is valid and keeps the bound log monotone;
  d. prune if LB >= incumbent - PRUNE_TOL;
  e. an integral LP with proven pricing is a feasible solution: new incumbent,
     node closed;
  f. otherwise branch on a pair chosen by the Scorer among fractional candidates;
  g. a heuristic integer master over the node's allowed pool may improve the
     incumbent (columns are exact: every cost comes from evaluate()).
Search is best-first on parent_lb. Global LB = min(parent_lb over open nodes, the
node in process, incumbent).
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field

from bp.colgen import generate_columns
from bp.grid import round_up
from bp.column import Column
from bp.master import RestrictedMaster
from bp.pricing import PricingMILP
from bp.pricing_lns import PricingLNS
from bp.scorers import Candidate, Scorer
from edarp import Instance

INT_TOL = 1e-6      # lambda within this of 0/1 counts as integral
FRAC_TOL = 1e-6     # candidates: FRAC_TOL < f_ij < 1 - FRAC_TOL
PRUNE_TOL = 1e-6


@dataclass
class Node:
    together: frozenset
    apart: frozenset
    parent_lb: float
    depth: int
    id: int
    branch: tuple | None = None       # (i, j, "together" | "apart") that created it
    parent_id: int | None = None


def violates(col: Column, together, apart) -> bool:
    s = col.served
    return any((i in s) != (j in s) for i, j in together) or any(
        i in s and j in s for i, j in apart)


def pair_flows(columns: list[Column], lambdas: list[float], tol: float = 1e-9) -> dict:
    """f_ij = sum of lambda over columns serving both i and j (i < j)."""
    f: dict = {}
    for col, lam in zip(columns, lambdas):
        if lam <= tol:
            continue
        s = sorted(col.served)
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                k = (s[a], s[b])
                f[k] = f.get(k, 0.0) + lam
    return f


@dataclass
class TreeResult:
    status: str                  # "optimal" or "time_limit"
    ub: float
    lb: float
    root_lb: float | None
    root_lb_kind: str | None     # "lp" (converged), "lagrangian", "parent" or "grid"
    lb_kind: str                 # kind of bound that defines the final lb
    incumbent: list              # served sets of the incumbent's columns
    incumbent_source: str | None  # "initial heuristic", "integer master", "integral LP", ...
    nodes_processed: int
    nodes_pruned: int
    nodes_open_at_end: int
    max_depth: int
    bound_log: list
    node_log: list
    time_total: float
    time_pricing: float
    time_scorer: float
    pricing_calls: int
    columns_generated: int


class BranchAndPrice:
    def __init__(self, inst: Instance, master: RestrictedMaster, pricing: PricingMILP,
                 lns: PricingLNS | None, scorer: Scorer, time_limit: float = 1800.0,
                 node_time_limit: float = 600.0, prove_time_limit: float = 1200.0,
                 ip_time_limit: float = 10.0, verbose: bool = False,
                 t0: float | None = None, grid: float | None = None) -> None:
        self.inst, self.master, self.pricing, self.lns = inst, master, pricing, lns
        self.scorer = scorer
        self.time_limit = time_limit
        self.node_time_limit = node_time_limit
        self.prove_time_limit = prove_time_limit
        self.ip_time_limit = ip_time_limit
        self.verbose = verbose
        # objective lattice step (bp.grid): node bounds are rounded up to it for the
        # global bound and pruning only; None disables (docs/GRID_ROUNDING.md)
        self.grid = grid
        self.incumbent_source = None
        self.t0 = time.perf_counter() if t0 is None else t0
        self.incumbent = math.inf
        self.incumbent_cols: list[Column] = []
        self.open: list = []                 # heap of (parent_lb, id, Node)
        self.current: Node | None = None
        self.bound_log: list = []
        self.node_log: list = []
        self.next_id = 0
        self.nodes_processed = self.nodes_pruned = 0
        self.max_depth = 0
        self.time_pricing = self.time_scorer = 0.0
        self.pricing_calls = 0
        self.root_lb = None
        self.root_lb_kind = None
        self.columns_at_start = len(master.columns)

    # ------------------------------------------------------------------ bounds

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def global_lb(self, current_lb: float | None = None) -> float:
        lbs = [lb for lb, _, _ in self.open]
        if current_lb is not None:
            lbs.append(current_lb)
        return min(lbs + [self.incumbent])

    def log_bounds(self, current_lb: float | None = None) -> None:
        lb = self.global_lb(current_lb)
        if not math.isfinite(lb) and not math.isfinite(self.incumbent):
            return
        entry = [self.elapsed(), lb, self.incumbent]
        if not self.bound_log or self.bound_log[-1][1:] != entry[1:]:
            self.bound_log.append(entry)

    def replay_bounds(self, node: Node, cg_log: list, t_start: float) -> None:
        """Rule 4: the bound improves inside a node too (Lagrangian bound per pricing
        call), not only when it finishes. Replay the node's column-generation log into
        the bound log: at each improvement the global LB is min(open nodes, this node's
        running bound, incumbent). Children are pushed only after this, so the open
        set is the one in force during the node."""
        running = node.parent_lb
        for row in cg_log:
            lb = row.get("lagrangian_lb")
            if lb is None or not math.isfinite(lb) or lb <= running:
                continue
            running = lb
            glb = min([b for b, _, _ in self.open] + [running, self.incumbent])
            entry = [t_start + row["elapsed"], glb, self.incumbent]
            if not self.bound_log or self.bound_log[-1][1:] != entry[1:]:
                self.bound_log.append(entry)

    def log_initial_incumbent(self) -> None:
        """Bound-log entry for an incumbent found before the search: nothing has been
        bounded yet, so the lower bound is -inf (global_lb() would wrongly return the
        incumbent itself while the open list is still empty)."""
        if math.isfinite(self.incumbent):
            self.bound_log.append([self.elapsed(), -math.inf, self.incumbent])

    def offer(self, cols: list[Column], source: str) -> bool:
        cost = sum(c.cost for c in cols)
        if cost < self.incumbent - 1e-9:
            self.incumbent, self.incumbent_cols = cost, list(cols)
            self.incumbent_source = source.split(" at node")[0]
            if self.verbose:
                print(f"   new incumbent {cost:.4f} ({source})", flush=True)
            return True
        return False

    # ------------------------------------------------------------------ nodes

    def new_node(self, together, apart, parent_lb, depth, branch=None, parent_id=None) -> Node:
        node = Node(frozenset(together), frozenset(apart), parent_lb, depth, self.next_id,
                    branch, parent_id)
        self.next_id += 1
        return node

    def push(self, node: Node) -> None:
        heapq.heappush(self.open, (node.parent_lb, node.id, node))
        self.max_depth = max(self.max_depth, node.depth)

    def allow(self, node: Node) -> list:
        """ub = 0 for pool columns violating the node's decisions; returns
        (var, previous ub) pairs for restore()."""
        blocked = []
        for col, var in zip(self.master.columns, self.master.vars):
            if violates(col, node.together, node.apart):
                blocked.append((var, var.UB))
                var.UB = 0.0
        self.master.model.update()
        return blocked

    def restore(self, blocked: list) -> None:
        for var, ub in blocked:
            var.UB = ub
        self.master.model.update()

    def process(self, node: Node) -> list[Node]:
        """Process one node; returns its children (empty if closed or pruned)."""
        t_node = time.perf_counter()
        t_node_elapsed = self.elapsed()
        self.current = node
        n_cols0 = len(self.master.columns)
        budget = min(self.time_limit - self.elapsed(), self.node_time_limit)
        blocked = self.allow(node)
        together, apart = tuple(sorted(node.together)), tuple(sorted(node.apart))
        rec = {"id": node.id, "parent": node.parent_id, "depth": node.depth,
               "branch": list(node.branch) if node.branch else None,
               "together": sorted(node.together), "apart": sorted(node.apart),
               "parent_lb": node.parent_lb, "blocked_columns": len(blocked)}
        children: list[Node] = []
        try:
            cg = generate_columns(self.inst, self.master, self.pricing, time_limit=budget,
                                  together=together, apart=apart, lns=self.lns,
                                  prove_time_limit=self.prove_time_limit)
            self.time_pricing += cg.time_pricing
            self.pricing_calls += cg.pricing_calls
            self.replay_bounds(node, cg.log, t_node_elapsed)
            sol = self.master.solve()      # lambdas aligned with master.columns
            if cg.converged:
                lb_raw, kind = cg.lp_value, "lp"
            elif math.isfinite(cg.lagrangian_lb):
                lb_raw, kind = cg.lagrangian_lb, "lagrangian"
            else:
                lb_raw, kind = node.parent_lb, "parent"
            lb = max(lb_raw, node.parent_lb)
            if self.grid is not None:
                lb_grid = round_up(lb, self.grid)
                if lb_grid > lb + 1e-9:
                    rec["lb_before_grid"] = lb
                    lb, kind = lb_grid, "grid"
            if node.id == 0:
                self.root_lb, self.root_lb_kind = lb, kind
            rec.update(lb=lb, lb_raw=lb_raw, lb_kind=kind, lp_value=sol.value,
                       pricing_proven=cg.pricing_proven,
                       artificial_active=sol.artificial_active,
                       colgen_iterations=cg.iterations, prove_timed_out=cg.prove_timed_out,
                       lagrangian_lb=cg.lagrangian_lb,
                       lagrangian_lb_iteration=cg.lagrangian_lb_iteration)

            # g. heuristic integer master over the node's allowed pool
            ip = self.master.solve_integer(time_limit=self.ip_time_limit)
            if ip.ub is not None:
                self.offer(ip.columns, f"integer master at node {node.id}")

            if lb >= self.incumbent - PRUNE_TOL:
                rec["outcome"] = "pruned_by_bound"
                return children

            lam = sol.lambdas
            real = [(c, x) for c, x in zip(self.master.columns, lam) if x > INT_TOL]
            integral = all(abs(x - 1.0) <= INT_TOL for _, x in real)
            proven = cg.pricing_proven and not sol.artificial_active
            if integral and not sol.artificial_active:
                self.offer([c for c, _ in real], f"integral LP at node {node.id}")
                if proven:
                    rec["outcome"] = "integral"
                    return children

            flows = pair_flows(self.master.columns, lam)
            cands = [Candidate(i, j, f) for (i, j), f in flows.items()
                     if FRAC_TOL < f < 1 - FRAC_TOL]
            rec["n_candidates"] = len(cands)
            if not cands and proven and not integral:
                # fractional lambdas but no fractional pair: every request's positive
                # columns serve one identical set; the cheapest column per set is an
                # integral solution of cost <= LP value, hence equal to it
                best: dict = {}
                for c, _ in real:
                    if c.served not in best or c.cost < best[c.served].cost:
                        best[c.served] = c
                self.offer(list(best.values()), f"same-set rounding at node {node.id}")
                if sum(c.cost for c in best.values()) <= lb + PRUNE_TOL:
                    rec["outcome"] = "integral_by_rounding"
                    return children
            if lb >= self.incumbent - PRUNE_TOL:
                rec["outcome"] = "pruned_by_bound"
                return children

            ts = time.perf_counter()
            if cands:
                pick = self.scorer.choose(cands, {"node": node, "master": self.master,
                                                  "solution": sol, "inst": self.inst,
                                                  "flows": flows})
                rec["branch_on"] = [pick.i, pick.j, pick.f]
            else:
                # not converged and nothing fractional: any pair is a valid branching
                multi = sorted((c for c, _ in real if len(c.served) >= 2),
                               key=lambda c: sorted(c.served))
                pair = tuple(sorted(multi[0].served)[:2]) if multi else (0, 1)
                pick = Candidate(pair[0], pair[1], flows.get(pair, 0.0))
                rec["branch_on"] = [pick.i, pick.j, pick.f, "fallback"]
            self.time_scorer += time.perf_counter() - ts
            pair = (pick.i, pick.j)
            children = [
                self.new_node(node.together | {pair}, node.apart, lb, node.depth + 1,
                              (pick.i, pick.j, "together"), node.id),
                self.new_node(node.together, node.apart | {pair}, lb, node.depth + 1,
                              (pick.i, pick.j, "apart"), node.id),
            ]
            rec["outcome"] = "branched"
            rec["children"] = [c.id for c in children]
            return children
        finally:
            self.restore(blocked)
            rec["columns_added"] = len(self.master.columns) - n_cols0
            rec["time"] = time.perf_counter() - t_node
            self.node_log.append(rec)
            self.nodes_processed += 1
            self.current = None
            if self.verbose:
                print(f"node {node.id:4d} d{node.depth:2d} {str(node.branch):28s} "
                      f"lb {rec.get('lb', float('nan')):9.4f} ({rec.get('lb_kind', '-')}) "
                      f"inc {self.incumbent:9.4f} +{rec['columns_added']:4d} cols "
                      f"{rec['time']:6.1f}s -> {rec.get('outcome', '?')}", flush=True)

    # ------------------------------------------------------------------ search

    def solve(self, start: list[Node] | None = None) -> TreeResult:
        """Best-first search from the root (default) or from given start nodes."""
        for node in start if start is not None else [self.new_node((), (), -math.inf, 0)]:
            self.push(node)
        while self.open and self.elapsed() < self.time_limit:
            parent_lb, _, node = heapq.heappop(self.open)
            if parent_lb >= self.incumbent - PRUNE_TOL:
                self.nodes_pruned += 1
                self.node_log.append({"id": node.id, "parent": node.parent_id,
                                      "depth": node.depth, "parent_lb": parent_lb,
                                      "outcome": "pruned_before_processing"})
                self.log_bounds()
                continue
            children = self.process(node)
            if self.node_log[-1].get("outcome") == "pruned_by_bound":
                self.nodes_pruned += 1
            for ch in children:
                self.push(ch)
            self.log_bounds()
        status = "optimal" if not self.open else "time_limit"
        lb = self.global_lb()
        if status == "optimal":
            lb, lb_kind = self.incumbent, "optimal"
        elif lb >= self.incumbent:
            lb_kind = "incumbent"
        else:
            # the open node with the smallest parent_lb inherited its parent's bound
            _, _, weakest = min(self.open)
            kinds = {r["id"]: r.get("lb_kind") for r in self.node_log}
            lb_kind = kinds.get(weakest.parent_id) or "parent"
        self.log_bounds()
        return TreeResult(
            status=status, ub=self.incumbent, lb=lb, root_lb=self.root_lb,
            root_lb_kind=self.root_lb_kind, lb_kind=lb_kind,
            incumbent=[sorted(c.served) for c in self.incumbent_cols],
            incumbent_source=self.incumbent_source,
            nodes_processed=self.nodes_processed, nodes_pruned=self.nodes_pruned,
            nodes_open_at_end=len(self.open), max_depth=self.max_depth,
            bound_log=self.bound_log, node_log=self.node_log,
            time_total=self.elapsed(), time_pricing=self.time_pricing,
            time_scorer=self.time_scorer, pricing_calls=self.pricing_calls,
            columns_generated=len(self.master.columns) - self.columns_at_start,
        )
