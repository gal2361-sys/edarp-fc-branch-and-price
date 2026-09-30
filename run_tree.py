"""
Branch-and-price for one instance; writes results/tree_<instance>[_tag].json.

    python run_tree.py b25_1 --scorer most_fractional --tag tree_v1

Same pricing flags as run_root.py, plus --scorer, --node-time-limit and --seed.
lb is a proven lower bound; ub is the best exact integer solution found.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import gurobipy as gp

from bp.grid import grid_step
from bp.initial import initial_solution
from bp.master import RestrictedMaster
from bp.pricing import PricingMILP
from bp.pricing_lns import PricingLNS
from bp.scorers import make_scorer
from bp.tree import BranchAndPrice
from edarp import load_instance
from run_compact import PUBLISHED, RESULTS_DIR, resolve_paths


def finite(x):
    """JSON-safe: infinities become None, recursively."""
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--node-time-limit", type=float, default=600.0)
    ap.add_argument("--scorer", choices=["most_fractional", "random"], default="most_fractional")
    ap.add_argument("--seed", type=int, default=0, help="scorer and LNS random seed")
    ap.add_argument("--pricing-time-limit", type=float, default=30.0)
    ap.add_argument("--pool-size", type=int, default=50)
    ap.add_argument("--artificial-cost", type=float, default=None,
                    help="default: sum over requests of their cheapest singleton cost")
    ap.add_argument("--ip-time-limit", type=float, default=10.0,
                    help="heuristic integer master after every node")
    ap.add_argument("--restricted-k", type=int, default=20)
    ap.add_argument("--early-stop", type=int, default=5)
    ap.add_argument("--lns-time", type=float, default=2.0)
    ap.add_argument("--prove-time-limit", type=float, default=1200.0)
    ap.add_argument("--no-dual-pruning", action="store_true")
    ap.add_argument("--no-valid-inequalities", action="store_true")
    ap.add_argument("--initial-solution", action=argparse.BooleanOptionalAction, default=True,
                    help="constructive full-fleet heuristic before column generation")
    ap.add_argument("--initial-time", type=float, default=5.0,
                    help="LNS seconds per vehicle in the initial heuristic")
    ap.add_argument("--initial-repair-time", type=float, default=60.0)
    ap.add_argument("--grid-round", action=argparse.BooleanOptionalAction, default=False,
                    help="round node bounds up to the objective lattice (docs/GRID_ROUNDING.md)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.perf_counter()
    req, circ = resolve_paths(args.instance, None)
    inst = load_instance(req, circ, name=args.instance)
    print(inst.summary())

    env = gp.Env(params={"OutputFlag": 0})
    pricing = PricingMILP(inst, time_limit=args.pricing_time_limit, pool_size=args.pool_size,
                          env=env, valid_inequalities=not args.no_valid_inequalities,
                          restricted_k=args.restricted_k or None,
                          early_stop=args.early_stop or None,
                          dual_pruning=not args.no_dual_pruning)
    lns = PricingLNS(inst, time_limit=args.lns_time, pool_size=args.pool_size,
                     seed=args.seed) if args.lns_time > 0 else None
    master = RestrictedMaster(inst, env=env)
    n_init = master.add_initial_columns(pricing)
    art_cost = args.artificial_cost if args.artificial_cost is not None \
        else master.singleton_cost_sum()
    master.set_artificial_cost(art_cost)
    init_calls = pricing.stats["calls"]

    init = None
    if args.initial_solution:
        init = initial_solution(inst, time_per_vehicle=args.initial_time,
                                repair_time=args.initial_repair_time, seed=args.seed)
        if init.covered:
            for c in init.columns:
                master.add_column(c)
        print(f"initial solution: covered={init.covered} cost {init.cost:.4f} "
              f"uncovered {len(init.uncovered)} ({init.time:.1f}s)")
    grid = grid_step(inst) if args.grid_round else None

    bp_ = BranchAndPrice(inst, master, pricing, lns, make_scorer(args.scorer, args.seed),
                         time_limit=args.time_limit, node_time_limit=args.node_time_limit,
                         prove_time_limit=args.prove_time_limit,
                         ip_time_limit=args.ip_time_limit, verbose=not args.quiet, t0=t0,
                         grid=grid)
    if init is not None and init.covered:
        bp_.offer(init.columns, "initial heuristic")
        bp_.log_initial_incumbent()   # first bound_log entry: [t, None, initial UB]
    tr = bp_.solve()

    pub = PUBLISHED.get(args.instance)
    gap = (tr.ub - tr.lb) / tr.ub if math.isfinite(tr.ub) and tr.ub > 0 else None
    res = {
        "instance": args.instance,
        "method": "branch_and_price",
        "scorer": args.scorer,
        "seed": args.seed,
        "tag": args.tag,
        "status": tr.status,
        "ub": tr.ub,
        "lb": tr.lb,
        "gap": gap,
        "root_lb": tr.root_lb,
        "root_lb_kind": tr.root_lb_kind,
        "lb_kind": tr.lb_kind,
        "nodes": tr.nodes_processed,
        "nodes_processed": tr.nodes_processed,
        "nodes_pruned": tr.nodes_pruned,
        "nodes_open_at_end": tr.nodes_open_at_end,
        "max_depth": tr.max_depth,
        "columns_generated": tr.columns_generated,
        "columns_initial": n_init,
        "pricing_calls": tr.pricing_calls + init_calls,
        "time_total": time.perf_counter() - t0,
        "time_pricing": tr.time_pricing,
        "time_scorer": tr.time_scorer,
        "bound_log": tr.bound_log,
        "published": pub,
        "matches_published": (abs(tr.ub - pub) < 1e-3) if pub is not None and
                             math.isfinite(tr.ub) else None,
        "incumbent": tr.incumbent,
        "incumbent_source": tr.incumbent_source,
        "initial_solution": None if init is None else {
            "covered": init.covered, "cost": init.cost if init.covered else None,
            "n_columns": len(init.columns), "uncovered": init.uncovered,
            "time": init.time, "stages": init.stages},
        "grid_step": grid,
        "node_log": tr.node_log,
        "config": {k: v for k, v in vars(args).items() if k not in ("instance", "tag")},
        "artificial_cost": art_cost,
        "artificial_cost_final": master.artificial_cost,
        "circuit": os.path.basename(circ),
    }
    print(f"\nstatus {tr.status}  ub {tr.ub:.4f}  lb {tr.lb:.4f}  root_lb {tr.root_lb}  "
          f"(lb {tr.lb_kind}, root {tr.root_lb_kind})  nodes {tr.nodes_processed} (pruned {tr.nodes_pruned}, open {tr.nodes_open_at_end}, "
          f"depth {tr.max_depth})  time {res['time_total']:.1f}s  published {pub}  "
          f"match {res['matches_published']}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"tree_{args.instance}{'_' + args.tag if args.tag else ''}.json")
    with open(out, "w") as f:
        json.dump(finite(res), f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
