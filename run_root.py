"""
Root column generation for one instance; writes results/root_<instance>.json.

    python run_root.py b25_0 --time-limit 1800 --pricing-time-limit 30

root_lp_bound is a valid lower bound only if converged is true.
lagrangian_lb is valid in every case.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import gurobipy as gp

from bp.colgen import generate_columns
from bp.master import RestrictedMaster
from bp.pricing import PricingMILP
from bp.pricing_lns import PricingLNS
from edarp import load_instance
from run_compact import PUBLISHED, RESULTS_DIR, resolve_paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--pricing-time-limit", type=float, default=30.0)
    ap.add_argument("--pool-size", type=int, default=50)
    ap.add_argument("--artificial-cost", type=float, default=None,
                    help="default: sum over requests of their cheapest singleton cost")
    ap.add_argument("--stabilize", action="store_true", help="Wentges dual smoothing")
    ap.add_argument("--alpha", type=float, default=0.5, help="smoothing weight on the centre")
    ap.add_argument("--ip-time-limit", type=float, default=60.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the results file name")
    ap.add_argument("--restricted-k", type=int, default=20,
                    help="restricted pricing over the k largest duals; 0 disables")
    ap.add_argument("--early-stop", type=int, default=5,
                    help="stop exact pricing after this many negative solutions; 0 disables")
    ap.add_argument("--lns-time", type=float, default=2.0,
                    help="LNS pricing heuristic seconds per call; 0 disables")
    ap.add_argument("--prove-time-limit", type=float, default=None,
                    help="single prove call limit (default: min(remaining, 1200))")
    ap.add_argument("--no-dual-pruning", action="store_true",
                    help="do not exclude requests with pi_i <= 0 from pricing")
    ap.add_argument("--seed", type=int, default=0, help="LNS random seed")
    ap.add_argument("--no-valid-inequalities", action="store_true",
                    help="pricing model without the Electronic Companion B families")
    args = ap.parse_args()

    t0 = time.perf_counter()
    req, circ = resolve_paths(args.instance, None)
    inst = load_instance(req, circ, name=args.instance)
    print(inst.summary())

    env = gp.Env(params={"OutputFlag": 0})
    pricing = PricingMILP(inst, time_limit=args.pricing_time_limit,
                          pool_size=args.pool_size, env=env,
                          valid_inequalities=not args.no_valid_inequalities,
                          restricted_k=args.restricted_k or None,
                          early_stop=args.early_stop or None,
                          dual_pruning=not args.no_dual_pruning)
    lns = PricingLNS(inst, time_limit=args.lns_time, pool_size=args.pool_size,
                     seed=args.seed) if args.lns_time > 0 else None
    master = RestrictedMaster(inst, env=env)
    n_init = master.add_initial_columns(pricing)
    art_cost = args.artificial_cost
    if art_cost is None:
        art_cost = master.singleton_cost_sum()
    master.set_artificial_cost(art_cost)
    t_init = time.perf_counter() - t0
    print(f"initial columns: {n_init} real + {inst.n} artificial at cost {art_cost:g} ({t_init:.1f}s)")

    remaining = args.time_limit - (time.perf_counter() - t0)
    cg = generate_columns(inst, master, pricing, time_limit=remaining,
                          verbose=not args.quiet, stabilize=args.stabilize,
                          alpha=args.alpha, lns=lns,
                          prove_time_limit=args.prove_time_limit if args.prove_time_limit else 1200.0)
    ip = master.solve_integer(time_limit=args.ip_time_limit)

    comp_root = None
    comp_path = os.path.join(
        RESULTS_DIR, f"compact_{args.instance}_c{os.path.basename(circ).split('_')[-1].split('.')[0]}.json")
    if os.path.exists(comp_path):
        with open(comp_path) as f:
            bl = json.load(f).get("bound_log") or []
        # bound_log[0] is the pre-root callback value; [1] is the root LP relaxation
        comp_root = bl[1][1] if len(bl) > 1 else None

    res = {
        "instance": args.instance,
        "method": "bp_root",
        "tag": args.tag,
        "circuit": os.path.basename(circ),
        "time_limit": args.time_limit,
        "pricing_time_limit": args.pricing_time_limit,
        "pool_size": args.pool_size,
        "stabilize": args.stabilize,
        "alpha": args.alpha if args.stabilize else None,
        "artificial_cost": art_cost,
        "artificial_cost_final": master.artificial_cost,
        "artificial_escalations": cg.artificial_escalations,
        "escalation_events": cg.escalation_events,
        "stalled": cg.stalled,
        "mispricings": cg.mispricings,
        "valid_inequalities": not args.no_valid_inequalities,
        "restricted_k": args.restricted_k or None,
        "early_stop": args.early_stop or None,
        "lns_time": args.lns_time,
        "dual_pruning": not args.no_dual_pruning,
        "prove_time_limit": args.prove_time_limit,
        "seed": args.seed,
        "prove_timed_out": cg.prove_timed_out,
        "stages": cg.stages,
        "lns_stats": lns.stats if lns else None,
        "final_pi": cg.final_pi,
        "final_mu": cg.final_mu,
        "positive_columns": cg.positive_columns,
        "prove_trajectories": cg.prove_trajectories,
        "columns_dropped": pricing.stats["dropped"],
        "dropped_reasons": pricing.stats["dropped_reasons"],
        "pricing_stats": {k: v for k, v in pricing.stats.items() if k != "times"},
        "root_lp_bound": cg.lp_value,
        "converged": cg.converged,
        "pricing_proven": cg.pricing_proven,
        "artificial_active": cg.artificial_active,
        "lagrangian_lb": cg.lagrangian_lb,
        "lagrangian_lb_iteration": cg.lagrangian_lb_iteration,
        "heuristic_ub": ip.ub,
        "heuristic_columns": [sorted(c.served) for c in ip.columns],
        "heuristic_status": ip.status,
        "n_columns": len(master.columns),
        "n_initial_columns": n_init,
        "n_iterations": cg.iterations,
        "pricing_calls": pricing.stats["calls"],
        "pricing_calls_colgen": cg.pricing_calls,
        "time_total": time.perf_counter() - t0,
        "time_init": t_init,
        "time_master": cg.time_master,
        "time_pricing": cg.time_pricing,
        "time_integer": ip.time,
        "compact_root_lb": comp_root,
        "published": PUBLISHED.get(args.instance),
        "log": cg.log,
    }
    if not cg.converged:
        why = ("artificial column active" if cg.artificial_active
               else "prove call timed out" if cg.prove_timed_out
               else "pricing stalled" if cg.stalled
               else "pricing did not prove convergence")
        print(f"WARNING: root LP value is NOT a valid bound ({why})")
    print(
        f"\nroot_lp {cg.lp_value:.4f} converged={cg.converged} "
        f"lagrangian_lb {cg.lagrangian_lb:.4f} heuristic_ub {ip.ub} "
        f"compact_root {comp_root} published {res['published']}\n"
        f"iterations {cg.iterations} pricing calls {cg.pricing_calls} columns {len(master.columns)} "
        f"time {res['time_total']:.1f}s (master {cg.time_master:.1f}s, "
        f"pricing {cg.time_pricing:.1f}s, init {t_init:.1f}s, ip {ip.time:.1f}s)"
    )
    st = pricing.stats
    print(
        f"pricing calls: restricted {st['restricted']} (hits {st['restricted_hits']}, "
        f"{st['time_restricted']:.1f}s), exact {st['exact']} (early-stopped "
        f"{st['early_stopped']}, {st['time_exact']:.1f}s), prove {st['prove']} "
        f"({st['time_prove']:.1f}s, hits {st['prove_hits']})\n"
        f"stages (calls/hits): " + ", ".join(
            f"{k} {v['calls']}/{v['hits']}" for k, v in cg.stages.items() if v['calls']) + "\n"
        f"dropped columns: {st['dropped']} {st['dropped_reasons']}"
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out = os.path.join(RESULTS_DIR, f"root_{args.instance}{suffix}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
