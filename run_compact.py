"""
Solve one instance with the compact MILP and write a results JSON.

    python run_compact.py b25_0
    python run_compact.py b25_0 --circuit 3 --time-limit 600
    python run_compact.py renmark100_0

Artificial instance b{n}_{z} is paired with circuit_artificial_instances_{z} unless
--circuit overrides it. Renmark instances all use circuit_renmark_instances.txt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

from edarp import load_instance, solve_compact

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "instances")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# published optima, Molenbruch et al. (2023) Table 10, Gurobi 10-hour runs.
# file b25_z is assumed to be the paper's a25_z; this table is what we validate against.
PUBLISHED = {
    "b25_0": 17.45, "b25_1": 24.00, "b25_2": 23.20, "b25_3": 26.35, "b25_4": 21.45,
    "b25_5": 26.50, "b25_6": 23.55, "b25_7": 20.85, "b25_8": 24.75, "b25_9": 24.95,
    "b50_1": 53.20, "b50_2": 52.00, "b50_3": 53.05, "b50_4": 57.10, "b50_5": 53.70,
    "b50_6": 45.95, "b50_7": 56.20, "b50_8": 52.30, "b50_9": 52.40,
    # b50_0: 56.75 was the best found at the 10-hour limit, not proven optimal
}


def resolve_paths(name: str, circuit: int | None) -> tuple[str, str]:
    req = os.path.join(DATA_DIR, f"{name}.txt")
    if name.startswith("renmark"):
        circ = os.path.join(DATA_DIR, "circuit_renmark_instances.txt")
    else:
        m = re.fullmatch(r"b\d+_(\d+)", name)
        if m is None:
            raise ValueError(f"cannot infer circuit for {name}; pass --circuit")
        z = circuit if circuit is not None else int(m.group(1))
        circ = os.path.join(DATA_DIR, f"circuit_artificial_instances_{z}.txt")
    return req, circ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--circuit", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=1800.0)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--valid-inequalities", action="store_true",
                    help="add Electronic Companion B valid inequalities (all families)")
    args = ap.parse_args()

    req, circ = resolve_paths(args.instance, args.circuit)
    inst = load_instance(req, circ, name=args.instance)
    print(inst.summary())
    print(f"circuit: {os.path.basename(circ)}")

    t0 = time.time()
    model, res = solve_compact(inst, time_limit=args.time_limit, log=not args.quiet,
                               valid_inequalities=args.valid_inequalities)
    res["valid_inequalities"] = args.valid_inequalities
    res["circuit"] = os.path.basename(circ)
    res["time_limit"] = args.time_limit
    res["n_vars"] = model.NumVars
    res["n_constrs"] = model.NumConstrs
    res["root_lb"] = res["bound_log"][0][1] if res["bound_log"] else None

    pub = PUBLISHED.get(args.instance)
    if pub is not None and res["ub"] is not None:
        res["published"] = pub
        res["matches_published"] = abs(res["ub"] - pub) < 1e-3
        verdict = "MATCH" if res["matches_published"] else "MISMATCH"
        print(f"\n{verdict}: ours={res['ub']:.4f}  published={pub:.2f}  "
              f"status={res['status']}  time={res['time_total']:.0f}s")
    else:
        print(f"\nstatus={res['status']}  ub={res['ub']}  lb={res['lb']:.4f}  "
              f"time={res['time_total']:.0f}s")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"compact_{args.instance}_c{os.path.basename(circ).split('_')[-1].split('.')[0]}{'_vi' if args.valid_inequalities else ''}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
