"""
Run root or compact solves sequentially and write results/summary_<tag>.csv.

    python run_batch.py root b25 b50 --tag cheap -- --time-limit 1800
    python run_batch.py compact b25 --tag vi -- --valid-inequalities --time-limit 1800
    python run_batch.py tree b25 --tag tree_v1

Instances: explicit names (b25_3) or a size shorthand (b25 = b25_0..b25_9).
Arguments after "--" go to run_root.py / run_compact.py unchanged. For root runs
the batch tag is passed as --tag. A CSV row is appended after every run, so a
partial batch still leaves a usable summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time

from run_compact import PUBLISHED, RESULTS_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

ROOT_FIELDS = ["instance", "root_lp_bound", "converged", "lagrangian_lb", "heuristic_ub",
               "published", "time_total", "time_pricing", "n_iterations", "n_columns",
               "prove_timed_out", "artificial_active", "exit_code"]
TREE_FIELDS = ["instance", "root_lb", "root_lb_kind", "lb", "lb_kind", "ub", "ub_source",
               "initial_ub", "gap", "status", "nodes", "nodes_pruned",
               "max_depth", "time_total", "time_pricing", "published", "matches_published",
               "exit_code"]
COMPACT_FIELDS = ["instance", "status", "lb", "ub", "gap", "nodes", "time_total",
                  "root_lp", "published", "exit_code"]


def expand(specs: list[str]) -> list[str]:
    out = []
    for s in specs:
        out += [f"{s}_{z}" for z in range(10)] if "_" not in s else [s]
    return out


def main() -> None:
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        k = argv.index("--")
        argv, extra = argv[:k], argv[k + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["root", "compact", "tree"])
    ap.add_argument("instances", nargs="+")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--summary", default=None,
                    help="summary file stem (default: the tag); JSON names always use the tag")
    args = ap.parse_args(argv)

    insts = expand(args.instances)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    stem = args.summary or args.tag
    out = os.path.join(RESULTS_DIR, f"summary_{stem}.csv")
    fields = {"root": ROOT_FIELDS, "tree": TREE_FIELDS}.get(args.mode, COMPACT_FIELDS)
    with open(out, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    # the exact arguments, so a later queue can tell whether a summary is reusable
    with open(os.path.join(RESULTS_DIR, f"summary_{stem}.args"), "w") as f:
        f.write(" ".join([args.mode] + args.instances + ["--"] + extra) + "\n")

    vi = "--valid-inequalities" in extra
    for inst in insts:
        t0 = time.time()
        if args.mode == "root":
            cmd = [PY, "run_root.py", inst, "--quiet", "--tag", args.tag] + extra
            jpath = os.path.join(RESULTS_DIR, f"root_{inst}_{args.tag}.json")
        elif args.mode == "tree":
            cmd = [PY, "run_tree.py", inst, "--quiet", "--tag", args.tag] + extra
            jpath = os.path.join(RESULTS_DIR, f"tree_{inst}_{args.tag}.json")
        else:
            z = inst.split("_")[1]
            cmd = [PY, "run_compact.py", inst, "--quiet"] + extra
            jpath = os.path.join(RESULTS_DIR, f"compact_{inst}_c{z}{'_vi' if vi else ''}.json")
        log = os.path.join(RESULTS_DIR, f"batch_{args.tag}_{inst}.log")
        print(f"[{time.strftime('%H:%M:%S')}] {inst}: {' '.join(cmd[1:])}", flush=True)
        with open(log, "w") as lf:
            code = subprocess.run(cmd, cwd=HERE, stdout=lf, stderr=subprocess.STDOUT).returncode
        row = {"instance": inst, "exit_code": code, "published": PUBLISHED.get(inst)}
        if code == 0 and os.path.exists(jpath) and os.path.getmtime(jpath) >= t0:
            with open(jpath) as jf:
                r = json.load(jf)
            if args.mode in ("root", "tree"):
                for k in fields:
                    if k in r and k not in ("instance", "published"):
                        row[k] = r[k]
                if args.mode == "tree":
                    row["ub_source"] = r.get("incumbent_source")
                    init = r.get("initial_solution") or {}
                    row["initial_ub"] = init.get("cost")
            else:
                bl = r.get("bound_log") or []
                row.update(status=r["status"], lb=r["lb"], ub=r["ub"], gap=r["gap"],
                           nodes=r["nodes"], time_total=r["time_total"],
                           root_lp=bl[1][1] if len(bl) > 1 else None)
        else:
            print(f"   run failed or wrote no fresh JSON (exit {code}); see {log}", flush=True)
        with open(out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)
        print(f"   done in {time.time() - t0:.0f}s", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
