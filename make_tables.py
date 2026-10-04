"""
Paper tables and figures from the result files; no numbers are typed in here.

    python make_tables.py

Writes the five tables docs/paper/tables/*.tex and the figure
docs/paper/figures/fig_bounds_b50_all.{pdf,png}, and prints, per output, the files and
the tree tag it used (the newest available: tree_v2, else tree_v1).
"""

from __future__ import annotations

import csv
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
RUNS = os.path.join(RES, "runs")      # committed per-run JSONs; new runs are written to RES
TAB = os.path.join(HERE, "docs", "paper", "tables")
FIG = os.path.join(HERE, "docs", "paper", "figures")
TREE_TAGS = ["tree_v2", "tree_v1"]          # newest first
ABLATIONS = [("abl_full", "Full configuration"), ("abl_no_lns", "No LNS pricing"),
             ("abl_no_restricted", "No restricted pricing"), ("abl_no_pruning", "No dual pruning"),
             ("abl_no_vi", "No valid inequalities"), ("abl_no_earlystop", "No early stop"),
             ("abl_stab", "Dual stabilisation")]
GAP_SHIFT = 1.0      # percentage points, for the shifted geometric mean of gaps
# fig_bounds_b50_all: size in inches, font sizes in pt, line width in pt
FIG_STYLE = {"figsize": (13, 6), "tick": 11, "label": 12, "inst": 11, "legend": 12,
             "lw": 1.4, "marker": 6}
used: dict[str, list[str]] = {}


# ------------------------------------------------------------------ helpers

def rel(path: str) -> str:
    return os.path.relpath(path, HERE)


def read_csv(name: str, table: str) -> list[dict]:
    path = os.path.join(RES, name)
    used.setdefault(table, []).append(rel(path))
    with open(path) as f:
        return list(csv.DictReader(f))


def read_json(path: str, table: str) -> dict:
    used.setdefault(table, []).append(rel(path))
    with open(path) as f:
        return json.load(f)


def num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fmt(x, nd=2, dash="--"):
    v = num(x)
    return dash if v is None else f"{v:.{nd}f}"


def tex(s: str) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%")


def thousands(n) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def geomean(xs):
    xs = [x for x in xs if x is not None]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


def shifted_geomean(xs, s=GAP_SHIFT):
    xs = [x for x in xs if x is not None]
    return math.exp(sum(math.log(x + s) for x in xs) / len(xs)) - s if xs else None


def newest_tree(size: str) -> str:
    for tag in TREE_TAGS:
        if os.path.exists(os.path.join(RES, f"summary_{tag}_{size}.csv")):
            return tag
    raise FileNotFoundError(f"no tree summary for {size}")


def write(name: str, lines: list[str]) -> None:
    with open(os.path.join(TAB, name), "w") as f:
        f.write("\n".join(lines) + "\n")


def by_instance(rows):
    return {r["instance"]: r for r in rows}


# ------------------------------------------------------------------ tables

B25_TREE_TAG = "tree_rerun"   # Table 1: the b25 rerun with the final code (--no-initial-solution)


def tab_b25(tag: str = B25_TREE_TAG, out: str = "tab_b25.tex") -> str:
    key = out[:-4]
    tree = read_csv(f"summary_{tag}_b25.csv", key)
    comp = by_instance(read_csv("summary_compact_vi_b25.csv", key))
    L = [r"\begin{tabular}{lrrrrrrl}", r"\toprule",
         r"Instance & Root LB & UB & Nodes & Time (s) & Compact+VI time (s) & Published & Match \\",
         r"\midrule"]
    for r in tree:
        c = comp[r["instance"]]
        match = "yes" if str(r.get("matches_published")).lower() == "true" else "no"
        L.append(f"{tex(r['instance'])} & {fmt(r['root_lb'], 3)} & {fmt(r['ub'])} & {int(float(r['nodes']))} & "
                 f"{fmt(r['time_total'], 0)} & {fmt(c['time_total'], 0)} & {fmt(r['published'])} & "
                 f"{match} \\\\")
    gt = geomean([num(r["time_total"]) for r in tree])
    gc = geomean([num(comp[r["instance"]]["time_total"]) for r in tree])
    solved = sum(r["status"] == "optimal" for r in tree)
    L += [r"\midrule",
          f"Geometric mean / optimal & & & {solved}/{len(tree)} & {fmt(gt, 0)} & {fmt(gc, 0)} & & \\\\",
          r"\bottomrule", r"\end{tabular}"]
    write(out, L)
    return tag


SOURCE_LABEL = {"initial heuristic": "initial", "integer master": "int. master",
                "integral LP": "integral LP", "same-set rounding": "rounding"}
SOLUTIONS = os.path.join(HERE, "data", "solutions")


def best_known(inst: str, table: str):
    """Objective of the paper's best-known solution file (first line OBJECTIVE_VALUE)."""
    path = os.path.join(SOLUTIONS, f"{inst}_best_solution.txt")
    if not os.path.exists(path):
        return None
    used.setdefault(table, []).append(rel(path))
    with open(path) as f:
        for line in f:
            if line.startswith("OBJECTIVE_VALUE"):
                return num(line.split()[-1])
    return None


def gap_pct(lb, ub):
    """Gap in percent; None when the UB is missing or the LB is not positive."""
    lb, ub = num(lb), num(ub)
    if lb is None or ub is None or lb <= 0:
        return None
    return 100 * (ub - lb) / ub


def tab_size(size: str) -> list[str]:
    """Our tree vs compact+VI. b50: one B&P group with LB kind and a published column
    (best-known values not proven optimal marked *). b75: pricing 30 s and 120 s
    groups, no LB kind (all Lagrangian) and no published column."""
    table = f"tab_{size}"
    main = newest_tree(size)
    if size == "b75":
        groups = [(main, "B\\&P, pricing 30 s")]
        if os.path.exists(os.path.join(RES, f"summary_{main}_p120_{size}.csv")):
            groups.append((f"{main}_p120", "B\\&P, pricing 120 s"))
        show_kind, show_pub = False, False
    else:
        groups = [(main, "B\\&P")]
        show_kind, show_pub = True, True
    data = [(tag, label, by_instance(read_csv(f"summary_{tag}_{size}.csv", table)))
            for tag, label in groups]
    comp = by_instance(read_csv("summary_compact_vi_b50_b75.csv", table))
    insts = list(data[0][2])
    per_group = 5 if show_kind else 4                 # LB, UB, gap, [LB kind], UB source
    cols = "l" + ("rrrll" if show_kind else "rrrl") * len(data) + "rrr" + ("r" if show_pub else "")
    L = [r"{\small\setlength{\tabcolsep}{3pt}", f"\\begin{{tabular}}{{{cols}}}", r"\toprule"]
    L.append(" & " + " & ".join(f"\\multicolumn{{{per_group}}}{{c}}{{{label}}}"
                                for _, label, _ in data)
             + r" & \multicolumn{3}{c}{Compact+VI}" + (" & " if show_pub else "") + r" \\")
    rules, c0 = [], 2
    for _ in data:
        rules.append(f"\\cmidrule(lr){{{c0}-{c0 + per_group - 1}}}")
        c0 += per_group
    rules.append(f"\\cmidrule(lr){{{c0}-{c0 + 2}}}")
    L.append("".join(rules))
    one = r"LB & UB & Gap (\%) & " + ("LB kind & " if show_kind else "") + "UB source"
    L.append("Instance & " + " & ".join(one for _ in data) + r" & LB & UB & Gap (\%)"
             + (" & Published" if show_pub else "") + r" \\")
    L.append(r"\midrule")
    gaps = {tag: {} for tag, _, _ in data}
    gaps["compact"] = {}
    higher = {tag: [0, 0] for tag, _, _ in data}
    starred = False
    for inst in insts:
        c = comp[inst]
        cells = [tex(inst)]
        for tag, _, rows in data:
            r = rows[inst]
            g = gap_pct(r["lb"], r["ub"])
            gaps[tag][inst] = g
            if num(r["lb"]) is not None and num(c["lb"]) is not None:
                higher[tag][1] += 1
                higher[tag][0] += num(r["lb"]) > num(c["lb"]) + 1e-6
            src = SOURCE_LABEL.get(r.get("ub_source") or "", tex(r.get("ub_source") or "--"))
            cells += [fmt(r["lb"]), fmt(r["ub"]), fmt(g, 1)] + \
                     ([tex(r["lb_kind"])] if show_kind else []) + [src]
        gc = gap_pct(c["lb"], c["ub"])
        gaps["compact"][inst] = gc
        cells += [fmt(c["lb"]), fmt(c["ub"]), fmt(gc, 1)]
        if show_pub:
            pub = num(data[0][2][inst]["published"])
            if pub is None:
                bk = best_known(inst, table)
                cells.append("--" if bk is None else f"{bk:.2f}$^*$")
                starred = starred or bk is not None
            else:
                cells.append(fmt(pub))
        L.append(" & ".join(cells) + r" \\")
    # shifted geometric mean over the instances where every side has a gap
    keys = [tag for tag, _, _ in data] + ["compact"]
    common = [i for i in insts if all(gaps[t][i] is not None for t in keys)]
    sgm = {t: shifted_geomean([gaps[t][i] for i in common]) for t in keys}
    L.append(r"\midrule")
    pad = [""] * (per_group - 3)
    cells = [f"Shifted geo.\\ mean gap ({len(common)} of {len(insts)})"]
    for tag, _, _ in data:
        cells += ["", "", fmt(sgm[tag], 1)] + pad
    cells += ["", "", fmt(sgm["compact"], 1)] + ([""] if show_pub else [])
    L.append(" & ".join(cells) + r" \\")
    cells = ["Our LB higher than compact"]
    for tag, _, _ in data:
        cells.append(f"\\multicolumn{{{per_group}}}{{l}}{{{higher[tag][0]} of {higher[tag][1]}}}")
    cells += ["", "", ""] + ([""] if show_pub else [])
    L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    if starred:
        L.append(r"% $^*$ best known value (data/solutions), not proven optimal")
    write(f"{table}.tex", L)
    return [tag for tag, _, _ in data]


def tab_ablation() -> None:
    rows = {}
    for tag, _ in ABLATIONS:
        rows[tag] = read_csv(f"summary_{tag}.csv", "tab_ablation")
    base_t = geomean([num(r["time_total"]) for r in rows["abl_full"]])
    base_i = geomean([num(r["n_iterations"]) for r in rows["abl_full"]])
    L = [r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Configuration & Converged & Time (s) & Ratio & Iterations & Ratio \\", r"\midrule"]
    for tag, label in ABLATIONS:
        rs = rows[tag]
        conv = sum(str(r["converged"]).lower() == "true" for r in rs)
        t = geomean([num(r["time_total"]) for r in rs])
        it = geomean([num(r["n_iterations"]) for r in rs])
        L.append(f"{label} & {conv}/{len(rs)} & {fmt(t, 1)} & {fmt(t / base_t)} & "
                 f"{fmt(it, 1)} & {fmt(it / base_i)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation.tex", L)


def tab_dp() -> None:
    rep = read_json(os.path.join(RES, "dp_prototype_report.json"), "tab_dp")
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Dual vector & DP time (s) & DP labels & MILP time (s) & DP / MILP \\", r"\midrule"]
    for row in rep["b25_0"]:
        dp = row["bound"]
        ok = dp["status"] == "complete"
        dpt = num(dp["time"])
        L.append(f"b25\\_0, iteration {row['iteration']} & {fmt(dpt, 1)}{'' if ok else '*'} & "
                 f"{thousands(dp['labels'])} & {fmt(row['milp_time'], 1)} & "
                 f"{fmt(dpt / num(row['milp_time']), 1)} \\\\")
    L += [r"\midrule", r"\multicolumn{5}{l}{b50\_0 at the final duals: " +
          tex(rep["b50_0"]["dp"]["status"].split(":")[0]) + r", labels per lap} \\"]
    for lap, cnt in sorted(rep["b50_0"]["dp"]["per_lap"].items(), key=lambda kv: int(kv[0])):
        L.append(f"\\quad lap {lap} & & {thousands(cnt)} & & \\\\")
    L.append(f"\\quad total, time & {fmt(rep['b50_0']['dp']['time'], 0)} & "
             f"{thousands(rep['b50_0']['dp']['labels'])} & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    write("tab_dp.tex", L)


# ------------------------------------------------------------------ figures

def run_file(name: str) -> str:
    """Per-run JSON: the committed copy in results/runs/, else a fresh one in results/."""
    committed = os.path.join(RUNS, name)
    return committed if os.path.exists(committed) else os.path.join(RES, name)


def bound_series(log, lb_i=1, ub_i=2):
    t, lb, ub = [], [], []
    for e in log:
        t.append(e[0])
        lb.append(num(e[lb_i]) if num(e[lb_i]) is not None and num(e[lb_i]) > -1e99 else math.nan)
        ub.append(num(e[ub_i]) if num(e[ub_i]) is not None and num(e[ub_i]) < 1e99 else math.nan)
    return t, lb, ub


def plot_bounds(ax, inst: str, tag: str, fig_key: str,
                mark_final: bool = False, best_known_ref: bool = False):
    """LB/UB step curves of our tree and of compact+VI. mark_final: a filled marker at
    the last value of each B&P curve (visible when the step sits at the right edge).
    best_known_ref: without a published optimum, draw the best-known value from
    data/solutions as the dotted reference line."""
    z = inst.split("_")[1]
    ours = read_json(run_file(f"tree_{inst}_{tag}.json"), fig_key)
    comp = read_json(run_file(f"compact_{inst}_c{z}_vi.json"), fig_key)
    t_end = max(ours["time_total"], comp["time_total"])
    for data, name, style in ((ours, "B&P", "-"), (comp, "Compact+VI", "--")):
        t, lb, ub = bound_series(data["bound_log"])
        if not t:
            continue
        # extend each step function to the end of its run
        t = t + [data["time_total"]]
        lb, ub = lb + [lb[-1]], ub + [ub[-1]]
        ax.step(t, lb, where="post", linestyle=style, color="C0" if name == "B&P" else "C1",
                label=f"{name} LB")
        ax.step(t, ub, where="post", linestyle=style, color="C0" if name == "B&P" else "C1",
                alpha=0.55, label=f"{name} UB")
        if mark_final and name == "B&P":
            for series, alpha in ((lb, 1.0), (ub, 0.55)):
                last = [(tt, v) for tt, v in zip(t, series) if not math.isnan(v)]
                if last and last[-1][1] >= 0:
                    ax.plot(*last[-1], marker="o", markersize=FIG_STYLE["marker"],
                            color="C0", alpha=alpha,
                            linestyle="none", clip_on=False, zorder=5)
    if ours.get("published") is not None:
        ax.axhline(ours["published"], color="0.4", linestyle=":",
                   label="Published")
    elif best_known_ref:
        bk = best_known(inst, fig_key)
        if bk is not None:
            ax.axhline(bk, color="0.4", linestyle=":", label="Best known")
    # the objective is non-negative, so a bound below 0 carries no information;
    # early Lagrangian bounds can be very negative and would flatten the plot
    ax.set_ylim(bottom=0)
    ax.set_xscale("log")
    ax.set_xlim(1, t_end)
    ax.set_xlabel("Time (s, log scale)")
    ax.set_ylabel("Objective")


def figures(tags: dict) -> None:
    """fig_bounds_b50_all: bounds over time for all 10 b50 instances (2 x 5)."""
    from matplotlib.lines import Line2D
    rc = {"xtick.labelsize": FIG_STYLE["tick"], "ytick.labelsize": FIG_STYLE["tick"],
          "axes.labelsize": FIG_STYLE["label"], "lines.linewidth": FIG_STYLE["lw"]}
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 5, figsize=FIG_STYLE["figsize"], sharex=False)
        for z, ax in enumerate(axes.flat):
            inst = f"b50_{z}"
            plot_bounds(ax, inst, tags["b50"], "fig_bounds_b50_all",
                        mark_final=True, best_known_ref=True)
            ax.text(0.03, 0.97, inst, transform=ax.transAxes, ha="left", va="top",
                    fontsize=FIG_STYLE["inst"], zorder=6,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
            if z % 5:
                ax.set_ylabel("")
            if z < 5:
                ax.set_xlabel("")
        # one shared legend above the panels (an in-panel legend hides curves)
        handles = [Line2D([], [], color="C0", label="B&P LB"),
                   Line2D([], [], color="C0", alpha=0.55, label="B&P UB"),
                   Line2D([], [], color="C1", linestyle="--", label="Compact+VI LB"),
                   Line2D([], [], color="C1", linestyle="--", alpha=0.55, label="Compact+VI UB"),
                   Line2D([], [], color="C0", marker="o", markersize=FIG_STYLE["marker"],
                          linestyle="none", label="B&P final value"),
                   Line2D([], [], color="0.4", linestyle=":",
                          label="Published (b50_0: best known)")]
        fig.legend(handles=handles, loc="upper center", ncol=3,
                   fontsize=FIG_STYLE["legend"], frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(os.path.join(FIG, "fig_bounds_b50_all.pdf"))
        fig.savefig(os.path.join(FIG, "fig_bounds_b50_all.png"), dpi=150)
        plt.close(fig)


def main() -> None:
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    tags = {"b25": tab_b25(), "b50": tab_size("b50"), "b75": tab_size("b75")}
    fig_tags = {"b50": tags["b50"][0]}
    tab_ablation()
    tab_dp()
    figures(fig_tags)
    for size, tag in tags.items():
        print(f"{size}: tree tag {tag}")
    for key, files in used.items():
        print(f"{key}: " + ", ".join(sorted(set(files))))


if __name__ == "__main__":
    main()
