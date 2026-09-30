# Branch-and-price for the electric dial-a-ride problem on a fixed circuit

Course project, Exact Algorithms for Combinatorial Optimization, Tel Aviv University
(Gal Gussarsky and Adi Shalit). Identical electric shuttles run laps of a fixed circular
route; the depot is the only charging point. Each request has pickup and dropoff
stations, a passenger count, an earliest pickup and a latest dropoff time. The goal is to
assign requests to vehicles and laps, choose stops and charging, and minimise
0.85 × laps + 0.15 × total passenger excess time (Molenbruch, Braekers, Eisenhandler and
Kaspi, Transportation Science 57(3), 594-612, 2023). This repository contains the compact
MILP from that paper (with its Electronic Companion B valid inequalities), a
branch-and-price method with Ryan-Foster branching, several pricing variants (MILP,
LNS heuristic, a labeling prototype), and the scripts that produce every table and
figure of our report.

## Requirements

- Python 3.14 (developed on 3.14.7).
- Gurobi through `gurobipy` 13 (13.0.3). A licence is required; we use an academic WLS
  licence in `~/gurobi.lic`. The licence file is never part of this repository.
- The packages in `requirements.txt`:

```
python3.14 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All results were obtained on an Apple M4 (10 cores, 16 GiB RAM), macOS 15.7.1, with
Gurobi's default thread count (up to 10). Running times depend on the machine.

## Data

`data/instances/` and `data/solutions/` are the instance and best-solution files of
Molenbruch et al. (2023), as distributed by the authors
(autolab.sites.tau.ac.il/publications). Only the artificial instances `b{n}_{z}` with
n = 25, 50, 75 and z = 0..9 are used; the Renmark files are included but untested.

**Request file** `b{n}_{z}.txt` (from `data/instances/structure_of_the_files.txt`):

- first line: number of requests, vehicle capacity, battery usage per lap, end of the
  time horizon;
- one line per request: id, pickup station, dropoff station, earliest pickup time,
  number of passengers.

**Circuit file** `circuit_artificial_instances_{z}.txt`:

- first line: number of stations, stop (service) duration;
- then the travel times between consecutive stations. The first value is the segment
  depot -> station 1 and the last is the segment back to the depot. This orientation
  was established by checking the published b25_0 solution against our model.

Instance `b{n}_{z}` is paired with circuit `z`. Parameters not in the files come from
the paper: beta_c = 1, beta_m = 11, V = 3 (n = 25, 50) and 4 (n = 75), latest dropoff
d_i = a_i + 1.5 × delta_i + 15. With these, the compact model reproduces the published
optima of b25_0, b25_1 and b25_2.

**Solution file** `data/solutions/b{n}_{z}_best_solution.txt`: objective, total delay,
number of laps, then per vehicle one line per lap, `Lap l[b]: node[time,load,station] ...`
(node < n is the pickup of request node, node >= n the dropoff of request node - n; b is
the charging time before the lap).

## Running one instance

Each script writes one JSON to `results/` and prints a summary.

```
# compact MILP (Gurobi), with the Electronic Companion B valid inequalities, 30 min
python run_compact.py b25_0 --valid-inequalities --time-limit 1800

# root column generation only
python run_root.py b25_0 --time-limit 1800

# branch-and-price (paper settings; defaults shown where they matter)
python run_tree.py b25_1 --time-limit 1800                          # b25: node limit 600 s
python run_tree.py b50_0 --time-limit 1800 --node-time-limit 1800   # b50, b75
python run_tree.py b75_0 --time-limit 1800 --node-time-limit 1800 --pricing-time-limit 120
```

Main `run_tree.py` flags and their defaults: `--pricing-time-limit 30` (per MILP pricing
call), `--pool-size 50`, `--lns-time 2`, `--restricted-k 20`, `--early-stop 5`,
`--prove-time-limit 1200`, `--node-time-limit 600`, `--ip-time-limit 10` (integer master
after every node), `--scorer most_fractional`, `--initial-solution` (on),
`--grid-round` (off), `--seed 0`. Every acceleration can be switched off
(`--lns-time 0`, `--restricted-k 0`, `--early-stop 0`, `--no-dual-pruning`,
`--no-valid-inequalities`, `--no-initial-solution`).

## Reproducing the tables and figures

`run_batch.py <mode> <sizes> --tag <tag> [--summary <name>] -- <flags>` runs all 10
instances of each size sequentially and writes `results/summary_<name>.csv` (the name
defaults to the tag). Run one batch at a time; timings are part of the results.

| table | summaries it reads | commands |
|---|---|---|
| `tab_b25` | `tree_v1_b25`, `compact_vi_b25` | `run_batch.py tree b25 --tag tree_v1 --summary tree_v1_b25 -- --no-initial-solution` and `run_batch.py compact b25 --tag compact_vi_b25 -- --valid-inequalities --time-limit 1800` |
| `tab_b50` | `tree_v2_b50`, `compact_vi_b50_b75` | `run_batch.py tree b50 --tag tree_v2 --summary tree_v2_b50 -- --node-time-limit 1800` and `run_batch.py compact b50 b75 --tag compact_vi_b50_b75 -- --valid-inequalities --time-limit 1800` |
| `tab_b75` | `tree_v2_b75`, `tree_v2_p120_b75`, `compact_vi_b50_b75` | `run_batch.py tree b75 --tag tree_v2 --summary tree_v2_b75 -- --node-time-limit 1800`, the same with `--tag tree_v2_p120 --summary tree_v2_p120_b75 -- --node-time-limit 1800 --pricing-time-limit 120`, and the compact batch above |
| `tab_ablation` | `abl_*` | `run_batch.py root b25 --tag <t> --summary <t> -- <flags>` for `abl_full` (no flags), `abl_no_lns` (`--lns-time 0`), `abl_no_restricted` (`--restricted-k 0`), `abl_no_pruning` (`--no-dual-pruning`), `abl_no_vi` (`--no-valid-inequalities`), `abl_no_earlystop` (`--early-stop 0`), `abl_stab` (`--stabilize`) |
| `tab_dp` | `dp_prototype_report.json` | `python dp_report.py` (needs `results/root_b50_0_artfix600.json` from `run_root.py b50_0 --time-limit 600 --tag artfix600`) |

Notes on code versions:

- **tree_v1 batches** (`tree_v1_b25`, and `tree_v1_b50`/`tree_v1_b75`, which the report
  quotes in text) were run before the initial-solution heuristic existed. The current
  code with `--no-initial-solution` is the closest equivalent. It still differs in two
  bookkeeping changes that do not affect the search: bounds are now also logged during
  a node, and the LNS time limit is restored between nodes.
- **The compact model** and **the root ablation code** are unchanged since those
  batches were run.
- **The tree_v2 batches** used the current code. `run_queue.sh` runs the three v2 tree
  batches in sequence under `nohup`, logging to `results/queue.log`.

Then:

```
python make_tables.py
```

It writes the five tables `docs/paper/tables/*.tex` and the figure
`docs/paper/figures/fig_bounds_b50_all.{pdf,png}` (lower and upper bounds over time for
all 10 b50 instances, ours and compact+VI), and prints which files fed each one. The
tables need only the tracked summaries, `results/dp_prototype_report.json` and
`data/solutions/`. The figure also reads the per-run JSONs `tree_b50_*_tree_v2.json`
and `compact_b50_*_vi.json`, which are not tracked (see `results/README.md`).
`tab_b50` and `tab_b75` use the newest tree tag available (tree_v2, else tree_v1).

## Results and JSON contents

`results/` holds one JSON per run (local only), the tracked summaries, and logs.

**All runs**: `instance`, `status`, `ub`, `lb`, `gap`, `time_total` and a
`bound_log`, a list of `[time_s, lb, ub]` appended at every change of either bound.

- **`compact_<inst>_c<z>[_vi].json`** (`run_compact.py`): Gurobi's bounds from the MIP
  callback. `bound_log[0]` is the value before the root, `bound_log[1]` the root LP
  relaxation; `ub = 1e100` means no incumbent yet. Also `nodes`, `n_vars`, `n_constrs`,
  `published` and `matches_published` where a published optimum exists,
  `valid_inequalities`.
- **`root_<inst>_<tag>.json`** (`run_root.py`): `root_lp_bound` (a valid bound only if
  `converged`), `lagrangian_lb` and `lagrangian_lb_iteration`, `heuristic_ub` (integer
  master over the generated columns), `artificial_active`, `stages` and `pricing_stats`
  (calls and hits per pricing stage), `columns_dropped` and `dropped_reasons` (plans
  rejected by the exact evaluator), `final_pi`/`final_mu` (duals), `positive_columns`,
  `prove_trajectories` (bound of each long prove call every 5 s), and `log` (one row per
  pricing call: iteration, stage, LP value, best reduced cost, pricing dual bound,
  columns added, times, Lagrangian bound).
- **`tree_<inst>_<tag>.json`** (`run_tree.py`): `status` (optimal / time_limit), `ub`,
  `lb`, `gap`, `root_lb` and `root_lb_kind`, `lb_kind` (`lp` when column generation was
  proven converged, `lagrangian`, `parent`, `optimal`), `nodes_processed`,
  `nodes_pruned`, `nodes_open_at_end`, `max_depth`, `columns_generated`,
  `pricing_calls`, `time_pricing`, `time_scorer`, `incumbent` (served sets) and
  `incumbent_source`, `initial_solution`, `matches_published`, `config` (all flags) and
  `node_log` (one entry per node: branching decision, bound and its kind, whether
  pricing was proven, time, columns added, outcome). In `bound_log`, `null` stands for
  an infinite value: the first entry of a run with an initial solution is
  `[t, null, UB]`. The lower bound is logged during a node too, from each improvement of
  its Lagrangian bound, and after every node.
- **`summary_<name>.csv`** (`run_batch.py`): one row per instance. Tree batches have
  root_lb, root_lb_kind, lb, lb_kind, ub, ub_source, initial_ub, gap, status, nodes,
  nodes_pruned, max_depth, time_total, time_pricing, published, matches_published,
  exit_code. Root and compact batches have the corresponding fields.

## Tests

```
pytest                 # 93 tests, about 25-40 minutes, needs Gurobi
pytest -m "not slow"   # skips the 8 long exactness checks
```

The tests cover:

- **Data and model:** the published b25_0 solution is feasible in our model at its
  published cost.
- **Valid inequalities:** each family is checked alone and together, and the pricing
  optimum is unchanged on sampled duals.
- **Components:** exact column evaluation, pricing (MILP, LNS, labeling) against the
  MILP optimum, master and column generation bookkeeping, the tree on toy instances
  against the compact optimum, the objective grid, and the initial heuristic.

One case is a strict expected failure: the initial heuristic misses a cover on toy
(n=10, seed 2).

## Source map

```
edarp.py              instance loader, derived parameters, compact MILP, valid inequalities
run_compact.py        compact MILP on one instance
run_root.py           root column generation on one instance
run_tree.py           branch-and-price on one instance
run_batch.py          one mode over all 10 instances of a size -> summary CSV
run_queue.sh          sequential experiment queue (nohup)
make_tables.py        paper tables and figures from results/
dp_report.py          labeling-pricing measurements -> results/dp_prototype_report.json
bp/column.py          Column (one vehicle's plan), exact evaluation of a plan
bp/published.py       parser for the published solution files
bp/pricing.py         MILP pricing: restricted, early stop, prove call, dual pruning
bp/pricing_lns.py     LNS pricing heuristic
bp/pricing_dp.py      labeling pricing prototype (exact, not used in the tree)
bp/master.py          restricted master LP, artificial columns, integer master
bp/colgen.py          column generation loop, Lagrangian bound, stabilisation (flag)
bp/tree.py            branch-and-price over Ryan-Foster pairs, bound log
bp/scorers/           branching pair scorers: most_fractional, random
bp/initial.py         constructive full-fleet heuristic for the initial upper bound
bp/grid.py            objective grid step and bound rounding (docs/GRID_ROUNDING.md)
tests/                pytest suite (fixture duals in tests/fixtures/)
docs/GRID_ROUNDING.md proof that the optimum lies on a 0.025 grid
docs/paper/           tables and figures generated by make_tables.py
```
