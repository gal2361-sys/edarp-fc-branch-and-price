# results/

Tracked here (small, needed for the paper tables):

- `summary_*.csv`: one row per instance of a batch (`run_batch.py`); the columns are
  listed in the top-level README.
- `grid_test.json`: outcome of `tests/test_grid.py::test_generated_columns_lie_on_the_grid`.
- `dp_prototype_report.json`: labeling-pricing measurements (`dp_report.py`).

`runs/`: the per-run JSONs behind every summary CSV, table and figure (172 files):
`tree_*`, `root_*` and `compact_*_vi` for each summary row (including the b25 rerun behind
Table 1), plus `root_b50_0_artfix600.json`, the input of `dp_report.py`, and
`compact_b25_0_c0.json`, the run behind the root bound without valid inequalities.
`make_tables.py` reads them from here.

Not tracked (gitignored): batch logs (`batch_<tag>_<instance>.log`), the queue log
(`queue.log`), argument records (`summary_<name>.args`) and per-run JSONs of runs not
behind any reported number. New runs write their JSON to `results/` directly. The table
below lists the per-run JSONs on the authors' machine when this file was written; those
not in `runs/` are available on request.

| pattern | files | produced by |
|---|---|---|
| `compact_b25_<z>_c<z>.json` | 12 | run_compact.py |
| `compact_b25_<z>_c<z>_vi.json` | 10 | run_compact.py |
| `compact_b50_<z>_c<z>.json` | 1 | run_compact.py |
| `compact_b50_<z>_c<z>_vi.json` | 10 | run_compact.py |
| `compact_b75_<z>_c<z>_vi.json` | 10 | run_compact.py |
| `root_b25_<z>_abl_full.json` | 10 | run_root.py |
| `root_b25_<z>_abl_no_earlystop.json` | 10 | run_root.py |
| `root_b25_<z>_abl_no_lns.json` | 10 | run_root.py |
| `root_b25_<z>_abl_no_pruning.json` | 10 | run_root.py |
| `root_b25_<z>_abl_no_restricted.json` | 10 | run_root.py |
| `root_b25_<z>_abl_no_vi.json` | 10 | run_root.py |
| `root_b25_<z>_abl_stab.json` | 10 | run_root.py |
| `root_b25_<z>_artfix.json` | 1 | run_root.py |
| `root_b25_<z>_fast.json` | 1 | run_root.py |
| `root_b25_<z>_final.json` | 10 | run_root.py |
| `root_b25_<z>_p50.json` | 3 | run_root.py |
| `root_b25_<z>_p50_stab.json` | 1 | run_root.py |
| `root_b50_<z>_artfix600.json` | 1 | run_root.py |
| `root_b50_<z>_cheap.json` | 1 | run_root.py |
| `root_b50_<z>_fast.json` | 1 | run_root.py |
| `root_b50_<z>_final.json` | 10 | run_root.py |
| `root_b50_<z>_p50.json` | 1 | run_root.py |
| `tree_b25_<z>_tree_v1.json` | 10 | run_tree.py |
| `tree_b50_<z>_tree_v1.json` | 10 | run_tree.py |
| `tree_b50_<z>_tree_v2.json` | 10 | run_tree.py |
| `tree_b75_<z>_tree_v1.json` | 10 | run_tree.py |
| `tree_b75_<z>_tree_v2.json` | 10 | run_tree.py |
| `tree_b75_<z>_tree_v2_p120.json` | 10 | run_tree.py |

Compact files named `compact_b25_0_c<z>.json` with z != 0 come from pairing b25_0 with
the other nine circuit files during data validation; the compact results without `_vi`
are the model without valid inequalities.
