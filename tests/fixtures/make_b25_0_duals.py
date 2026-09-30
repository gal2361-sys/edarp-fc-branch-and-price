"""Regenerate tests/fixtures/b25_0_duals.json: master duals of a b25_0 root column
generation at iterations 2, 4, 6, 8, 10 (plain MILP pricing, pool 50).

    PYTHONPATH=. .venv/bin/python tests/fixtures/make_b25_0_duals.py
"""

import json
import os

import gurobipy as gp

from bp.master import RestrictedMaster
from bp.pricing import PricingMILP
from edarp import load_instance

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KEEP = {2, 4, 6, 8, 10}

inst = load_instance(os.path.join(ROOT, "data/instances/b25_0.txt"),
                     os.path.join(ROOT, "data/instances/circuit_artificial_instances_0.txt"), "b25_0")
env = gp.Env(params={"OutputFlag": 0, "Threads": 1})
pricing = PricingMILP(inst, time_limit=60, pool_size=50, env=env, restricted_k=None, early_stop=None)
master = RestrictedMaster(inst, env=env)
master.add_initial_columns(pricing)
master.set_artificial_cost(2 * max(c.cost for c in master.columns))
out = []
for it in range(1, max(KEEP) + 1):
    sol = master.solve()
    if it in KEEP:
        out.append({"iteration": it, "lp_value": sol.value, "pi": sol.pi, "mu": sol.mu})
    res = pricing.solve(sol.pi, sol.mu, prove=True)
    for c in res.columns:
        master.add_column(c)
with open(os.path.join(ROOT, "tests/fixtures/b25_0_duals.json"), "w") as f:
    json.dump(out, f, indent=1)
print([(d["iteration"], round(d["lp_value"], 3)) for d in out])
