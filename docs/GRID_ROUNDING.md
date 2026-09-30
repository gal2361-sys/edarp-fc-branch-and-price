# Objective grid: when may a lower bound be rounded up?

**Result.** For every instance in this project (integer travel times, service time,
earliest pickups, loads, beta_d, beta_m, and beta_c = 1) some optimal solution has all
times and charging durations in (1/2)Z. Hence the optimal objective lies in
0.85 Z + 0.075 Z, a subset of 0.025 Z, and any valid lower bound may be rounded up to the
next multiple of 0.025. If every deadline d_i is an integer the step is 0.05. All 30
artificial instances have half-integer deadlines, so their step is 0.025
(`bp.grid.grid_step`). The stronger claim below (step 0.05) is **not** proven.

## The claim that was to be checked

"Every feasible column cost is a multiple of 0.05, because 0.85 * laps and
0.15 * q_i * excess with integer excess."

1. **False for columns.** A column is a feasible plan with its charging durations, and
   charging is continuous. The published b25_0 vehicle 0 with its lap-1 charge raised
   from 8 to 8.3 is feasible and costs 6.185 = 123.7 / 20
   (`tests/test_grid.py::test_feasible_column_can_be_off_grid`). Pricing columns come
   from a MILP with continuous charging, so off-grid columns do occur. An assertion on
   every column in `Column.evaluate` would reject valid columns; none was added.
2. **What matters is the optimum.** Rounding a bound LB up to a lattice point is valid
   if the optimal value OPT lies on the lattice: OPT >= LB implies OPT >= ceil(LB).

## Proof that some optimal solution has times in (1/2)Z

Take an optimal solution and fix, for each vehicle, its discrete part: performed laps,
stops x, pickups u (hence the dropoff laps), and which depot stops may charge. The
remaining variables of one vehicle are continuous: service times T[s,l], charging
durations c_l, battery levels B_l, dropoff times tau_i. The cost is the sum over
vehicles of 0.85 * laps (fixed) + 0.15 * sum_i q_i (tau_i - eps_i), so it suffices that
each vehicle's continuous problem (an LP, "minimise sum q_i tau_i") has an optimal
vertex in (1/2)Z.

Substitute cumulative charge S_l = c_0 + ... + c_l (S_{-1} = 0) and shifted times
R[s,l] = T[s,l] - S_l for s >= 1, R[0,l] = T[0,l] - S_{l-1} (depot arrival, before
charging), rho_i = tau_i - S_{l'} with l' the lap of the depot passage preceding i's
dropoff. With beta_c = 1, every constraint of the compact model becomes a difference
constraint or a simple bound:

| constraint | original | after substitution |
|---|---|---|
| (14b) depot to station 1 | T[1,l] >= T[0,l] + c_l + t_1 | R[1,l] - R[0,l] >= t_1 |
| (14c) station to station | T[s,l] >= T[s-1,l] + sigma x + t_s | R[s,l] - R[s-1,l] >= const |
| (14a) lap to lap | T[0,l] >= T[S-1,l-1] + sigma x + t_0 | R[0,l] - R[S-1,l-1] >= const |
| (16) earliest pickup | T[s,l] >= a_i | R[s,l] - (-S_l) >= a_i |
| (17) dropoff time | tau_i >= T[...] + sigma x + t | rho_i - R[...] >= const |
| (18) deadline | tau_i <= d_i | rho_i - (-S_{l'}) <= d_i |
| (6)-(9) battery | B_l = S_{l-1} - l beta_d in [0, beta_m], c_l >= 0 | bounds on S_l; (-S_{l-1}) - (-S_l) >= 0 |
| no depot stop | c_l = 0 | S_l - S_{l-1} = 0 |

In the variables (R, rho, -S) every row has at most one +1 and one -1: the transpose of
a network matrix plus unit rows, which is totally unimodular. The right-hand sides are
integers except the deadlines d_i = a_i + 1.5 delta_i + 15 and the horizon bounds
derived from them, which are in (1/2)Z. Multiplying all variables by 2 gives integer
right-hand sides, so every vertex has all variables in (1/2)Z. The LP is feasible (the
optimal solution restricted to this vehicle), bounded below (sum q_i tau_i >= 0), and
its polyhedron is pointed (S has finite bounds and every other variable is linked to
S or to the fixed T[0,0] = 0 by difference constraints), so an optimal vertex exists.
Replacing each vehicle's schedule by such a vertex keeps the solution optimal.

Then tau_i is in (1/2)Z, eps_i = a_i + sigma + delta_i is an integer, q_i is an integer,
so each excess term is in (1/2)Z and OPT is in 0.85 Z + 0.15 * (1/2) Z = 0.85 Z + 0.075 Z,
a subset of 0.025 Z (gcd(34, 3) = 1 in units of 0.025).

## Why the step 0.05 is not proven

It would need tau_i integer at an optimum. The earliest schedule for fixed integer
charges is integral, but charges are decisions: charging more at an early depot stop
(in idle time) can avoid a later charge that delays several passengers, and that early
charge may be capped by a passenger's half-integer deadline, which then binds at the
optimum. This is the same mechanism that disproved just-in-time charging
(`tests/test_pricing_dp.py::test_a_jit_charging_is_not_exact`). No counterexample with
a half-integer optimal excess has been constructed either; the question is open.

## Use in the code

`bp.grid.grid_step(inst)` returns 0.025 (0.05 if all d_i are integers) when the
conditions hold and None otherwise. With `run_tree.py --grid-round` the tree rounds
each node bound up with `bp.grid.round_up` (first lowering it by 1e-5 against float
noise) and uses the rounded value only for pruning and the global bound; such nodes are
logged with lb_kind "grid". The flag is off by default.
