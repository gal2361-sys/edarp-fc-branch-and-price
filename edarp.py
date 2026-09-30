"""
eDARP-FC: instance loader + compact MILP.

Step 0 of the project. Purpose: validate that our parsing and our reading of the
formulation are correct, by reproducing published optima from Molenbruch et al.
(2023), Table 10. Only once a25_0 comes out at 17.45 do we start on branch-and-price.

Parameters not present in the instance files, taken from the paper:
  artificial (a*): S=6, sigma=1, theta=24, C=6, beta_c=1, beta_d=4, beta_m=11,
                   V=3 for n in {25,50}, V=4 for n=75
  Renmark (r*):    S=10, sigma=1, theta=30, C=10, beta_c=1, beta_d=16, beta_m=40
  latest dropoff:  d_i = a_{i-n} + gamma1 * delta(i-n, i) + gamma2,
                   baseline gamma1=1.5, gamma2=15
  objective:       alpha1=0.85, alpha2=0.15

Gurobi: a licence is required (see README.md); the pip-installed gurobipy alone is
size-limited and cannot solve the b50 and b75 instances.
"""

from __future__ import annotations

import math
from fractions import Fraction
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

ALPHA1 = 0.85
ALPHA2 = 0.15
GAMMA1 = 1.5
GAMMA2 = 15.0


# --------------------------------------------------------------------------
# instance
# --------------------------------------------------------------------------


@dataclass
class Instance:
    name: str

    # from the requests file
    n: int                  # number of requests
    C: int                  # vehicle passenger capacity
    beta_d: float           # battery discharge per lap
    H_file: float           # planning horizon as given in the file

    pickup_station: list[int]   # s(i) for i in P, length n
    drop_station: list[int]     # s(i+n) for i in P, length n
    a: list[float]              # earliest pickup time, length n
    q: list[int]                # passengers, length n

    # from the infrastructure file
    S: int                  # number of stations, 0 = depot
    sigma: float            # service duration per stop
    t: list[float]          # t[s] = travel time INTO station s from its predecessor

    # from the paper, not in the files
    V: int
    beta_c: float
    beta_m: float

    # derived
    theta: float = field(init=False)
    L: int = field(init=False)
    H: float = field(init=False)
    d: list[float] = field(init=False)        # latest dropoff, per request
    eps: list[float] = field(init=False)      # earliest possible dropoff, per request
    I: list[int] = field(init=False)          # 1 if the request crosses the depot, else 0
    delta: list[float] = field(init=False)    # direct pickup->dropoff travel time

    def __post_init__(self) -> None:
        assert len(self.t) == self.S, "expected one travel time per station"
        self.theta = sum(self.t)

        self.delta = [
            self._arc_time(self.pickup_station[i], self.drop_station[i])
            for i in range(self.n)
        ]
        self.d = [
            self.a[i] + GAMMA1 * self.delta[i] + GAMMA2 for i in range(self.n)
        ]
        self.eps = [self.a[i] + self.sigma + self.delta[i] for i in range(self.n)]

        # I(i) as in the course document: 0 when the dropoff station is after the
        # pickup station on the circle (same-lap request), 1 otherwise, i.e. the
        # request crosses the depot and is dropped one lap later.
        self.I = [
            1 if self.drop_station[i] < self.pickup_station[i] else 0
            for i in range(self.n)
        ]

        lap_time = self.theta + self.sigma + self.beta_d / self.beta_c
        self.L = int(min(math.ceil(max(self.d) / lap_time), 2 * self.n))
        self.H = self.theta + max(self.d)

    def _arc_time(self, s_from: int, s_to: int) -> float:
        """Travel time from station s_from to station s_to along the circle."""
        if s_from <= s_to:
            return sum(self.t[k] for k in range(s_from + 1, s_to + 1))
        return sum(self.t[k] for k in range(s_from + 1, self.S)) + sum(
            self.t[k] for k in range(0, s_to + 1)
        )

    # request sets by station, as in the formulation
    def Theta(self, s: int) -> list[int]:
        """Pickup requests located at station s."""
        return [i for i in range(self.n) if self.pickup_station[i] == s]

    def Phi_plus(self, s: int) -> list[int]:
        """Dropoffs at s whose pickup is earlier on the circle (same lap)."""
        return [
            i
            for i in range(self.n)
            if self.drop_station[i] == s and self.pickup_station[i] < s
        ]

    def Phi_minus(self, s: int) -> list[int]:
        """Dropoffs at s whose pickup is later on the circle (previous lap)."""
        return [
            i
            for i in range(self.n)
            if self.drop_station[i] == s and self.pickup_station[i] > s
        ]

    def summary(self) -> str:
        crossing = sum(1 for i in range(self.n) if self.I[i] == 1)
        return (
            f"{self.name}: n={self.n} S={self.S} V={self.V} C={self.C} "
            f"theta={self.theta:g} sigma={self.sigma:g} "
            f"beta(c/d/m)={self.beta_c:g}/{self.beta_d:g}/{self.beta_m:g} "
            f"L={self.L} H={self.H:g} depot-crossing={crossing}/{self.n}"
        )


def _numbers(line: str) -> list[float]:
    return [float(tok) for tok in line.replace(",", " ").split()]


def load_instance(requests_path: str, infra_path: str, name: str = "") -> Instance:
    """Read the two input files described in the project document.

    requests file
        line 1: n, C, beta_d, H
        line k: id, pickup station, dropoff station, earliest pickup, passengers

    infrastructure file
        line 1: S, sigma
        line k: travel time between consecutive stations

    Circuit orientation: the first travel time in the file is the segment
    depot (station 0) -> station 1, the second is 1 -> 2, and the last is
    (S-1) -> depot. Since t[s] is the time INTO station s, we store
    t[1] = first value, ..., t[S-1] = second to last, t[0] = last value.
    Validated by feeding the published b25_0 solution (Molenbruch et al. 2023)
    into the model: only this reading reproduces its excess time (37) and
    objective (17.45); the unrotated reading gives excess 42.
    """
    with open(requests_path) as f:
        rlines = [ln for ln in (x.strip() for x in f) if ln]
    with open(infra_path) as f:
        ilines = [ln for ln in (x.strip() for x in f) if ln]

    head = _numbers(rlines[0])
    n, C, beta_d, H_file = int(head[0]), int(head[1]), head[2], head[3]

    pickup, drop, a, q = [], [], [], []
    for ln in rlines[1 : 1 + n]:
        vals = _numbers(ln)
        pickup.append(int(vals[1]))
        drop.append(int(vals[2]))
        a.append(vals[3])
        q.append(int(vals[4]))

    ihead = _numbers(ilines[0])
    S, sigma = int(ihead[0]), ihead[1]
    t = []
    for ln in ilines[1:]:
        t.extend(_numbers(ln))
    # file order is depot->1, 1->2, ..., (S-1)->depot; t[s] is the time INTO s
    t = t[:S]
    t = [t[-1]] + t[:-1]

    name = name or requests_path.split("/")[-1]
    artificial = not name.lower().startswith("r")

    if artificial:
        beta_c, beta_m = 1.0, 11.0
        V = 3 if n <= 50 else 4
    else:
        beta_c, beta_m = 1.0, 40.0
        V = 4 if n <= 100 else 9

    return Instance(
        name=name,
        n=n,
        C=C,
        beta_d=beta_d,
        H_file=H_file,
        pickup_station=pickup,
        drop_station=drop,
        a=a,
        q=q,
        S=S,
        sigma=sigma,
        t=t,
        V=V,
        beta_c=beta_c,
        beta_m=beta_m,
    )


# --------------------------------------------------------------------------
# compact MILP, constraints (1)-(24) of the project document
# --------------------------------------------------------------------------


VI_FAMILIES = frozenset(f"B{k}" for k in range(1, 13))


def _vi_families(flag, pricing: bool) -> frozenset[str]:
    """Normalize the valid_inequalities flag to a set of family names."""
    if flag is True:
        return VI_FAMILIES - ({"B2"} if pricing else set())
    if not flag:
        return frozenset()
    fams = frozenset(flag)
    unknown = fams - VI_FAMILIES
    if unknown:
        raise ValueError(f"unknown valid-inequality families {sorted(unknown)}")
    if pricing and "B2" in fams:
        raise ValueError("B2 (vehicle symmetry breaking) is invalid in the pricing model")
    return fams


def latest_drop_lap(inst: Instance, i: int) -> int:
    """l_i = ceil(d_i / lap_time) - 1, computed exactly (no float rounding)."""
    lap_time = Fraction(inst.theta) + Fraction(inst.sigma) + Fraction(inst.beta_d) / Fraction(inst.beta_c)
    return math.ceil(Fraction(inst.d[i]) / lap_time) - 1


def build_compact(
    inst: Instance,
    env: gp.Env | None = None,
    pricing: bool = False,
    valid_inequalities: bool | set[str] | frozenset[str] = False,
) -> tuple[gp.Model, dict[str, gp.tupledict]]:
    """Build the compact MILP. Returns (model, vars) with vars keyed by
    "x","u","w","b","B","T","Q","tau" (and "y" when pricing=True or when a
    valid-inequality family needs it).

    valid_inequalities: False, True (all families of Electronic Companion B that
    apply), or a set of family names such as {"B3", "B7"}. See _add_valid_inequalities.

    pricing=True builds the single-vehicle pricing model: V must be 1, (2) becomes
    "<= 1" so the vehicle serves any subset, y[i] = sum_l u[i,l,0] marks request i
    as served, and (18) plus the earliest-dropoff tightening are multiplied by
    y[i], so an unserved request has tau = 0. The objective is the column cost
    0.85 * laps + 0.15 * sum_i q_i * (tau_i - eps_i * y_i); callers subtract duals.
    """
    if pricing and inst.V != 1:
        raise ValueError("pricing model requires inst.V == 1")
    S, L, V, n = inst.S, inst.L, inst.V, inst.n
    t, sigma, C, H = inst.t, inst.sigma, inst.C, inst.H
    bc, bd, bm = inst.beta_c, inst.beta_d, inst.beta_m

    m = gp.Model(f"edarp_{inst.name}", env=env)

    # (14a-c) chain T through every lap in range(L), performed or not, so T on
    # empty trailing laps keeps growing (by at least theta per lap, plus sigma per
    # stop) past H. The document's model puts no upper bound on T; H is only the
    # big-M of (17). BIG_T bounds T on any lap and replaces H as that big-M.
    BIG_T = H + L * (inst.theta + S * sigma)

    x = m.addVars(S, L, V, vtype=GRB.BINARY, name="x")          # stop at station
    u = m.addVars(n, L, V, vtype=GRB.BINARY, name="u")          # request on lap
    w = m.addVars(L, V, vtype=GRB.BINARY, name="w")             # lap performed
    b = m.addVars(L, V, lb=0.0, name="b")                       # charging duration
    B = m.addVars(L, V, lb=0.0, ub=bm, name="B")                # battery at lap start
    T = m.addVars(S, L, V, lb=0.0, ub=BIG_T, name="T")          # service start time
    Q = m.addVars(S, L, V, lb=0.0, ub=C, name="Q")              # onboard on arrival
    tau = m.addVars(n, lb=0.0, ub=H, name="tau")                # dropoff time
    y = m.addVars(n, vtype=GRB.BINARY, name="y") if pricing else None  # served

    # (1)
    if pricing:
        m.setObjective(
            ALPHA1 * gp.quicksum(w[l, 0] for l in range(L))
            + ALPHA2 * gp.quicksum(
                inst.q[i] * (tau[i] - inst.eps[i] * y[i]) for i in range(n)
            ),
            GRB.MINIMIZE,
        )
    else:
        m.setObjective(
            ALPHA1 * gp.quicksum(w[l, v] for l in range(L) for v in range(V))
            + ALPHA2 * gp.quicksum(inst.q[i] * (tau[i] - inst.eps[i]) for i in range(n)),
            GRB.MINIMIZE,
        )

    # (2) every request served exactly once; a depot-crossing request (I=1)
    #     cannot start on the last lap, since it is dropped one lap later
    for i in range(n):
        last = L - inst.I[i]              # laps 0..L-1-I(i); I=1 cannot start last lap
        served = gp.quicksum(u[i, l, v] for l in range(last) for v in range(V))
        if pricing:
            m.addConstr(served <= 1, name=f"assign_{i}")
            m.addConstr(y[i] == served, name=f"served_{i}")
        else:
            m.addConstr(served == 1, name=f"assign_{i}")
        for l in range(last, L):
            for v in range(V):
                m.addConstr(u[i, l, v] == 0, name=f"nolastlap_{i}_{l}_{v}")

    # (3a) pickup stop, (3b) dropoff stop on lap l + I(i): I=0 same lap, I=1 next
    for i in range(n):
        si, sj = inst.pickup_station[i], inst.drop_station[i]
        off = inst.I[i]                    # laps between pickup and dropoff
        for v in range(V):
            for l in range(L):
                m.addConstr(u[i, l, v] <= x[si, l, v], name=f"pick_{i}_{l}_{v}")
                if l + off < L:
                    m.addConstr(
                        u[i, l, v] <= x[sj, l + off, v], name=f"drop_{i}_{l}_{v}"
                    )

    # (4) stop only on a lap that is performed, (5) laps are consecutive
    for v in range(V):
        for l in range(L):
            for s in range(S):
                m.addConstr(x[s, l, v] <= w[l, v], name=f"stoplap_{s}_{l}_{v}")
            if l > 0:
                m.addConstr(w[l, v] <= w[l - 1, v], name=f"lapseq_{l}_{v}")

    # (6)-(9) battery
    for v in range(V):
        m.addConstr(B[0, v] == 0.0, name=f"bat0_{v}")                         # (7)
        for l in range(1, L):
            m.addConstr(                                                      # (6a)
                B[l, v] == B[l - 1, v] + bc * b[l - 1, v] - bd * w[l - 1, v],
                name=f"bat_{l}_{v}",
            )
        m.addConstr(                                                          # (6b)
            B[L - 1, v] + bc * b[L - 1, v] - bd * w[L - 1, v] >= 0,
            name=f"batend_{v}",
        )
        for l in range(L):
            m.addConstr(bc * b[l, v] <= bm * x[0, l, v], name=f"chgdepot_{l}_{v}")  # (8)
            m.addConstr(B[l, v] + bc * b[l, v] <= bm, name=f"chgcap_{l}_{v}")       # (9)

    # (10)-(13) passenger flow and capacity
    for v in range(V):
        m.addConstr(Q[0, 0, v] == 0.0, name=f"load00_{v}")                    # (11)
        for l in range(L):
            for s in range(1, S):
                board = gp.quicksum(
                    inst.q[i] * u[i, l, v] for i in inst.Theta(s - 1)
                )
                alight_same = gp.quicksum(
                    inst.q[i] * u[i, l, v] for i in inst.Phi_plus(s - 1)
                )
                if l == 0:                                                    # (10a)
                    m.addConstr(
                        Q[s, 0, v] == Q[s - 1, 0, v] + board - alight_same,
                        name=f"flow_{s}_0_{v}",
                    )
                else:                                                         # (10b)
                    alight_prev = gp.quicksum(
                        inst.q[i] * u[i, l - 1, v] for i in inst.Phi_minus(s - 1)
                    )
                    m.addConstr(
                        Q[s, l, v]
                        == Q[s - 1, l, v] + board - alight_same - alight_prev,
                        name=f"flow_{s}_{l}_{v}",
                    )
            if l > 0:                                                         # (10d)
                board = gp.quicksum(
                    inst.q[i] * u[i, l - 1, v] for i in inst.Theta(S - 1)
                )
                alight = gp.quicksum(
                    inst.q[i] * u[i, l - 1, v] for i in inst.Phi_plus(S - 1)
                )
                m.addConstr(
                    Q[0, l, v] == Q[S - 1, l - 1, v] + board - alight,
                    name=f"flowdepot_{l}_{v}",
                )
            m.addConstr(Q[1, l, v] == Q[0, l, v], name=f"flow1_{l}_{v}")      # (10c)
            m.addConstr(                                                      # (13)
                Q[0, l, v] <= C * (1 - x[0, l, v]), name=f"emptydepot_{l}_{v}"
            )

    # (14)-(16) timing
    for v in range(V):
        m.addConstr(T[0, 0, v] == 0.0, name=f"t00_{v}")                       # (15)
        for l in range(L):
            if l > 0:                                                         # (14a)
                m.addConstr(
                    T[0, l, v]
                    >= T[S - 1, l - 1, v] + sigma * x[S - 1, l - 1, v] + t[0],
                    name=f"tdepot_{l}_{v}",
                )
            m.addConstr(                                                      # (14b)
                T[1, l, v] >= T[0, l, v] + b[l, v] + t[1], name=f"t1_{l}_{v}"
            )
            for s in range(2, S):                                             # (14c)
                m.addConstr(
                    T[s, l, v] >= T[s - 1, l, v] + sigma * x[s - 1, l, v] + t[s],
                    name=f"t_{s}_{l}_{v}",
                )
            for i in range(n):                                                # (16)
                si = inst.pickup_station[i]
                m.addConstr(
                    T[si, l, v]
                    >= inst.a[i] * gp.quicksum(u[i, lp, v] for lp in range(l + 1)),
                    name=f"tw_{i}_{l}_{v}",
                )

    # (17) dropoff time, (18) latest dropoff
    for i in range(n):
        sj = inst.drop_station[i]
        off = inst.I[i]
        for v in range(V):
            for l in range(L):
                if l + off >= L:
                    continue
                if sj != 1:                                                   # (17a)
                    m.addConstr(
                        tau[i]
                        >= T[sj - 1, l + off, v]
                        + sigma * x[sj - 1, l + off, v]
                        + t[sj]
                        - BIG_T * (1 - u[i, l, v]),
                        name=f"tau_{i}_{l}_{v}",
                    )
                else:                                                         # (17b)
                    if l + 1 < L:
                        m.addConstr(
                            tau[i] >= T[0, l + 1, v] + t[1] - BIG_T * (1 - u[i, l, v]),
                            name=f"tau1_{i}_{l}_{v}",
                        )
        if pricing:
            m.addConstr(tau[i] <= inst.d[i] * y[i], name=f"latest_{i}")      # (18)
            m.addConstr(tau[i] >= inst.eps[i] * y[i], name=f"earliest_{i}")
        else:
            m.addConstr(tau[i] <= inst.d[i], name=f"latest_{i}")              # (18)
            m.addConstr(tau[i] >= inst.eps[i], name=f"earliest_{i}")          # tightening

    vars_ = {"x": x, "u": u, "w": w, "b": b, "B": B, "T": T, "Q": Q, "tau": tau}
    if pricing:
        vars_["y"] = y
    fams = _vi_families(valid_inequalities, pricing)
    if fams:
        _add_valid_inequalities(m, inst, vars_, fams, pricing)
    m.update()
    return m, vars_


def _add_valid_inequalities(m, inst: Instance, v_: dict, fams, pricing: bool) -> None:
    """Valid inequalities of Molenbruch et al. (2023), Electronic Companion B.

    All are optimality-valid: they cut only plans with a useless stop or an empty
    lap (never needed, since waiting is free) or, for B2, vehicle relabelings.
    B5 and B3/B4 rely on every performed lap having a passenger-station stop,
    i.e. on B10/B11, and on the battery: B_l >= 0 forces at least l * beta_d/beta_c
    of charging time before lap l. Constraint names start with the family name.
    """
    S, L, V, n = inst.S, inst.L, inst.V, inst.n
    t, sigma = inst.t, inst.sigma
    x, u, w, b, T = v_["x"], v_["u"], v_["w"], v_["b"], v_["T"]
    lap_time = inst.theta + sigma + inst.beta_d / inst.beta_c

    # B1: y[i,v] = sum_l u[i,l,v] (the pricing model already has y[i] = sum_l u[i,l,0])
    if pricing:
        y = v_["y"]

        def yv(i, v):
            return y[i]
    elif fams & {"B1", "B2", "B9"}:
        yc = m.addVars(n, V, vtype=GRB.BINARY, name="y")
        v_["y"] = yc
        for i in range(n):
            for v in range(V):
                m.addConstr(
                    yc[i, v] == gp.quicksum(u[i, l, v] for l in range(L)), name=f"B1_{i}_{v}"
                )

        def yv(i, v):
            return yc[i, v]

    # B2: request i < V may only be served by vehicles 0..i (compact model only)
    if "B2" in fams:
        for i in range(min(V, n)):
            m.addConstr(gp.quicksum(yv(i, vp) for vp in range(i + 1)) == 1, name=f"B2_{i}")

    # B3 / B4: latest pickup lap from the deadline, as variable bounds
    for i in range(n):
        li = latest_drop_lap(inst, i)
        if inst.I[i] == 0 and "B3" in fams:
            first_cut = li + 1
        elif inst.I[i] == 1 and "B4" in fams:
            first_cut = li
        else:
            continue
        for v in range(V):
            for l in range(max(first_cut, 0), L):
                u[i, l, v].UB = 0.0

    pick_at = {s: [i for i in range(n) if inst.pickup_station[i] == s] for s in range(S)}
    drop_same = {s: [i for i in range(n) if inst.drop_station[i] == s and inst.I[i] == 0]
                 for s in range(S)}
    drop_cross = {s: [i for i in range(n) if inst.drop_station[i] == s and inst.I[i] == 1]
                  for s in range(S)}
    crossing = [i for i in range(n) if inst.I[i] == 1]

    for v in range(V):
        # B5: T[s,l] >= l * lap_time + sum_{s'=1..s} t[s']
        if "B5" in fams:
            for l in range(L):
                for s in range(S):
                    m.addConstr(
                        T[s, l, v] >= l * lap_time + sum(t[k] for k in range(1, s + 1)),
                        name=f"B5_{s}_{l}_{v}",
                    )
        # B6: T[0,l] >= l * (theta + sigma) + sum_{l'<l} b[l']. The companion writes
        # beta_c * sum b; the physically correct term is the charging TIME sum b.
        # With beta_c = 1 in every instance both agree.
        if "B6" in fams:
            for l in range(1, L):
                m.addConstr(
                    T[0, l, v] >= l * (inst.theta + sigma) + gp.quicksum(b[lp, v] for lp in range(l)),
                    name=f"B6_{l}_{v}",
                )
        # B7 / B8: a stop at station s >= 1 must serve something there
        for s in range(1, S):
            if "B8" in fams:
                m.addConstr(
                    x[s, 0, v] <= gp.quicksum(u[i, 0, v] for i in pick_at[s])
                    + gp.quicksum(u[i, 0, v] for i in drop_same[s]),
                    name=f"B8_{s}_{v}",
                )
            if "B7" in fams:
                for l in range(1, L):
                    m.addConstr(
                        x[s, l, v] <= gp.quicksum(u[i, l, v] for i in pick_at[s])
                        + gp.quicksum(u[i, l, v] for i in drop_same[s])
                        + gp.quicksum(u[i, l - 1, v] for i in drop_cross[s]),
                        name=f"B7_{s}_{l}_{v}",
                    )
            # B9: stops at s at most the served requests touching s
            if "B9" in fams:
                touching = sorted(set(pick_at[s]) | set(drop_same[s]) | set(drop_cross[s]))
                m.addConstr(
                    gp.quicksum(x[s, l, v] for l in range(L))
                    <= gp.quicksum(yv(i, v) for i in touching),
                    name=f"B9_{s}_{v}",
                )
        # B10 / B11: a performed lap serves something
        if "B11" in fams:
            m.addConstr(w[0, v] <= gp.quicksum(u[i, 0, v] for i in range(n)), name=f"B11_{v}")
        if "B10" in fams:
            for l in range(1, L):
                m.addConstr(
                    w[l, v] <= gp.quicksum(u[i, l, v] for i in range(n))
                    + gp.quicksum(u[i, l - 1, v] for i in crossing),
                    name=f"B10_{l}_{v}",
                )
        # B12: a performed lap stops somewhere (depot included)
        if "B12" in fams:
            for l in range(L):
                m.addConstr(
                    w[l, v] <= gp.quicksum(x[s, l, v] for s in range(S)), name=f"B12_{l}_{v}"
                )


def solve_compact(
    inst: Instance, time_limit: float = 1800, log: bool = True, valid_inequalities=False
):
    m, _ = build_compact(inst, valid_inequalities=valid_inequalities)
    m.Params.TimeLimit = time_limit
    m.Params.OutputFlag = 1 if log else 0

    bound_log: list[tuple[float, float, float]] = []

    def cb(model, where):
        if where == GRB.Callback.MIP:
            rt = model.cbGet(GRB.Callback.RUNTIME)
            lb = model.cbGet(GRB.Callback.MIP_OBJBND)
            ub = model.cbGet(GRB.Callback.MIP_OBJBST)
            if not bound_log or (lb, ub) != bound_log[-1][1:]:
                bound_log.append((rt, lb, ub))

    m.optimize(cb)

    result = {
        "instance": inst.name,
        "method": "compact",
        "status": "optimal" if m.Status == GRB.OPTIMAL else "timeout",
        "ub": m.ObjVal if m.SolCount > 0 else None,
        "lb": m.ObjBound,
        "gap": m.MIPGap if m.SolCount > 0 else None,
        "nodes": m.NodeCount,
        "time_total": m.Runtime,
        "bound_log": bound_log,
    }
    return m, result


# --------------------------------------------------------------------------
# a tiny synthetic instance, so the pipeline can be tested before the real
# files are downloaded from autolab.sites.tau.ac.il/publications
# --------------------------------------------------------------------------


def toy_instance(n: int = 6, seed: int = 0) -> Instance:
    import random

    rng = random.Random(seed)
    S = 6
    t = [4.0] * S                      # theta = 24, as in the artificial set
    pickup, drop, a, q = [], [], [], []
    for _ in range(n):
        p = rng.randint(1, S - 1)
        dd = rng.choice([s for s in range(1, S) if s != p])
        pickup.append(p)
        drop.append(dd)
        a.append(float(rng.randint(30, 80)))
        q.append(1)

    return Instance(
        name="toy",
        n=n,
        C=6,
        beta_d=4.0,
        H_file=0.0,
        pickup_station=pickup,
        drop_station=drop,
        a=a,
        q=q,
        S=S,
        sigma=1.0,
        t=t,
        V=2,
        beta_c=1.0,
        beta_m=11.0,
    )


if __name__ == "__main__":
    inst = toy_instance()
    print(inst.summary())
    m, res = solve_compact(inst, time_limit=120)
    print(res["status"], res["ub"], res["lb"])
