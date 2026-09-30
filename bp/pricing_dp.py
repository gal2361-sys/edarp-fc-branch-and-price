"""Labeling (DP) pricing prototype. Exact; not yet wired into column generation.

A label sits at a position (lap l, station s) and is a tuple
    T        time after finishing at this position
    battery  battery level (after this lap's discharge once the lap has started)
    load     passengers on board
    onboard  bitmask of requests on board (bit i = request i)
    served   bitmask of requests picked up so far
    rc       reduced cost so far: 0.85 per lap + 0.15 * excess of dropped
             requests - pi of picked requests (mu is subtracted on completion)
    parent, decision  for plan reconstruction
    active   whether anything happened on the current lap

Extension follows bp.column.evaluate() exactly:
  depot of lap l: a label with empty onboard is a complete plan of l laps. To run
    lap l, charge (only with an empty vehicle; see charging below), the battery
    after charging must be >= beta_d, then battery -= beta_d and rc += 0.85.
  station s >= 1: T += t[s]; forced dropoffs of onboard requests whose dropoff
    station is s (tau = arrival; prune if tau > d_i); then every subset of the
    candidate pickups at s (unserved, pi_i > 0, capacity; a depot-crossing request
    cannot start on lap L-1); service start = max(arrival, a_i of the chosen);
    a stop (any pickup or dropoff) adds sigma.

Exact reductions (each behind a flag, so it can be switched off):
  - empty laps: a lap in which nothing happened is discarded at its end (waiting
    is free; the argument behind valid inequality B.10);
  - deadline reachability: a candidate or onboard request whose deadline cannot be
    met even by direct travel is not picked / the label is dropped;
  - charging (`jit_charging`): charge to exactly k * beta_d, k = 1..floor(beta_m /
    beta_d) (0 if already there). Exchange argument: moving charge to a later
    empty-vehicle depot stop never delays a passenger, so some optimal plan charges
    exactly what it needs until the next charging opportunity, a whole number of
    laps. With jit_charging=False: every integer charge 0..beta_m - battery.
  - dominance, at the same position:
      "none";
      "same_sets": identical onboard and served; T <=, battery >=, rc <=;
      "subset": identical onboard (buckets), served_A a subset of served_B,
        T_A <= T_B, battery_A >= battery_B, rc_A <= rc_B, and
        (eff_active_A or not eff_active_B) with eff_active = active or onboard != 0.
        Any completion of B is then feasible for A at no higher cost: A has picked
        a subset, carries the same passengers, is earlier, has more battery. The
        effective-active condition guarantees A's current lap is not discarded as
        empty when B's is kept (onboard passengers must be dropped on this lap, so
        a carrying label's lap always ends active).
  - completion bound (`prune_bound`): gain = sum of pi over requests still
    pickable at or after this position and not yet served; drop the label if
    rc - mu - gain >= -RC_TOL. Future laps and excess only add cost, so no
    completion can reach a negative reduced cost. With the bound on, the DP's
    minimum is exact only when it is negative (pruned labels may have held a
    non-negative minimum).
"""

from __future__ import annotations

import itertools
import resource
import sys
import time

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field

from bp.column import Column, evaluate, lap_plans_from_pickups
from bp.pricing import RC_TOL
from edarp import ALPHA1, ALPHA2, Instance, latest_drop_lap

# label tuple indices
T_, BAT, LOAD, ONB, SRV, RC, PAR, DEC, ACT = range(9)
DOMINANCE_RULES = ("none", "same_sets", "subset")


class LabelLimitExceeded(RuntimeError):
    """Raised when more than max_labels labels are created. Carries the position
    (lap, station) where it happened, the per-position counts and the time."""

    def __init__(self, msg: str, pos=None, counts=None, total=0, elapsed=0.0):
        super().__init__(msg)
        self.pos, self.counts, self.total, self.elapsed = pos, counts or {}, total, elapsed


@dataclass
class DPResult:
    best_rc: float                     # minimum reduced cost over the plans kept
    best: Column | None                # best negative column, None if there is none
    columns: list[Column]              # every complete plan with rc < -RC_TOL, best first
    reduced_costs: list[float]
    labels_total: int
    labels_by_pos: dict = field(default_factory=dict)   # (lap, station) -> created
    max_group: int = 0                 # largest dominance bucket compared
    pruned_by_bound: int = 0
    time: float = 0.0


def bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


class PricingDP:
    def __init__(self, inst: Instance, max_labels: int = 5_000_000,
                 dominance: str = "subset", jit_charging: bool = False,
                 prune_bound: bool = True, check: bool = True,
                 verbose: bool = False, max_rss_gb: float | None = None) -> None:
        if dominance not in DOMINANCE_RULES:
            raise ValueError(f"unknown dominance rule {dominance!r}")
        for v in list(inst.t) + [inst.sigma, inst.beta_d, inst.beta_m, inst.beta_c]:
            if float(v) != int(v):
                raise ValueError("PricingDP assumes integer times and battery data")
        self.inst = inst
        self.max_labels = max_labels
        self.dominance = dominance
        self.jit_charging = jit_charging
        self.prune_bound = prune_bound
        self.check = check
        self.verbose = verbose
        self.max_rss_gb = max_rss_gb
        S, L, n = inst.S, inst.L, inst.n
        self.travel = [[inst._arc_time(s, r) if s != r else 0.0 for r in range(S)]
                       for s in range(S)]
        self.drop_mask = [sum(1 << i for i in range(n) if inst.drop_station[i] == s)
                          for s in range(S)]
        # last lap on which request i can be picked up (valid-inequality l_i logic)
        self.last_pick = []
        for i in range(n):
            li = latest_drop_lap(inst, i) - inst.I[i]
            if inst.I[i] == 1:
                li = min(li, L - 2)
            self.last_pick.append(min(li, L - 1))
        # pickable[(l, s)]: requests that can still be picked strictly after
        # finishing station s on lap l (s = 0: the lap's stations 1..S-1 onward)
        self.pickable = {}
        for l in range(L + 1):
            for s in range(S):
                m = 0
                for i in range(n):
                    lp = self.last_pick[i]
                    if (inst.pickup_station[i] > s and l <= lp) or l + 1 <= lp:
                        m |= 1 << i
                self.pickable[(l, s)] = m
        # at the depot on arrival for lap l (before charging): lap l onward
        self.pickable_depot = [sum(1 << i for i in range(n) if l <= self.last_pick[i])
                               for l in range(L + 1)]

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _mask_sum_table(values: list[float]):
        """pi_mask_sum via 8-bit chunk lookup tables."""
        n = len(values)
        tables = []
        for c in range(0, n, 8):
            chunk = values[c:c + 8]
            tables.append([sum(chunk[b] for b in range(len(chunk)) if (m >> b) & 1)
                           for m in range(256)])

        def mask_sum(mask: int) -> float:
            s, k = 0.0, 0
            while mask:
                s += tables[k][mask & 255]
                mask >>= 8
                k += 1
            return s
        return mask_sum

    def _dominate(self, labels: list, stats: dict) -> list:
        rule = self.dominance
        if rule == "none" or len(labels) < 2:
            return labels
        groups = defaultdict(list)
        if rule == "same_sets":
            for lab in labels:
                groups[(lab[ONB], lab[SRV])].append(lab)
        else:
            for lab in labels:
                groups[lab[ONB]].append(lab)
        out = []
        for grp in groups.values():
            if len(grp) == 1:
                out.append(grp[0])
                continue
            stats["max_group"] = max(stats["max_group"], len(grp))
            if rule == "same_sets":
                grp.sort(key=lambda x: (x[T_], -x[BAT], x[RC]))
                kept = []
                for lab in grp:
                    if not any(k[T_] <= lab[T_] and k[BAT] >= lab[BAT] and k[RC] <= lab[RC]
                               for k in kept):
                        kept.append(lab)
            else:
                # sort by rc (then T, -battery, popcount of served) so that any
                # dominator of a label precedes it or equals it
                grp.sort(key=lambda x: (x[RC], x[T_], -x[BAT], bin(x[SRV]).count("1")))
                if self.inst.n <= 63:
                    out.extend(self._subset_filter_np(grp))
                    continue
                kept = []
                for lab in grp:
                    T, bat, srv = lab[T_], lab[BAT], lab[SRV]
                    eff_b = lab[ACT] or lab[ONB] != 0
                    dominated = False
                    for k in kept:
                        if (k[T_] <= T and k[BAT] >= bat and (k[SRV] & srv) == k[SRV]
                                and (k[ACT] or k[ONB] != 0 or not eff_b)):
                            dominated = True
                            break
                    if not dominated:
                        kept.append(lab)
            out.extend(kept)
        return out

    @staticmethod
    def _subset_filter_np(grp: list) -> list:
        """Same rule as the pure-Python "subset" loop, vectorized over the kept
        labels (served masks as int64; requires n <= 63). grp is sorted by rc."""
        k = len(grp)
        Ts = np.empty(k)
        Bs = np.empty(k)
        Ss = np.empty(k, dtype=np.int64)
        Es = np.empty(k, dtype=bool)
        kept = []
        m = 0
        for lab in grp:
            T, bat, srv = lab[T_], lab[BAT], lab[SRV]
            eff_b = bool(lab[ACT] or lab[ONB] != 0)
            if m:
                sm = Ss[:m]
                dom = (Ts[:m] <= T) & (Bs[:m] >= bat) & ((sm & srv) == sm)
                if eff_b:
                    dom &= Es[:m]
                if dom.any():
                    continue
            Ts[m], Bs[m], Ss[m], Es[m] = T, bat, srv, eff_b
            m += 1
            kept.append(lab)
        return kept

    @staticmethod
    def _plan(label) -> tuple:
        decisions = []
        while label is not None:
            if label[DEC] is not None:
                decisions.append(label[DEC])
            label = label[PAR]
        laps: list[list] = []
        for dec in reversed(decisions):
            if dec[0] == "lap":
                laps.append([set(), dec[1]])
            else:
                laps[-1][0] |= set(bits(dec[1]))
        return tuple((frozenset(p), float(c)) for p, c in laps)

    # ------------------------------------------------------------------ solve

    def solve(self, pi: list[float], mu: float, together=(), apart=()) -> DPResult:
        if together or apart:
            raise NotImplementedError("branching decisions are not supported in the prototype")
        t0 = time.perf_counter()
        inst = self.inst
        S, L, n = inst.S, inst.L, inst.n
        t, sigma, C = inst.t, inst.sigma, inst.C
        bc, bd, bm = inst.beta_c, inst.beta_d, inst.beta_m
        pos_pi = [max(p, 0.0) for p in pi]
        pi_mask_sum = self._mask_sum_table(pos_pi)
        cand_at = defaultdict(list)
        for i in range(n):
            if pi[i] > 0:
                cand_at[inst.pickup_station[i]].append(i)
        excess_coef = [ALPHA2 * inst.q[i] for i in range(n)]

        counts: dict = defaultdict(int)
        stats = {"max_group": 0, "pruned": 0}
        total = 0
        complete: list = []

        def bump(pos, k):
            nonlocal total
            counts[pos] += k
            total += k
            if self.max_rss_gb is not None:
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                rss_gb = rss / 1e9 if sys.platform == "darwin" else rss / 1e6
                if rss_gb > self.max_rss_gb:
                    el = time.perf_counter() - t0
                    raise LabelLimitExceeded(
                        f"memory limit {self.max_rss_gb} GB exceeded ({rss_gb:.1f} GB) at lap "
                        f"{pos[0]}, station {pos[1]} with {total:,} labels after {el:.1f}s",
                        pos=pos, counts=dict(counts), total=total, elapsed=el,
                    )
            if total > self.max_labels:
                el = time.perf_counter() - t0
                raise LabelLimitExceeded(
                    f"label limit {self.max_labels:,} exceeded at lap {pos[0]}, station "
                    f"{pos[1]} after {el:.1f}s",
                    pos=pos, counts=dict(counts), total=total, elapsed=el,
                )

        def bound_ok(lab, pickable_mask) -> bool:
            if not self.prune_bound:
                return True
            gain = pi_mask_sum(pickable_mask & ~lab[SRV])
            if lab[RC] - mu - gain >= -RC_TOL:
                stats["pruned"] += 1
                return False
            return True

        depot = [(0.0, 0.0, 0, 0, 0, 0.0, None, None, False)]
        bump((0, 0), 1)
        for l in range(L + 1):
            for lab in depot:
                if not lab[ONB]:
                    complete.append((lab[RC] - mu, lab))
            if l == L:
                break
            if self.prune_bound:
                depot = [lab for lab in depot if bound_ok(lab, self.pickable_depot[l])]
            # start lap l
            new = []
            for lab in depot:
                T, bat, load = lab[T_], lab[BAT], lab[LOAD]
                if load == 0:
                    if self.jit_charging:
                        charges = sorted({max(0.0, (k * bd - bat) / bc)
                                          for k in range(1, int(bm // bd) + 1)})
                    else:
                        charges = range(0, int((bm - bat) / bc) + 1)
                else:
                    charges = (0,)
                for c in charges:
                    b2 = bat + bc * c
                    if b2 < bd or b2 > bm + 1e-9:
                        continue
                    new.append((T + c, b2 - bd, load, lab[ONB], lab[SRV], lab[RC] + ALPHA1,
                                lab, ("lap", c), False))
            bump((l, 0), len(new))
            labels = self._dominate([x for x in new if bound_ok(x, self.pickable[(l, 0)])], stats)

            for s in range(1, S):
                nxt = []
                cands_s = [i for i in cand_at[s] if l <= self.last_pick[i]]
                dmask = self.drop_mask[s]
                pickable_s = self.pickable[(l, s)]
                for lab in labels:
                    T = lab[T_] + t[s]
                    onb, srv, rc, load = lab[ONB], lab[SRV], lab[RC], lab[LOAD]
                    drops = onb & dmask
                    if drops:
                        ok = True
                        for i in bits(drops):
                            if T > inst.d[i] + 1e-9:
                                ok = False
                                break
                            rc += excess_coef[i] * (T - inst.eps[i])
                            load -= inst.q[i]
                        if not ok:
                            continue
                        onb &= ~drops
                    avail = [i for i in cands_s if not (srv >> i) & 1
                             and max(T, inst.a[i]) + sigma + inst.delta[i] <= inst.d[i] + 1e-9]
                    for k in range(len(avail) + 1):
                        for chosen in itertools.combinations(avail, k):
                            q = sum(inst.q[i] for i in chosen)
                            if load + q > C:
                                continue
                            cmask = sum(1 << i for i in chosen)
                            start = max([T] + [inst.a[i] for i in chosen])
                            stop = bool(drops) or bool(chosen)
                            T2 = start + sigma if stop else T
                            onb2 = onb | cmask
                            late = False
                            for i in bits(onb2):
                                if T2 + self.travel[s][inst.drop_station[i]] > inst.d[i] + 1e-9:
                                    late = True
                                    break
                            if late:
                                continue
                            lab2 = (T2, lab[BAT], load + q, onb2, srv | cmask,
                                    rc - sum(pi[i] for i in chosen), lab,
                                    ("pick", cmask) if chosen else None, lab[ACT] or stop)
                            if bound_ok(lab2, pickable_s):
                                nxt.append(lab2)
                bump((l, s), len(nxt))
                td = time.perf_counter()
                labels = self._dominate(nxt, stats)
                if self.verbose:
                    print(f"  lap {l} st {s}: created {len(nxt):>9,} kept {len(labels):>9,} "
                          f"max bucket {stats['max_group']:>7,} dominance {time.perf_counter() - td:6.1f}s "
                          f"elapsed {time.perf_counter() - t0:7.1f}s", flush=True)

            depot = [(lab[T_] + t[0],) + lab[1:8] + (False,) for lab in labels if lab[ACT]]
            depot = self._dominate(depot, stats)

        # ---- results
        complete.sort(key=lambda z: z[0])
        best_rc = complete[0][0] if complete else float("inf")
        cols, rcs = [], []
        for rc, lab in complete:
            if rc >= -RC_TOL:
                break
            cols.append(self._to_column(lab, rc, pi, mu))
            rcs.append(rc)
        return DPResult(best_rc=best_rc, best=cols[0] if cols else None, columns=cols,
                        reduced_costs=rcs, labels_total=total, labels_by_pos=dict(counts),
                        max_group=stats["max_group"], pruned_by_bound=stats["pruned"],
                        time=time.perf_counter() - t0)

    def _to_column(self, lab, rc, pi, mu) -> Column:
        plan = self._plan(lab)
        col = evaluate(self.inst, lap_plans_from_pickups(self.inst, plan))
        if self.check:
            assert col is not None, f"DP plan infeasible under evaluate(): {plan}"
            dp_cost = rc + mu + sum(pi[i] for i in col.served)
            assert abs(col.cost - dp_cost) < 1e-6, (col.cost, dp_cost, plan)
        return col
