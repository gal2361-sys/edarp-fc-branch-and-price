"""Baseline scorer: the pair with f_ij closest to 0.5 (ties: smallest (i, j))."""

from __future__ import annotations

from bp.scorers import Candidate


class MostFractional:
    name = "most_fractional"

    def choose(self, candidates: list[Candidate], context: dict) -> Candidate:
        return min(candidates, key=lambda c: (abs(c.f - 0.5), c.i, c.j))
