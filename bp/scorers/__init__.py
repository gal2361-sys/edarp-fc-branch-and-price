"""Ryan-Foster pair scorers.

A scorer only chooses among valid candidates: pairs (i, j) with a fractional
f_ij = sum of lambda over allowed columns serving both. It never prunes, fixes
variables or stops column generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Candidate:
    i: int
    j: int
    f: float          # f_ij, strictly between 0 and 1


class Scorer(Protocol):
    name: str

    def choose(self, candidates: list[Candidate], context: dict) -> Candidate:
        """Return one of `candidates`. `context` carries the node, the master, its LP
        solution and the instance, for scorers that compute features."""
        ...


def make_scorer(name: str, seed: int = 0) -> Scorer:
    if name == "most_fractional":
        from bp.scorers.most_fractional import MostFractional
        return MostFractional()
    if name == "random":
        from bp.scorers.random import RandomScorer
        return RandomScorer(seed)
    raise ValueError(f"unknown scorer {name!r}")
