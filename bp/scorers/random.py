"""Random scorer with its own seeded generator (a reproducible lower baseline)."""

from __future__ import annotations

import random as _random

from bp.scorers import Candidate


class RandomScorer:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.rng = _random.Random(seed)

    def choose(self, candidates: list[Candidate], context: dict) -> Candidate:
        ordered = sorted(candidates, key=lambda c: (c.i, c.j))
        return ordered[self.rng.randrange(len(ordered))]
