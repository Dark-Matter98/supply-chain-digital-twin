"""Keyed random draws.

Every random quantity in the twin comes from a generator seeded by a key that
names WHAT is drawn (kind, day, entity, ...), not by the order in which the
simulation happens to ask for it. Two runs that differ only in the plan see
the same traffic, supplier delays and handling times for the same physical
event, which gives common random numbers across policies for free.

Keys are integers only. Python's hash() of a string is salted per process,
so it must never be used to build a key.
"""
from __future__ import annotations

import enum
import math

import numpy as np


class Kind(enum.IntEnum):
    NETWORK = 1
    TRAVEL = 2
    SUPPLIER_LATE = 3
    UNLOAD = 4
    RECEIVING = 5
    BACKGROUND = 6
    CORRIDOR = 7     # corridor congestion path (Phase 3, drifting regime)
    INCIDENT = 8     # traffic incidents on a corridor
    PING = 9         # GPS ping noise
    PREDICT = 10     # the predictor's own Monte Carlo scenarios
    NETWORK_EXTRA = 11  # network draws added after Phase 1 (kept apart so old networks don't move)


def keyed_rng(seed: int, kind: Kind, *key: int) -> np.random.Generator:
    parts = [int(seed), int(kind), *(int(k) for k in key)]
    if any(p < 0 for p in parts):
        raise ValueError(f"keys must be non-negative integers, got {parts}")
    return np.random.default_rng(np.random.SeedSequence(parts))


def lognormal_mean_cv(rng: np.random.Generator, mean: float, cv: float) -> float:
    """One lognormal draw with the given mean and coefficient of variation."""
    if cv <= 0:
        return float(mean)
    s2 = math.log1p(cv * cv)
    return float(rng.lognormal(math.log(mean) - s2 / 2, math.sqrt(s2)))
