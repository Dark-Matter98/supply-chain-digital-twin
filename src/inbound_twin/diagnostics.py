"""Loggers that ride along a twin run to score the estimator and predictor
against the truth. Only diagnostics read the truth; controllers never do."""
from __future__ import annotations

import math

import numpy as np

from .estimate import Estimator
from .observe import observe
from .predict import Candidate, Predictor


class StarveLogger:
    """Every check_min (line not already stopped): P(line stops within trigger_h),
    and each part's P(short within trigger_h)."""

    def __init__(self, cfg, net, plan, seed):
        self.cfg = cfg
        self.est = Estimator(cfg, net, "bayes")
        self.pred = Predictor(cfg, net, plan, seed)
        self.line_rows: list[tuple[float, float]] = []
        self.part_rows: list[tuple[float, int, float]] = []

    def on_event(self, twin, kind, info):
        pass

    def process(self, twin):
        while True:
            yield twin.env.timeout(self.cfg.dynamic.check_min)
            obs = observe(twin)
            self.est.ingest(obs)
            if obs.line_status == "starved":
                continue
            f = self.pred.forecast(obs, self.est, [Candidate()])[0]
            self.line_rows.append((obs.t, f.p_line))
            self.part_rows += [(obs.t, p, float(f.p_starve[p])) for p in range(len(f.p_starve))]


def line_outcomes(tw, rows, horizon_min: float) -> tuple[np.ndarray, np.ndarray]:
    """(predicted P, did the line stop within the horizon) for a StarveLogger's line rows."""
    starts = [a for a, _, _ in tw.downtime]                      # TRUTH, scoring only
    p = np.array([r[1] for r in rows])
    y = np.array([any(t < a <= t + horizon_min for a in starts) for t, _ in rows], dtype=float)
    return p, y


class CorridorTrace:
    """Filter mean and 80 % band for one corridor every check_min, with the true offset."""

    def __init__(self, cfg, net, supplier: int):
        self.cfg, self.supplier = cfg, supplier
        self.est = Estimator(cfg, net, "bayes")
        self.rows: list[tuple[float, float, float, float]] = []   # (t, mean, sd, truth)

    def on_event(self, twin, kind, info):
        pass

    def process(self, twin):
        k = self.supplier - 1
        while True:
            yield twin.env.timeout(self.cfg.dynamic.check_min)
            obs = observe(twin)
            self.est.ingest(obs)
            m, P = self.est.corridor_now(obs)
            truth = twin.scen.corridor_offset(self.supplier, obs.t)      # TRUTH, plotting only
            self.rows.append((obs.t, float(m[k]), math.sqrt(max(P[k, k], 0.0)), truth))
