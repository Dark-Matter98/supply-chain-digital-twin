"""Phase 3 stop gates, run on the tuning seeds (1-5) before any experiment.

Gate 1, estimator: 80 % intervals for the remaining time of legs in progress
cover 75-85 % in the drifting regime (R2), and the estimator scores at least as
well as the prior (no filter, no pings) in both regimes.

Gate 2, predictor: logged every 15 min under the "never re-plan" policy,
P(part starves within 2 h) is calibrated against what the twin then does
(calibration slope 0.7-1.3) and beats climatology on Brier score.

This script reads the twin's TRUTH (true leg ends, actual line stops) to score
the estimates; nothing in the controller path does.

    .venv/Scripts/python scripts/validate_phase3.py --gate 1
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr

from inbound_twin import Twin, generate_network, load_config
from inbound_twin.config import override
from inbound_twin.estimate import Estimator
from inbound_twin.evaluate import load_phase2_plan
from inbound_twin.observe import observe
from inbound_twin.diagnostics import StarveLogger

ROOT = Path(__file__).resolve().parents[1]
REGIMES = {"R1": (), "R2": (ROOT / "config" / "drifting.yaml",)}


class LegScorer:
    """Every check_min: score each leg in progress under the estimator and the prior."""

    def __init__(self, cfg, net):
        self.cfg = cfg
        self.est = Estimator(cfg, net, "bayes")
        self.prior = Estimator(cfg, net, "bayes")   # never ingests, never sees pings
        self.rows = []

    def on_event(self, twin, kind, info):
        pass

    def process(self, twin):
        while True:
            yield twin.env.timeout(self.cfg.dynamic.check_min)
            obs = observe(twin)
            self.est.ingest(obs)
            for tr_obs, tr in zip(obs.trucks, twin.state.trucks):
                if math.isnan(tr_obs.leg_start):
                    continue
                truth = tr.leg_end_true - obs.t                       # TRUTH, scoring only
                post = self.est.leg_posterior(tr_obs, obs.t)
                bare = tr_obs.__class__(**{**tr_obs.__dict__, "ping": None})
                prior = self.prior.leg_posterior(bare, obs.t)
                self.rows.append((covered(post, truth), log_score(post, truth),
                                  covered(prior, truth), log_score(prior, truth)))


def quantile(post, u):
    return float(Estimator.sample_remaining(post, np.array([u]))[0])


def covered(post, truth, level=0.8):
    lo, hi = quantile(post, (1 - level) / 2), quantile(post, 1 - (1 - level) / 2)
    return lo <= truth <= hi


def log_score(post, truth):
    d = truth + post.elapsed
    z = (math.log(d) - post.mu) / post.sd
    z_c = (math.log(post.elapsed) - post.mu) / post.sd
    tail = max(1.0 - float(ndtr(z_c)), 1e-300)
    return -0.5 * z * z - math.log(post.sd * d * math.sqrt(2 * math.pi)) - math.log(tail)


def gate1(seeds, days):
    ok = True
    for name, overlays in REGIMES.items():
        cfg = load_config(ROOT / "config" / "base.yaml", *overlays)
        net = generate_network(cfg)
        plan = load_phase2_plan(cfg, net)
        rows = []
        for seed in seeds:
            scorer = LegScorer(cfg, net)
            Twin(cfg, net, plan, seed=seed, days=days, controller=scorer).run()
            rows += scorer.rows
        r = np.array(rows, dtype=float)
        cov, ls, cov0, ls0 = r[:, 0].mean(), r[:, 1].mean(), r[:, 2].mean(), r[:, 3].mean()
        print(f"{name}: {len(r)} leg checks | 80% interval coverage {cov:.3f} (prior {cov0:.3f}) | "
              f"mean log score {ls:.3f} (prior {ls0:.3f})")
        if name == "R2" and not 0.75 <= cov <= 0.85:
            print("   FAIL: R2 coverage outside 0.75-0.85")
            ok = False
        if ls < ls0 - 1e-9:
            print("   FAIL: estimator scores worse than the prior")
            ok = False
    print("GATE 1:", "PASS" if ok else "FAIL")
    return ok


def logistic_slope(p, y):
    """Maximum-likelihood b in y ~ sigmoid(a + b * logit(p)); b = 1 means the spread is right."""
    pc = np.clip(p, 1e-4, 1 - 1e-4)
    x = np.log(pc / (1 - pc))

    def nll(beta):
        z = np.clip(beta[0] + beta[1] * x, -30, 30)
        return float(np.sum(np.logaddexp(0, z) - y * z))

    return float(minimize(nll, np.array([0.0, 1.0]), method="Nelder-Mead",
                          options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000}).x[1])


def report(label, p, y, bins):
    brier, clim = float(np.mean((p - y) ** 2)), float(y.mean() * (1 - y.mean()))
    slope = logistic_slope(p, y)
    print(f"  {label}: n={len(p)}, events {int(y.sum())} | Brier {brier:.5f} vs climatology {clim:.5f} | "
          f"calibration slope {slope:.2f}")
    for lo, hi in bins:
        sel = (p >= lo) & (p < hi)
        if sel.any():
            print(f"     predicted {lo:.2f}-{min(hi, 1):.2f}: n={int(sel.sum()):6d}  mean predicted "
                  f"{p[sel].mean():.3f}  observed {y[sel].mean():.3f}")
    return slope, brier, clim


def gate2(seeds, days, trigger_h=None):
    """Scored at the line level. Per part is reported too, but a part's outcome is censored:
    the twin records only the parts short when the line stops, and a stopped line stops
    consuming, so a part about to run out behind another one is never 'starved'."""
    ok = True
    bins = [(0, 0.01), (0.01, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.01)]
    for name, overlays in REGIMES.items():
        cfg = load_config(ROOT / "config" / "base.yaml", *overlays)
        if trigger_h:
            cfg = override(cfg, {"dynamic.trigger_h": trigger_h})
        horizon = cfg.dynamic.trigger_h * 60
        net = generate_network(cfg)
        plan = load_phase2_plan(cfg, net)
        lp, ly, pp, py = [], [], [], []
        for seed in seeds:
            lg = StarveLogger(cfg, net, plan, seed)
            tw = Twin(cfg, net, plan, seed=seed, days=days, controller=lg).run()
            starts = [(a, set(parts)) for a, _, parts in tw.downtime]           # TRUTH, scoring only
            for t, p in lg.line_rows:
                lp.append(p)
                ly.append(any(t < a <= t + horizon for a, _ in starts))
            for t, part, p in lg.part_rows:
                pp.append(p)
                py.append(any(t < a <= t + horizon and part in parts for a, parts in starts))
        print(f"{name}: P(line stops within {horizon / 60:.0f} h), checked every {cfg.dynamic.check_min:.0f} min")
        slope, brier, clim = report("line level (gate)", np.array(lp), np.array(ly, dtype=float), bins)
        report("per part (censored, for reference)", np.array(pp), np.array(py, dtype=float), bins)
        if not (0.7 <= slope <= 1.3 and brier < clim):
            print("   FAIL")
            ok = False
    print("GATE 2:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gate", type=int, choices=(1, 2), required=True)
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--trigger-h", type=float, default=None,
                    help="gate 2 only: score P(line stops within this many hours) instead of the config's 2 h")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    if args.gate == 1:
        gate1(seeds, args.days)
    else:
        gate2(seeds, args.days, args.trigger_h)


if __name__ == "__main__":
    main()
