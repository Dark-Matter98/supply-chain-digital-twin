"""Phase 3: dynamic re-planning on top of the P2 plan.

1. Tune: seeds 1-5, per regime. The triggers' thresholds, each gate, chosen by
   mean total cost at $5k per line-down minute.
2. Test: seeds 101-120, per regime, every arm, paired with "never".
3. U-curves: safety stock 0.5-4 h for never / det+value / bayes+value, seeds 101-110.
Then hypotheses H1-H4, figures 7-11, and a hash of the results.

    .venv/Scripts/python scripts/run_phase3.py
    .venv/Scripts/python scripts/run_phase3.py --figures-only                 (re-render from the CSVs)
    .venv/Scripts/python scripts/run_phase3.py --trigger-h 6 --figures-only   (the post-hoc run's figures)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from inbound_twin import Twin
from inbound_twin.diagnostics import CorridorTrace, StarveLogger, line_outcomes
from inbound_twin.evaluate import _setup, bootstrap_ci, run_arm
from inbound_twin.policy import Controller, PolicySpec
from inbound_twin.scenario import corridor_path

ROOT = Path(__file__).resolve().parents[1]
REGIMES = ("R1", "R2")
TUNE_SEEDS = range(1, 6)
TEST_SEEDS = range(101, 121)
CURVE_SEEDS = range(101, 111)
TAUS = (0.05, 0.1, 0.2, 0.4)
MARGINS = (-30, -15, 0, 15, 30, 60)   # negative too: point projections use mean (late-skewed) durations
SS_LEVELS = (0.5, 1.0, 2.0, 3.0, 4.0)
RATE = 5000.0
RATES = (1000.0, 5000.0, 20000.0)


def run_pool(jobs, workers):
    with Pool(workers) as pool:
        return pd.DataFrame(pool.map(run_arm, jobs, chunksize=1))


def total_at(df, rate):
    return df.logistics + rate * df.line_down_min


def tune(days, workers, extra=()):
    grid = [PolicySpec("det", g, m) for g in ("value", "always") for m in MARGINS]
    grid += [PolicySpec("bayes", g, t) for g in ("value", "always") for t in TAUS]
    grid += [PolicySpec("fixed", "value", t) for t in TAUS]
    df = run_pool([(r, s, seed, days, None, extra) for r in REGIMES for s in grid for seed in TUNE_SEEDS], workers)
    df["total_5k"] = total_at(df, RATE)
    return df, pick_best(df)


def pick_best(df):
    """The tuned threshold per (regime, trigger, gate): least mean total cost at RATE."""
    df = df.assign(total_5k=total_at(df, RATE))
    best = {}
    for r in REGIMES:
        means = df[df.regime == r].groupby(["trigger", "gate", "threshold"]).total_5k.mean()
        for (trig, gate), grp in means.groupby(level=[0, 1]):
            thr = grp.idxmin()[2]
            best[(r, trig, gate)] = PolicySpec(trig, gate, float(thr))
    return best


def test_arms(best, regime):
    arms = [PolicySpec("never"), PolicySpec("periodic", "value"), PolicySpec("periodic", "always"),
            PolicySpec("event", "value"), PolicySpec("event", "always")]
    arms += [best[(regime, t, g)] for t in ("det", "bayes") for g in ("value", "always")]
    arms.append(best[(regime, "fixed", "value")])
    return arms


def paired(df, a, b, metric):
    x = df[df.policy == a].set_index("seed")[metric]
    y = df[df.policy == b].set_index("seed")[metric].loc[x.index]
    d = (x - y).values
    lo, hi = bootstrap_ci(d)
    return float(d.mean()), lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=15)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "phase3"))
    ap.add_argument("--quick", action="store_true", help="tiny smoke run of the whole pipeline")
    ap.add_argument("--trigger-h", type=float, default=None,
                    help="override dynamic.trigger_h (post-hoc lookahead run); writes outputs/phase3_lookahead<H>")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="re-render figures 7-11 from the CSVs already in the output folder (no experiment)")
    args = ap.parse_args()
    extra = (("dynamic.trigger_h", args.trigger_h),) if args.trigger_h else ()
    if args.trigger_h:
        args.out = str(ROOT / "outputs" / f"phase3_lookahead{args.trigger_h:g}")
    if args.quick:
        global TUNE_SEEDS, TEST_SEEDS, CURVE_SEEDS, SS_LEVELS
        TUNE_SEEDS, TEST_SEEDS, CURVE_SEEDS, SS_LEVELS = range(1, 3), range(101, 104), range(101, 103), (1.0, 2.0)
        args.days = min(args.days, 4)
        args.out = str(ROOT / "outputs" / "phase3_quick")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        test, curves = pd.read_csv(out / "test.csv"), pd.read_csv(out / "ucurves.csv")
        best = pick_best(pd.read_csv(out / "tuning.csv"))
        missing = [(k, v.name) for k, v in best.items() if v.name not in set(test[test.regime == k[0]].policy)]
        assert not missing, f"tuning.csv does not reproduce the tested thresholds: {missing}"
        figures(out, test, curves, best, args.days, extra)
        print(f"Figures re-rendered in {out}")
        return

    t0 = time.time()
    tuning, best = tune(args.days, args.workers, extra)
    tuning.to_csv(out / "tuning.csv", index=False, float_format="%.4f")
    print(f"Tuning: {len(tuning)} runs in {time.time() - t0:.0f} s")
    for k, v in sorted(best.items()):
        print(f"   {k[0]} {v.name}")

    t0 = time.time()
    jobs = [(r, s, seed, args.days, None, extra) for r in REGIMES for s in test_arms(best, r) for seed in TEST_SEEDS]
    test = run_pool(jobs, args.workers)
    test.to_csv(out / "test.csv", index=False, float_format="%.4f")
    print(f"Test: {len(test)} runs in {time.time() - t0:.0f} s")

    rows = []
    for r in REGIMES:
        df = test[test.regime == r]
        for pol in df.policy.unique():
            if pol == "never":
                continue
            for metric in ("line_down_min", "logistics", "premium", "disruption"):
                rows.append((r, pol, "never", metric, *paired(df, pol, "never", metric)))
            for rate in RATES:
                df2 = df.assign(total=total_at(df, rate))
                rows.append((r, pol, "never", f"total_at_{int(rate)}", *paired(df2, pol, "never", "total")))
    cmp = pd.DataFrame(rows, columns=["regime", "policy", "vs", "metric", "mean_diff", "ci_low", "ci_high"])

    # Hypotheses (stated in the plan before any results)
    hyp = []
    for r in REGIMES:
        df = test[test.regime == r].assign(total=lambda d: total_at(d, RATE))
        bv, dv = best[(r, "bayes", "value")].name, best[(r, "det", "value")].name
        ba = best[(r, "bayes", "always")].name
        m, lo, hi = paired(df, bv, dv, "total")
        hyp.append((r, "bayes+value - det+value, total $/day", m, lo, hi))
        m, lo, hi = paired(df, bv, ba, "total")
        hyp.append((r, "bayes+value - bayes+always, total $/day", m, lo, hi))
        m, lo, hi = paired(df, bv, ba, "executed_per_day")
        hyp.append((r, "bayes+value - bayes+always, executed actions/day", m, lo, hi))
    hyp = pd.DataFrame(hyp, columns=["regime", "comparison", "mean_diff", "ci_low", "ci_high"])

    t0 = time.time()
    curve_specs = {r: [PolicySpec("never"), best[(r, "det", "value")], best[(r, "bayes", "value")]] for r in REGIMES}
    jobs = [(r, s, seed, args.days, ss, extra) for r in REGIMES for s in curve_specs[r] for ss in SS_LEVELS
            for seed in CURVE_SEEDS]
    curves = run_pool(jobs, args.workers)
    curves.to_csv(out / "ucurves.csv", index=False, float_format="%.4f")
    print(f"U-curves: {len(curves)} runs in {time.time() - t0:.0f} s")
    curves["total_5k"] = total_at(curves, RATE)
    cmin = curves.groupby(["regime", "policy", "ss_h"]).total_5k.mean().reset_index()
    for (r, pol), g in cmin.groupby(["regime", "policy"]):
        i = g.total_5k.idxmin()
        hyp.loc[len(hyp)] = (r, f"U-curve minimum, {pol}: ss_h={g.loc[i, 'ss_h']:g}", g.loc[i, "total_5k"],
                             np.nan, np.nan)

    cmp.to_csv(out / "compare.csv", index=False, float_format="%.4f")
    hyp.to_csv(out / "hypotheses.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 160)
    summary = test.assign(total_5k=total_at(test, RATE)).groupby(["regime", "policy"], sort=False)[
        ["total_5k", "logistics", "line_down_min", "premium", "disruption", "assessments_per_day",
         "executed_per_day", "reroutes_per_day", "couriers_per_day", "hot_runs_per_day",
         "missed_stops_per_day"]].mean()
    print("\nTest seeds 101-120, mean per day:")
    print(summary.round(2).to_string())
    print("\nPaired vs never (95% bootstrap CI):")
    print(cmp[cmp.metric.isin(["line_down_min", "total_at_5000"])].round(1).to_string(index=False))
    print("\nHypotheses:")
    print(hyp.round(2).to_string(index=False))

    if not args.no_figures:
        figures(out, test, curves, best, args.days, extra)
    digest = hashlib.sha256((cmp.round(4).to_csv(index=False) + hyp.round(4).to_csv(index=False)
                             + curves.round(4).to_csv(index=False)).encode()).hexdigest()[:16]
    print(f"\nOutput: {out}\nPhase 3 hash: {digest}")


def figures(out, test, curves, best, days, extra=()):
    from inbound_twin.plots import fig_arms, fig_corridor, fig_reliability, fig_trace, fig_ucurves
    note = f" - post hoc, trigger lookahead {dict(extra)['dynamic.trigger_h']:g} h" if extra else ""
    # fig7: a corridor incident on an R2 day, the filter tracking it
    cfg, net, plan = _setup("R2", None, extra)
    pick = None
    for seed in TUNE_SEEDS:
        for day in range(1, 6):
            for s in range(1, len(net.suppliers) + 1):
                path = corridor_path(cfg, seed, day, s)
                if path.max() > 0.5 and pick is None:
                    pick = (seed, day, s)
    if pick:
        seed, day, s = pick
        tr = CorridorTrace(cfg, net, s)
        Twin(cfg, net, plan, seed=seed, days=day + 1, controller=tr).run()
        fig_corridor(tr.rows, day, s, out / "fig7_filter_vs_truth.png")
    # fig8: predictor reliability, both regimes (tuning seeds)
    rel = {}
    for r in REGIMES:
        cfg, net, plan = _setup(r, None, extra)
        ps, ys = [], []
        for seed in TUNE_SEEDS:
            lg = StarveLogger(cfg, net, plan, seed)
            tw = Twin(cfg, net, plan, seed=seed, days=10, controller=lg).run()
            p, y = line_outcomes(tw, lg.line_rows, cfg.dynamic.trigger_h * 60)
            ps.append(p)
            ys.append(y)
        rel[r] = (np.concatenate(ps), np.concatenate(ys))
    fig_reliability(rel, cfg.dynamic.trigger_h, out / "fig8_predictor_reliability.png")
    fig_arms(test, RATE, out / "fig9_arms.png", note)
    fig_ucurves(curves, RATE, out / "fig10_ucurves.png", note)
    # fig11: one R2 test day, never vs the tuned bayes+value policy
    cfg, net, plan = _setup("R2", None, extra)
    spec = best[("R2", "bayes", "value")]
    never = Twin(cfg, net, plan, seed=TEST_SEEDS[0], days=days).run()
    worst = max(range(1, days), key=lambda d: sum(min(b, (d + 1) * 1440) - max(a, d * 1440)
                                                 for a, b, _ in never.downtime if a < (d + 1) * 1440 and b > d * 1440))
    ctl = Controller(cfg, net, plan, TEST_SEEDS[0], spec)
    log_ctl = _TracingController(ctl)
    dyn = Twin(cfg, net, plan, seed=TEST_SEEDS[0], days=days, controller=log_ctl).run()
    fig_trace(never, dyn, log_ctl, worst, spec, cfg.dynamic.trigger_h, out / "fig11_decision_trace.png")


class _TracingController:
    """Wraps a controller and records its P(line stop) at every risk check, for fig11."""

    def __init__(self, inner):
        self.inner = inner
        self.trace: list[tuple[float, float]] = []
        inner_pred = inner.pred
        outer = self

        class _Pred:
            def __getattr__(self, name):
                return getattr(inner_pred, name)

            def forecast(self, obs, est, cands):
                fs = inner_pred.forecast(obs, est, cands)
                if len(cands) == 1:
                    outer.trace.append((obs.t, fs[0].p_line))
                return fs
        inner.pred = _Pred()

    def on_event(self, twin, kind, info):
        self.inner.on_event(twin, kind, info)

    def process(self, twin):
        return self.inner.process(twin)

    @property
    def decisions(self):
        return self.inner.decisions


if __name__ == "__main__":
    main()
