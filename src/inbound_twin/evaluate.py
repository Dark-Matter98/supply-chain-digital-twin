"""Evaluate plans in the twin and compare them with paired bootstrap intervals.

Every plan runs on the same seeds. The twin's random draws are keyed by the
physical event (rng.py), so a seed gives every plan the same traffic and
supplier delays, and the per-seed differences between plans are paired.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .costs import daily_costs
from .kpi import daily_kpis
from .network import Network
from .plan import Plan, Route
from .sim import Twin

LOGISTICS = ["transport", "driver", "trucks", "handling", "holding_capital", "holding_space"]
PHASE2_PLANS = Path(__file__).resolve().parents[2] / "outputs" / "phase2" / "plans.csv"


def load_phase2_plan(cfg: Config, net: Network, name: str = "optimized") -> Plan:
    """The Phase 2 plan of that name from outputs/phase2/plans.csv, rebuilt if the file is missing."""
    if not PHASE2_PLANS.exists():
        from .optimize import optimize_plan
        return optimize_plan(net, cfg).plan
    df = pd.read_csv(PHASE2_PLANS)
    rows = df[df.plan == name]
    if rows.empty:
        raise ValueError(f"no plan named {name!r} in {PHASE2_PLANS}")
    return Plan(tuple(Route(int(r.truck), float(r.dispatch_h) * 60.0,
                            tuple(int(x) for x in str(r.suppliers).split("-")), float(r.lead_planned_min))
                      for r in rows.itertuples()), name)


def measured_lead_min(tw) -> float:
    """Mean minutes from dispatch until the load is in the supermarket, after warm-up."""
    leads = []
    for t in tw.trips:
        if t.day < tw.cfg.warmup_days:
            continue
        for truck, t_arrive, t_received, _ in tw.batches:
            if truck == t.truck and t.t_dispatch <= t_arrive <= t.t_end:
                leads.append(t_received - t.t_dispatch)
    return float(np.mean(leads)) if leads else float("nan")


def run_plan(cfg: Config, net: Network, plan: Plan, seeds, days: int) -> pd.DataFrame:
    """One row per seed: mean daily KPIs and costs over the days after warm-up."""
    rows = []
    for seed in seeds:
        tw = Twin(cfg, net, plan, seed=seed, days=days).run()
        k = daily_kpis(tw)
        c = daily_costs(tw, k)
        row = {"plan": plan.name, "seed": seed}
        row.update(c.drop(columns="day").mean().to_dict())
        row["logistics"] = row["total"] - row["line_down"]
        for col in ("line_down_min", "line_stops", "vehicles", "km", "trips", "driver_shifts",
                    "overtime_h", "trucks_used", "truck_fill", "dispatch_late_min", "door_util",
                    "gate_to_supermarket_min"):
            row[col] = float(k[col].mean())
        row["lead_measured_min"] = measured_lead_min(tw)
        row["lead_planned_min"] = float(np.mean([r.lead_min for r in plan.routes]))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_ci(values, n: int = 10_000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval for the mean of per-seed values."""
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def compare(results: pd.DataFrame, baseline: str, rates: tuple[float, ...]) -> pd.DataFrame:
    """Paired per-seed differences (plan - baseline) with bootstrap 95 % intervals.

    Total cost is reported at several line-down rates, because the rate is a
    judgment value and decides which plan wins.
    """
    base = results[results.plan == baseline].set_index("seed")
    rows = []
    for name, df in results.groupby("plan", sort=False):
        if name == baseline:
            continue
        d = df.set_index("seed").loc[base.index]
        metrics = {"logistics": d.logistics - base.logistics,
                   "line_down_min": d.line_down_min - base.line_down_min}
        for term in LOGISTICS:
            metrics[term] = d[term] - base[term]
        for r in rates:
            metrics[f"total_at_{int(r)}"] = ((d.logistics + r * d.line_down_min)
                                             - (base.logistics + r * base.line_down_min))
        for metric, diff in metrics.items():
            lo, hi = bootstrap_ci(diff.values)
            rows.append({"plan": name, "metric": metric, "mean_diff": float(diff.mean()),
                         "ci_low": lo, "ci_high": hi, "n_seeds": len(diff)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Phase 3
ROOT = Path(__file__).resolve().parents[2]
REGIME_OVERLAYS = {"R1": (), "R2": (ROOT / "config" / "drifting.yaml",)}
_CACHE: dict = {}


def _setup(regime: str, ss_h: float | None, extra: tuple = ()):
    """Config, network and P2 plan for a regime; extra = ((dotted key, value), ...) overrides."""
    key = (regime, ss_h, extra)
    if key not in _CACHE:
        from .config import load_config, override
        from .network import generate_network
        cfg = load_config(ROOT / "config" / "base.yaml", *REGIME_OVERLAYS[regime])
        if ss_h is not None:
            cfg = override(cfg, {"inventory.ss_h": ss_h})
        if extra:
            cfg = override(cfg, dict(extra))
        net = generate_network(cfg)
        _CACHE[key] = (cfg, net, load_phase2_plan(cfg, net))
    return _CACHE[key]


def run_arm(job: tuple) -> dict:
    """One twin run under one policy: (regime, spec, seed, days, ss_h) -> summary row.

    Top-level and picklable so a process pool can run it."""
    from .policy import Controller
    regime, spec, seed, days, ss_h, *rest = job
    cfg, net, plan = _setup(regime, ss_h, rest[0] if rest else ())
    ctl = Controller(cfg, net, plan, seed, spec)
    tw = Twin(cfg, net, plan, seed=seed, days=days, controller=ctl).run()
    k = daily_kpis(tw)
    c = daily_costs(tw, k)
    w = cfg.warmup_days
    n_days = days - w
    stops = [(a, b) for a, b, _ in tw.downtime if a >= w * 1440]
    fires = np.array(ctl.fires)
    missed = sum(1 for a, _ in stops if not ((fires > a - 60) & (fires <= a)).any())
    row = {"regime": regime, "policy": spec.name, "trigger": spec.trigger, "gate": spec.gate,
           "threshold": spec.threshold, "seed": seed, "ss_h": cfg.inventory.ss_h}
    row.update(c.drop(columns="day").mean().to_dict())
    row["logistics"] = row["total"] - row["line_down"]
    row["line_down_min"] = float(k["line_down_min"].mean())
    row["line_stops"] = float(k["line_stops"].mean())
    after = [d for d in ctl.decisions if d["t"] >= w * 1440]
    acts = [a for a in tw.actions if a["day"] >= w]
    row["assessments_per_day"] = len(after) / n_days
    row["executed_per_day"] = sum(d["executed"] for d in after) / n_days
    row["reroutes_per_day"] = sum(a["kind"] == "reroute" for a in acts) / n_days
    row["couriers_per_day"] = sum(a["kind"] == "courier" for a in acts) / n_days
    row["hot_runs_per_day"] = sum(a["kind"] == "hot_run" for a in acts) / n_days
    row["missed_stops_per_day"] = missed / n_days
    row["predictor_calls"] = ctl.pred.calls
    return row
