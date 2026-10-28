"""Stress tests (src/inbound_twin/stress.py): each disruption on the same 20 held-out days,
run under the fixed plan and under the Shortage Early Warning (det + always, SPEC 8.2).

For each scenario it reports the disruption day's line-down minutes and total cost under both,
paired by seed, and exports the median day (by fixed-plan cost) as a replay for the control
tower. A "baseline" row runs the same days with no disruption.

    .venv/Scripts/python scripts/stress_test.py
"""
from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from inbound_twin import Twin
from inbound_twin.costs import daily_costs
from inbound_twin.evaluate import _setup, bootstrap_ci
from inbound_twin.kpi import daily_kpis
from inbound_twin.policy import Controller, PolicySpec
from inbound_twin.stress import SCENARIOS, STRESS_DAY, extra

ROOT = Path(__file__).resolve().parents[1]
REGIME = "R1"                      # normal traffic, so the effect is the disruption's alone
SEEDS = range(101, 121)
POLICIES = {"fixed": PolicySpec("never"), "warning": PolicySpec("det", "always", 0.0)}


def one_day(job) -> dict:
    scenario, seed, pol = job
    cfg, net, plan = _setup(REGIME, None, extra(scenario) if scenario != "baseline" else ())
    spec = POLICIES[pol]
    ctl = Controller(cfg, net, plan, seed, spec) if spec.trigger != "never" else None
    tw = Twin(cfg, net, plan, seed=seed, days=STRESS_DAY + 1, controller=ctl).run()
    k = daily_kpis(tw).set_index("day").loc[STRESS_DAY]
    c = daily_costs(tw, daily_kpis(tw)).set_index("day").loc[STRESS_DAY]
    return {"scenario": scenario, "seed": seed, "policy": pol, "line_down_min": float(k["line_down_min"]),
            "vehicles": int(tw.vehicles_by_day[STRESS_DAY]), "total_usd": float(c["total"]),
            "action_usd": float(c["premium"] + c["disruption"])}


def summarize(df: pd.DataFrame) -> list[dict]:
    rows = []
    for name in ["baseline", *SCENARIOS]:
        d = df[df.scenario == name]
        f = d[d.policy == "fixed"].set_index("seed").sort_index()
        w = d[d.policy == "warning"].set_index("seed").sort_index()
        diff = (w.total_usd - f.total_usd).values
        lo, hi = bootstrap_ci(diff)
        rows.append({
            "scenario": name, "label": SCENARIOS[name]["label"] if name in SCENARIOS else "No disruption",
            "days": len(f),
            "fixed_stop_days": int((f.line_down_min > 0).sum()), "warning_stop_days": int((w.line_down_min > 0).sum()),
            "fixed_line_down_min": round(float(f.line_down_min.mean()), 1),
            "warning_line_down_min": round(float(w.line_down_min.mean()), 1),
            "fixed_total_usd": round(float(f.total_usd.mean())), "warning_total_usd": round(float(w.total_usd.mean())),
            "warning_action_usd": round(float(w.action_usd.mean())),
            "saving_usd": round(float(-diff.mean())), "saving_ci": [round(-hi), round(-lo)],
            "saving_pct": round(float(-100 * diff.mean() / f.total_usd.mean()), 1) if f.total_usd.mean() else 0.0,
        })
    return rows


def main():
    jobs = [(s, seed, p) for s in ["baseline", *SCENARIOS] for seed in SEEDS for p in POLICIES]
    with Pool(max(1, (os.cpu_count() or 2) - 2)) as pool:
        df = pd.DataFrame(pool.map(one_day, jobs, chunksize=1))
    out = ROOT / "outputs" / "stress"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "days.csv", index=False, float_format="%.2f")
    rows = summarize(df)
    pd.DataFrame(rows).drop(columns="saving_ci").to_csv(out / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(rows).to_string(index=False))

    # the median day of each scenario (by fixed-plan cost), exported as a replay
    import export_replay
    picks = {}
    for name in SCENARIOS:
        f = df[(df.scenario == name) & (df.policy == "fixed")].sort_values(["total_usd", "seed"])
        seed = int(f.iloc[len(f) // 2].seed)
        picks[name] = seed
        export_replay.export(REGIME, seed, STRESS_DAY, name)
    manifest = {"regime": REGIME, "day": STRESS_DAY, "seeds": f"{SEEDS.start}-{SEEDS.stop - 1}",
                "scenarios": [{**r, "detail": SCENARIOS[r["scenario"]]["detail"] if r["scenario"] in SCENARIOS else "",
                               "replay_seed": picks.get(r["scenario"])} for r in rows]}
    (ROOT / "showcase" / "web" / "data" / "stress_summary.json").write_text(json.dumps(manifest, indent=1))
    print("Replays:", {k: f"seed {v}" for k, v in picks.items()})


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
