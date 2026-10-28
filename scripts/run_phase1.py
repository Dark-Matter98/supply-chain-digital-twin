"""Run the Phase 1 twin (naive fixed plan) and write KPIs, costs, events and figures.

    .venv/Scripts/python scripts/run_phase1.py --days 30 --seed 1
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from inbound_twin import Twin, generate_network, load_config, naive_plan
from inbound_twin.costs import daily_costs
from inbound_twin.kpi import daily_kpis, littles_law
from inbound_twin.plots import fig_dock, fig_gantt, fig_inventory, worst_day

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "base.yaml"))
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "phase1"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    net = generate_network(cfg)
    plan = naive_plan(net, cfg)
    tw = Twin(cfg, net, plan, seed=args.seed, days=args.days).run()

    kpis = daily_kpis(tw)
    costs = daily_costs(tw, kpis)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kpis.to_csv(out / "kpis.csv", index=False, float_format="%.4f")
    costs.to_csv(out / "costs.csv", index=False, float_format="%.2f")
    pd.DataFrame(tw.log, columns=["t_min", "event", "entity", "value", "info"]).to_csv(
        out / "events.csv", index=False, float_format="%.3f")

    day = worst_day(tw, kpis)
    fig_inventory(tw, day, out / "fig1_inventory.png")
    fig_gantt(tw, day, out / "fig2_truck_gantt.png")
    fig_dock(tw, day, out / "fig3_dock.png")

    digest = hashlib.sha256((kpis.round(6).to_csv(index=False)
                             + costs.round(4).to_csv(index=False)).encode()).hexdigest()[:16]
    pd.set_option("display.width", 120)
    print(f"{args.days} days, seed {args.seed}, KPIs exclude {cfg.warmup_days} warm-up day(s)")
    print("\nMean per day:")
    print(kpis.drop(columns="day").mean().round(3).to_string())
    print("\nMean cost per day (USD, illustrative rates):")
    print(costs.drop(columns="day").mean().round(0).to_string())
    ll = littles_law(tw)
    print(f"\nLittle's law, plant segment: L = {ll['L']:.3f}, lambda*W = {ll['lambda_W']:.3f} containers")
    print(f"Figures use day {day} (most line-down minutes). Output: {out}")
    print(f"KPI hash: {digest}")


if __name__ == "__main__":
    main()
