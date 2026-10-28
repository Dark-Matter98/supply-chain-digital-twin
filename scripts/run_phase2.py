"""Phase 2: build the naive, routing-only and fully optimized plans, then run all
three in the twin on the same held-out seeds and compare them.

    .venv/Scripts/python scripts/run_phase2.py --days 30 --seeds 20
"""
from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pandas as pd

from inbound_twin import generate_network, load_config, naive_plan
from inbound_twin.evaluate import LOGISTICS, compare, run_plan
from inbound_twin.optimize import PlanResult, optimize_plan, planning_cost, routing_only_plan
from inbound_twin.plan import wave_index
from inbound_twin.plots import fig_costs, fig_promise, fig_routes

ROOT = Path(__file__).resolve().parents[1]
FIRST_SEED = 101   # held out: nothing in Phase 2 was tuned on seeds >= 101


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config" / "base.yaml"))
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--rates", default="1000,5000,20000", help="line-down $/min for total cost")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "phase2"))
    args = ap.parse_args()
    rates = tuple(float(r) for r in args.rates.split(","))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    net = generate_network(cfg)
    t0 = time.time()
    p0 = naive_plan(net, cfg)
    results = [PlanResult(p0, {}, planning_cost(net, cfg, p0)),
               routing_only_plan(net, cfg), optimize_plan(net, cfg)]
    print(f"Plans built in {time.time() - t0:.0f} s")
    for r in results:
        print(f"\n[{r.plan.name}] fleet {r.plan.fleet}, {len(r.plan.routes)} trips, "
              f"planning cost ${r.planning['total']:,.0f}/day")
        for line in r.log:
            print("   ", line)

    pd.DataFrame([{"plan": r.plan.name, "truck": x.truck, "wave": wave_index(cfg, x.dispatch_min),
                   "dispatch_h": x.dispatch_min / 60, "suppliers": "-".join(map(str, x.suppliers)),
                   "lead_planned_min": round(x.lead_min, 2)}
                  for r in results for x in r.plan.routes]).to_csv(out / "plans.csv", index=False)

    seeds = range(FIRST_SEED, FIRST_SEED + args.seeds)
    t0 = time.time()
    sim = pd.concat([run_plan(cfg, net, r.plan, seeds, args.days) for r in results], ignore_index=True)
    print(f"\nTwin: {len(results)} plans x {args.seeds} seeds x {args.days} days in {time.time() - t0:.0f} s")
    for r in results:
        for key, value in r.planning.items():
            sim.loc[sim.plan == r.plan.name, f"planned_{key}"] = value
    sim.to_csv(out / "daily.csv", index=False, float_format="%.4f")
    cmp = compare(sim, p0.name, rates)
    cmp.to_csv(out / "compare.csv", index=False, float_format="%.4f")

    summary = sim.groupby("plan", sort=False).mean(numeric_only=True)
    pd.set_option("display.width", 140)
    cols = ["planned_total", "logistics", *LOGISTICS, "line_down_min", "line_stops", "km",
            "driver_shifts", "overtime_h", "trucks_used", "truck_fill", "dispatch_late_min",
            "lead_planned_min", "lead_measured_min"]
    print("\nMean per day over seeds (planned_total = the optimizer's promise, no line-down):")
    print(summary[cols].T.round(2).to_string())
    print("\nPaired differences vs naive, 95% bootstrap CI over seeds:")
    print(cmp.round(2).to_string(index=False))

    fig_routes(net, cfg, [r.plan for r in results], wave=1, path=out / "fig4_routes.png")
    fig_costs(sim, cmp, path=out / "fig5_costs.png")
    fig_promise(sim, path=out / "fig6_promise_vs_twin.png")
    digest = hashlib.sha256(cmp.round(4).to_csv(index=False).encode()).hexdigest()[:16]
    print(f"\nOutput: {out}\nCompare hash: {digest}")


if __name__ == "__main__":
    main()
