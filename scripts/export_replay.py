"""Replays for the showcase video and the web viewer.

    --find    scan the test seeds for a "hero day": the static plan stops the line,
              and the pre-registered winner, det(m=0)+always, keeps it running
    --seed S --day D [--regime R2]
              export that day, under both policies, to showcase/web/data/<name>.json

    .venv/Scripts/python scripts/export_replay.py --find
    .venv/Scripts/python scripts/export_replay.py --regime R2 --seed 107 --day 6

Every position and time in a replay comes from the twin's own records; the
viewer only interpolates between them. See docs/REPLAY.md for the format.
"""
from __future__ import annotations

import argparse
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from inbound_twin import Twin
from inbound_twin.evaluate import _setup
from inbound_twin.policy import Controller, PolicySpec

ROOT = Path(__file__).resolve().parents[1]
HERO = PolicySpec("det", "always", 0.0)   # the pre-registered winner (SPEC section 17.1)
DAYS = 15                                 # as in the Phase 3 test runs
TEST_SEEDS = range(101, 121)


def run_pair(regime: str, seed: int, days: int = DAYS, record=None, extra: tuple = ()):
    """The same seed under the static plan and under HERO (common random numbers)."""
    cfg, net, plan = _setup(regime, None, extra)
    never = Twin(cfg, net, plan, seed=seed, days=days).run()
    ctl = Controller(cfg, net, plan, seed, HERO)
    if record is not None:
        ctl = record(ctl)
    dyn = Twin(cfg, net, plan, seed=seed, days=days, controller=ctl).run()
    return cfg, net, plan, never, dyn, ctl


def down_min(tw, day: int) -> float:
    return sum(b - a for a, b, _ in tw.downtime if int(a // 1440) == day)


def late_waits(tw, day: int) -> list[tuple[int, float, float]]:
    """(supplier, arrival, minutes waited) for not-ready stops that day; segments give the wait."""
    waits = {round(a, 6): b - a for _, kind, a, b in tw.segments if kind == "late"}
    return [(s, t, waits.get(round(t, 6), 0.0)) for s, t, late in tw.stop_log
            if late and int(t // 1440) == day]


def slow_legs(tw, day: int, ratio: float = 1.6) -> list[tuple[int, int, int, float]]:
    """(truck, from, to, actual / nominal) for legs that day much slower than nominal."""
    out = []
    for truck, i, j, a, b in tw.legs:
        if int(a // 1440) == day:
            r = (b - a) / max(tw.net.nominal_min(i, j), 1e-9)
            if r >= ratio:
                out.append((truck, i, j, r))
    return out


def scan(job) -> list[dict]:
    regime, seed = job
    cfg, net, _, never, dyn, _ = run_pair(regime, seed)
    rows = []
    for d in range(cfg.warmup_days, DAYS):
        stops = sorted((a, b, parts) for a, b, parts in never.downtime if int(a // 1440) == d)
        if not stops:
            continue
        a0, _, parts0 = stops[0]
        sups = {net.parts[p].supplier for p in parts0}
        cause = [f"s{s} late {w:.0f}m" for s, t, w in late_waits(never, d) if s in sups and t < a0]
        cause += [f"leg {i}-{j} x{r:.1f}" for _, i, j, r in slow_legs(never, d) if (i in sups or j in sups)]
        acts = [x for x in dyn.actions if x["day"] == d]
        rows.append({
            "regime": regime, "seed": seed, "day": d,
            "never_down": round(down_min(never, d), 1), "hero_down": round(down_min(dyn, d), 1),
            "first_stop": f"{(a0 - d * 1440) / 60:05.2f}h", "parts": "-".join(map(str, parts0)),
            "suppliers": "-".join(map(str, sorted(sups))), "cause": "; ".join(cause) or "-",
            "n_actions": len(acts), "kinds": ",".join(sorted({x["kind"] for x in acts})) or "-",
        })
    return rows


def find(workers: int) -> None:
    jobs = [(r, s) for r in ("R1", "R2") for s in TEST_SEEDS]
    with Pool(workers) as pool:
        rows = [row for part in pool.map(scan, jobs, chunksize=1) for row in part]
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "replay"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "hero_candidates.csv", index=False)
    # A good hero day: a long stop under the static plan, none under the twin, a visible
    # cause on the part's own supplier, in daylight hours, and few actions (a clean story).
    good = df[(df.hero_down == 0) & (df.never_down >= 20) & (df.cause != "-")].copy()
    good["hour"] = good.first_stop.str.rstrip("h").astype(float)
    good = good[(good.hour >= 8) & (good.hour <= 20)]
    good = good.sort_values(["never_down", "n_actions"], ascending=[False, True])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)
    print(f"{len(df)} days with a stop under the static plan; {len(good)} good hero candidates:\n")
    print(good.head(20).drop(columns="hour").to_string(index=False))
    print(f"\nAll candidates: {out / 'hero_candidates.csv'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--find", action="store_true")
    ap.add_argument("--regime", default="R2", choices=("R1", "R2"))
    ap.add_argument("--seed", type=int)
    ap.add_argument("--day", type=int)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    if args.find:
        find(args.workers)
    elif args.seed is not None and args.day is not None:
        export(args.regime, args.seed, args.day)
    else:
        ap.error("use --find, or give --seed and --day")


# ------------------------------------------------------------------ export
SCHEMA = "inbound-twin-replay/1"
PLANT_STATES = (("gate", "t_arrive", "t_yard"), ("yard", "t_yard", "t_door"), ("door_wait", "t_door", "t_crew"),
                ("unload", "t_crew", "t_unloaded"), ("dock_out", "t_unloaded", "t_leave"))


def with_risk_trace(ctl):
    """Record what the trigger saw at every risk check: the least projected slack and its part."""
    ctl.risk = []
    inner = ctl.pred.forecast

    def forecast(obs, est, cands):
        fs = inner(obs, est, cands)
        if len(cands) == 1 and (not ctl.risk or ctl.risk[-1][0] != obs.t):
            m = fs[0].margin_min
            p = int(np.argmin(m))
            ctl.risk.append((obs.t, float(m[p]), p))
        return fs

    ctl.pred.forecast = forecast
    return ctl


def _r(x: float, nd: int = 2):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def truck_tracks(tw, lo: float, hi: float) -> list[dict]:
    """Per milk-run truck, every state it was in that overlaps [lo, hi], in minutes from lo."""
    legs = {(tr, round(a, 6)): (i, j) for tr, i, j, a, _ in tw.legs}
    out = []
    for truck in range(tw.cfg.fleet.n_trucks):
        rows = []
        node = 0
        for tr, kind, a, b in sorted((s for s in tw.segments if s[0] == truck), key=lambda s: s[2]):   # by time
            if kind == "travel":
                i, j = legs[(tr, round(a, 6))]
                row = {"state": "travel", "from": i, "to": j}
                node = j
            else:
                row = {"state": kind, "at": node}
            if b > lo and a < hi:
                rows.append({"t0": _r(a - lo), "t1": _r(b - lo), **row})
        for v in tw.visits:
            if v.kind != "milk" or v.key != truck:
                continue
            for state, ta, tb in PLANT_STATES:
                a, b = getattr(v, ta), getattr(v, tb)
                if not (math.isnan(a) or math.isnan(b)) and b > lo and a < hi:
                    rows.append({"t0": _r(a - lo), "t1": _r(b - lo), "state": state, "at": 0})
        out.append({"id": truck, "track": sorted(rows, key=lambda r: r["t0"])})
    return out


def doors_out(tw) -> list[tuple[float, float]]:
    """Stress test: (out, back) for each door taken out of service, from the event log."""
    outs = sorted(t for t, ev, *_ in tw.log if ev == "door_out")
    backs = sorted(t for t, ev, *_ in tw.log if ev == "door_back")
    return [(a, backs[i] if i < len(backs) else math.inf) for i, a in enumerate(outs)]


def dock_visits(tw, lo: float, hi: float) -> list[dict]:
    """Every truck at the dock (milk runs and other carriers), with a door number.

    The twin's doors are one pooled resource; numbers are assigned here, first free door
    in order of getting one, only so that a viewer can draw each truck at a door. Doors out
    of service take the highest numbers."""
    n = tw.cfg.plant.n_doors
    free_at = [-math.inf] * n
    blocked = [(n - 1 - k, a, b) for k, (a, b) in enumerate(doors_out(tw))]
    out = []
    for v in sorted(tw.visits, key=lambda v: (math.inf if math.isnan(v.t_door) else v.t_door)):
        door = None
        if not math.isnan(v.t_door):
            end = math.inf if math.isnan(v.t_leave) else v.t_leave
            ok = [k for k in range(n) if free_at[k] <= v.t_door + 1e-6
                  and not any(d == k and a < end and v.t_door < b for d, a, b in blocked)]
            door = min(ok) if ok else min(range(n), key=lambda k: free_at[k])
            free_at[door] = end
        end = v.t_leave if not math.isnan(v.t_leave) else hi
        if end > lo and v.t_arrive < hi:
            out.append({"truck": v.key if v.kind == "milk" else None, "kind": v.kind, "containers": v.n,
                        "door": door, **{k: _r(getattr(v, k) - lo) for _, k, _ in PLANT_STATES},
                        "t_leave": _r(v.t_leave - lo)})
    return out


def run_record(tw, ctl, day: int) -> dict:
    from inbound_twin.costs import daily_costs
    from inbound_twin.kpi import daily_kpis
    cfg, net = tw.cfg, tw.net
    lo, hi = day * 1440.0, (day + 1) * 1440.0
    rate_h = np.array([p.usage for p in net.parts], dtype=float) * cfg.line.jph
    m = (tw.samples["t"] >= lo) & (tw.samples["t"] <= hi)
    cost = daily_costs(tw, daily_kpis(tw)).set_index("day").loc[day]
    labels = {round(c["t"], 6): c["label"] for c in (ctl.decisions if ctl else []) if c["executed"]}
    rec = {
        "trucks": truck_tracks(tw, lo, hi),
        "dock": dock_visits(tw, lo, hi),
        "receipts": [{"truck": tr, "t_arrive": _r(a - lo), "t_received": _r(b - lo), "containers": n}
                     for tr, a, b, n in tw.batches if lo <= b < hi],
        "couriers": [{"id": c, "supplier": s, "t_dispatch": _r(a - lo), "t_at_supplier": _r(at - lo),
                      "t_pickup": _r(p - lo), "t_at_plant": _r(r - lo - cfg.dynamic.courier_receiving_min),
                      "t_received": _r(r - lo), "containers": n}
                     for c, s, a, at, p, r, n in tw.courier_log if r > lo and a < hi],
        "tugger": [{"t_depart": _r(t - lo), "t_drop": _r(t - lo + cfg.tugger.loop_min / 2), "containers": n}
                   for t, n in tw.tugger_loads if lo <= t < hi],
        "hot_runs": [{"t_start": _r(t - lo - cfg.dynamic.hot_run_min), "t_drop": _r(t - lo), "part": p,
                      "containers": n} for t, ev, p, n, _ in tw.log if ev == "hot_run" and lo <= t < hi],
        "line_stops": [{"t0": _r(a - lo), "t1": _r(b - lo), "parts": list(parts)}
                       for a, b, parts in tw.downtime if lo <= a < hi],
        "doors_out": [{"door": tw.cfg.plant.n_doors - 1 - k, "t0": _r(a - lo), "t1": _r(min(b, hi) - lo)}
                      for k, (a, b) in enumerate(doors_out(tw)) if a < hi and b > lo],
        "cover_h": {   # hours of production each part's stock covers, every 5 min
            "t": [_r(t - lo, 1) for t in tw.samples["t"][m]],
            "line": (tw.samples["line"][m] / rate_h).round(2).T.tolist(),
            "supermarket": (tw.samples["supermarket"][m] / rate_h).round(2).T.tolist(),
        },
        "actions": [{"t": _r(a["t"] - lo), "kind": a["kind"], "truck": a["truck"], "supplier": a["supplier"],
                     "part": a["part"], "premium_usd": _r(a["premium"]), "label": labels.get(round(a["t"], 6), "")}
                    for a in tw.actions if a["day"] == day],
        "risk": [{"t": _r(t - lo), "least_slack_min": _r(s, 1), "part": p}
                 for t, s, p in (getattr(ctl, "risk", []) if ctl else []) if lo <= t < hi],
        "day_kpis": {"vehicles": int(tw.vehicles_by_day[day]),
                     "line_down_min": _r(sum(b - a for a, b, _ in tw.downtime if lo <= a < hi), 1),
                     "line_down_usd": _r(cost["line_down"], 0), "premium_usd": _r(cost["premium"], 0),
                     "disruption_usd": _r(cost["disruption"], 0),
                     "logistics_usd": _r(cost["total"] - cost["line_down"], 0), "total_usd": _r(cost["total"], 0)},
    }
    return rec


def headline() -> dict:
    """The study-wide numbers a viewer may show next to the one day, read from the result files."""
    p2 = pd.read_csv(ROOT / "outputs" / "phase2" / "compare.csv").set_index(["plan", "metric"])
    plans = pd.read_csv(ROOT / "outputs" / "phase2" / "plans.csv")
    t3 = pd.read_csv(ROOT / "outputs" / "phase3" / "test.csv")
    t3["total"] = t3.logistics + 5000.0 * t3.line_down_min
    mean = t3.groupby(["regime", "policy"]).total.mean()
    logi = pd.read_csv(ROOT / "outputs" / "phase2" / "daily.csv").groupby("plan").logistics.mean()
    return {
        "source": "outputs/phase2/{compare,plans,daily}.csv, outputs/phase3/test.csv",
        "optimization": {
            "logistics_usd_per_day": _r(p2.loc[("optimized", "logistics"), "mean_diff"], 0),
            "logistics_pct": _r(100 * (logi["optimized"] / logi["naive"] - 1), 1),
            "line_down_min_per_day": _r(p2.loc[("optimized", "line_down_min"), "mean_diff"], 1),
            "pickups_per_day": {name: int(sum(len(str(s).split("-")) for s in g.suppliers))
                                for name, g in plans.groupby("plan")},
            "seeds": "101-120, 30 days",
        },
        "re_planning": {r: {"never_usd_per_day": _r(mean[(r, "never")], 0),
                            "twin_usd_per_day": _r(mean[(r, HERO.name)], 0),
                            "change_pct": _r(100 * (mean[(r, HERO.name)] / mean[(r, "never")] - 1), 0)}
                        for r in ("R1", "R2")},
        "re_planning_note": f"{HERO.name} vs never re-plan, 2 h safety stock, $5,000/line-down min, seeds 101-120",
    }


def story(net, never, dyn, day: int) -> dict:
    """Pointers a viewer can use to frame the shot: the first stop under the static plan and
    what the twin did about it. Derived, not hand-picked. Without a stop, the part that ran
    lowest at the line under the static plan."""
    lo = day * 1440.0
    stops = sorted((a, b, p) for a, b, p in never.downtime if int(a // 1440) == day)
    first = min((x for x in dyn.actions if x["day"] == day), key=lambda x: x["t"], default=None)
    if not stops:
        m = (never.samples["t"] >= lo) & (never.samples["t"] < lo + 1440)
        rate = np.array([p.usage for p in net.parts], dtype=float) * never.cfg.line.jph
        part = int(np.argmin((never.samples["line"][m] / rate).min(axis=0)))
        return {"focus_part": part, "focus_supplier": net.parts[part].supplier, "focus_truck": None,
                "static_stop": None, "first_twin_action": _r(first["t"] - lo) if first else None,
                "warning_lead_min": None}
    a0, b0, parts = stops[0]
    part = parts[0]
    sup = net.parts[part].supplier
    # The late load: the earliest pickup at the part's supplier that was made before the stop
    # but only received after it started.
    def received_at(tr, t):   # the batch of the trip that made this pickup: the truck's next arrival
        return min(((ta, tb) for b, ta, tb, _ in never.batches if b == tr and ta >= t), default=(None, math.inf))[1]

    late = [(t, tr) for t, ev, tr, _, s in never.log
            if ev == "pickup" and s == sup and lo - 360 <= t < a0 and received_at(tr, t) > a0]
    carrier = min(late)[1] if late else None
    return {"focus_part": part, "focus_supplier": sup, "focus_truck": carrier,
            "static_stop": [_r(a0 - lo), _r(b0 - lo)],
            "first_twin_action": _r(first["t"] - lo) if first else None,
            "warning_lead_min": _r(a0 - first["t"], 0) if first and first["t"] < a0 else None}


def export(regime: str, seed: int, day: int, scenario: str | None = None) -> Path:
    import json
    from dataclasses import asdict

    from inbound_twin.scenario import corridor_path
    from inbound_twin.stress import SCENARIOS, extra
    cfg, net, plan, never, dyn, ctl = run_pair(regime, seed, day + 1, record=with_risk_trace,
                                               extra=extra(scenario) if scenario else ())
    plans = pd.read_csv(ROOT / "outputs" / "phase2" / "plans.csv")
    rep = {
        "schema": SCHEMA,
        "meta": {
            "regime": regime, "seed": seed, "day": day, "synthetic": True,
            "time": "minutes from 00:00 of the replayed day",
            "policies": {"static": "never re-plan: the Phase 2 optimized plan runs untouched",
                         "twin": f"{HERO.name}: re-plan when the projected slack of any part drops "
                                 "below 0 within 6 h; act on the best candidate"},
            "line": {"jph": cfg.line.jph, "takt_min": cfg.line.takt_min,
                     "prod_start_h": cfg.line.prod_start_h, "prod_end_h": cfg.line.prod_end_h},
            "plant": {"doors": cfg.plant.n_doors, "crews": cfg.plant.n_crews,
                      "tugger_loop_min": cfg.tugger.loop_min, "tugger_capacity": cfg.tugger.capacity,
                      "hot_run_min": cfg.dynamic.hot_run_min},
            "fleet": {"trucks": cfg.fleet.n_trucks, "capacity": cfg.fleet.capacity},
            "line_down_usd_per_min": cfg.costs.line_down_per_min,
            "stress": None if not scenario else {
                "id": scenario, "label": SCENARIOS[scenario]["label"], "detail": SCENARIOS[scenario]["detail"],
                **{k: v for k, v in asdict(cfg.stress).items() if k != "day"},
                "start": cfg.stress.start_h * 60.0, "end": cfg.stress.end_h * 60.0},
        },
        "network": {
            "plant": {"id": 0, "x_km": 0.0, "y_km": 0.0},
            "suppliers": [{"id": s.id, "x_km": _r(s.x_km), "y_km": _r(s.y_km), "p_late": _r(s.p_late, 3),
                           "parts": list(net.parts_of(s.id))} for s in net.suppliers],
            "parts": [{"id": p.id, "supplier": p.supplier, "usage": p.usage, "upc": p.upc, "kanban": p.kanban}
                      for p in net.parts],
            "road_km": np.round(net.dist_km, 2).tolist(),
        },
        "traffic": {   # true corridor congestion (hidden from the controller), x multiplier per 15 min
            "step_min": 15,
            "multiplier": {s.id: np.round(np.exp(corridor_path(cfg, seed, day, s.id)), 3).tolist()
                           for s in net.suppliers},
        },
        "plans": {name: [{"truck": int(r.truck), "dispatch_h": float(r.dispatch_h),
                          "suppliers": [int(x) for x in str(r.suppliers).split("-")]} for r in g.itertuples()]
                  for name, g in plans.groupby("plan")},
        "runs": {"static": run_record(never, None, day), "twin": run_record(dyn, ctl, day)},
        "story": story(net, never, dyn, day),
        "headline": headline(),
    }
    out = ROOT / "showcase" / "web" / "data" / (f"replay_{scenario}.json" if scenario else f"replay_{regime}_s{seed}_d{day}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, separators=(",", ":")))
    s, k = rep["story"], {r: rep["runs"][r]["day_kpis"] for r in rep["runs"]}
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")
    print(f"Story: part {s['focus_part']} (supplier {s['focus_supplier']}, truck {s['focus_truck']}); "
          f"static stop {s['static_stop']} min; first twin action at {s['first_twin_action']} min, "
          f"lead {s['warning_lead_min']} min")
    for r, v in k.items():
        print(f"  {r:6s} {v}")
    print("Headline:", rep["headline"]["optimization"], rep["headline"]["re_planning"])
    return out


if __name__ == "__main__":
    main()
