"""Per-day KPIs and Little's-law inputs, computed from a finished Twin run.

Utilizations of trucks, doors and crews are busy time over the operating
window (config.plant.operating_window_h), so 1.0 means busy all window long.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .costs import shift_hours
from .plan import wave_index


def _overlap(intervals, lo: float, hi: float) -> float:
    return sum(max(0.0, min(b, hi) - max(a, lo)) for a, b in intervals)


def _end(t: float, horizon: float) -> float:
    return horizon if math.isnan(t) else t


def shift_spans_h(tw, day: int) -> list[float]:
    """Span in hours (first dispatch to last door release) of each driver shift on a day.

    A shift is one truck's trips in the waves that config.fleet.shift_of_wave
    maps to the same shift; trips belong to the day they were scheduled on.
    """
    cfg = tw.cfg
    spans: dict[tuple[int, int], list[float]] = {}
    for t in tw.trips:
        if t.day != day:
            continue
        shift = cfg.fleet.shift_of_wave[wave_index(cfg, t.t_sched)]
        lo_hi = spans.setdefault((t.truck, shift), [math.inf, -math.inf])
        lo_hi[0] = min(lo_hi[0], t.t_dispatch)
        lo_hi[1] = max(lo_hi[1], t.t_end)
    return [(hi - lo) / 60.0 for lo, hi in spans.values()]


def daily_kpis(tw) -> pd.DataFrame:
    cfg = tw.cfg
    horizon = tw.days * 1440.0
    window = cfg.plant.window_min
    p_start, p_end = cfg.line.prod_start_h * 60.0, cfg.line.prod_end_h * 60.0
    rate_h = np.array([p.usage for p in tw.net.parts]) * cfg.line.jph
    t_s = tw.samples["t"]

    door_iv = [(v.t_door, _end(v.t_leave, horizon)) for v in tw.visits if not math.isnan(v.t_door)]
    crew_iv = [(v.t_crew, _end(v.t_unloaded, horizon)) for v in tw.visits if not math.isnan(v.t_crew)]
    truck_iv = [(t.t_dispatch, t.t_end) for t in tw.trips]
    down_iv = [(a, b) for a, b, _ in tw.downtime]

    rows = []
    for d in range(cfg.warmup_days, tw.days):
        lo, hi = d * 1440.0, (d + 1) * 1440.0
        trips = [t for t in tw.trips if t.day == d]
        milk = [v for v in tw.visits if v.kind == "milk" and lo <= v.t_arrive < hi
                and not math.isnan(v.t_crew)]
        batches = [b for b in tw.batches if lo <= b[1] < hi]
        loads = [n for t, n in tw.tugger_loads if lo + p_start <= t < lo + p_end]
        in_prod = (t_s >= lo + p_start) & (t_s < lo + p_end)
        cover_h = tw.samples["supermarket"][in_prod] / rate_h

        n_batch = sum(b[3] for b in batches)
        n_shifts, overtime_h = shift_hours(shift_spans_h(tw, d), cfg.costs.driver_min_paid_h)
        rows.append({
            "day": d,
            "vehicles": int(tw.vehicles_by_day[d]),
            "line_down_min": _overlap(down_iv, lo, hi),
            "line_stops": sum(1 for a, _ in down_iv if lo <= a < hi),
            "trips": len(trips),
            "km": sum(t.km for t in trips),
            "driver_h": _overlap(truck_iv, lo, hi) / 60.0,
            "driver_shifts": n_shifts,
            "overtime_h": overtime_h,
            "trucks_used": len({t.truck for t in trips}),
            "containers_picked": sum(t.n for t in trips),
            "containers_unloaded": sum(v.n for v in milk),
            "truck_fill": float(np.mean([t.n for t in trips])) / cfg.fleet.capacity if trips else 0.0,
            "dispatch_late_min": float(np.mean([t.t_dispatch - t.t_sched for t in trips])) if trips else 0.0,
            "truck_util": _overlap(truck_iv, lo, hi) / (cfg.fleet.n_trucks * window),
            "door_util": _overlap(door_iv, lo, hi) / (cfg.plant.n_doors * window),
            "crew_util": _overlap(crew_iv, lo, hi) / (cfg.plant.n_crews * window),
            "tugger_util": float(np.mean(loads)) / cfg.tugger.capacity if loads else 0.0,
            "yard_wait_min": float(np.mean([v.t_door - v.t_yard for v in milk])) if milk else 0.0,
            "gate_to_unload_min": float(np.mean([v.t_crew - v.t_arrive for v in milk])) if milk else 0.0,
            "gate_to_supermarket_min": (sum(b[3] * (b[2] - b[1]) for b in batches) / n_batch
                                        if n_batch else 0.0),
            "supermarket_cover_h": float(cover_h.mean()) if cover_h.size else 0.0,
            "min_part_cover_h": float(cover_h.min()) if cover_h.size else 0.0,
        })
    return pd.DataFrame(rows)


def littles_law(tw) -> dict[str, float]:
    """Plant segment = gate arrival to receiving done (milk-run containers).

    L comes from the time integral of the state; lambda and W come from the
    per-batch records. They are computed independently, so L = lambda * W is
    a real check on the bookkeeping.
    """
    horizon = tw.days * 1440.0
    upc = np.array([p.upc for p in tw.net.parts], dtype=float)
    container_minutes = ((tw.area["in_plant"].sum(axis=0) + tw.area["staging"].sum(axis=0)) / upc).sum()
    n = sum(b[3] for b in tw.batches)
    lam = n / horizon
    w = sum(b[3] * (b[2] - b[1]) for b in tw.batches) / n
    return {"L": container_minutes / horizon, "lambda": lam, "W": w, "lambda_W": lam * w}
