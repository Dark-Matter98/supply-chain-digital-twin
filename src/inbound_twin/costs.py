"""Cost model. Every rate is illustrative (config.costs), not real company data.

Daily total = transport + driver + trucks + handling + holding (capital + space)
+ line-down. Premium freight and disruption are defined here and used from
Phase 3 on.

Drivers are paid by the shift, idle time between waves included: each shift
is paid at least driver_min_paid_h, and span beyond that is overtime.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import CostCfg

HOURS_PER_YEAR = 8760.0


def shift_hours(spans_h: Iterable[float], min_paid_h: float) -> tuple[int, float]:
    """(number of shifts, overtime hours) for a set of shift spans in hours."""
    spans = list(spans_h)
    return len(spans), float(sum(max(0.0, s - min_paid_h) for s in spans))


def driver_cost(n_shifts: int, overtime_h: float, c: CostCfg) -> float:
    return c.driver_per_h * (n_shifts * c.driver_min_paid_h + c.overtime_factor * overtime_h)


def holding_capital(unit_minutes: np.ndarray, unit_value: np.ndarray, rate_per_year: float) -> float:
    """Capital tied up: sum over parts of unit-hours x unit value x hourly carrying rate."""
    unit_hours = np.asarray(unit_minutes, dtype=float) / 60.0
    return float((unit_hours * unit_value).sum() * rate_per_year / HOURS_PER_YEAR)


def space_cost(positions: float, rate_per_position_day: float, days: float = 1.0) -> float:
    """Rack space is a capacity cost: positions the plant must provide, not average use."""
    return float(positions * rate_per_position_day * days)


def premium_freight_cost(km: float, c: CostCfg) -> float:
    return c.premium_freight_fixed + c.premium_freight_per_km * km


def disruption_cost(changes: int, stops_changed: int, c: CostCfg) -> float:
    return c.disruption_per_change * changes + c.disruption_per_stop * stops_changed


def space_positions(tw) -> int:
    """Supermarket + staging positions sized to the run's peak (after warm-up),
    plus the fixed line-side rack positions (one per kanban card)."""
    peak = int(tw.peak_positions[tw.cfg.warmup_days:].max())
    return peak + int(sum(p.kanban for p in tw.net.parts))


def daily_costs(tw, kpis: pd.DataFrame) -> pd.DataFrame:
    c = tw.cfg.costs
    value = np.array([p.unit_value for p in tw.net.parts])
    positions = space_positions(tw)
    rows = []
    for _, k in kpis.iterrows():
        d = int(k["day"])
        unit_minutes = sum(tw.area[loc][d] for loc in tw.area)
        row = {
            "day": d,
            "transport": k["km"] * c.truck_per_km,
            "driver": driver_cost(int(k["driver_shifts"]), k["overtime_h"], c),
            "trucks": c.truck_per_day * tw.plan.fleet,
            "handling": k["containers_unloaded"] * c.handling_per_container,
            "holding_capital": holding_capital(unit_minutes, value, c.carrying_rate_per_year),
            "holding_space": space_cost(positions, c.space_per_position_day),
            "line_down": k["line_down_min"] * c.line_down_per_min,
            "premium": sum(a["premium"] for a in tw.actions if a["day"] == d),
            "disruption": sum(a["disruption"] for a in tw.actions if a["day"] == d),
        }
        row["total"] = sum(v for key, v in row.items() if key != "day")
        rows.append(row)
    return pd.DataFrame(rows)
