"""Stress tests: named disruptions, each a set of config overrides (see StressCfg).

Each one hits a different part of the inbound system: a supplier, a road, the dock.
They run on one day of an otherwise normal run, so the day before is a clean baseline.
"""
from __future__ import annotations

STRESS_DAY = 7          # runs are STRESS_DAY + 1 days long; day 0 is the warm-up

SCENARIOS = {
    "supplier_outage": {
        "label": "Supplier 10 shut for 8 hours",
        "detail": "Supplier 10 cannot load anything from 08:00 to 16:00. Trucks and couriers that arrive wait until it reopens.",
        "overrides": {"stress.day": STRESS_DAY, "stress.start_h": 8.0, "stress.end_h": 16.0, "stress.supplier_down": 10},
    },
    "road_closure": {
        "label": "Road to supplier 5 at a crawl",
        "detail": "Legs to and from supplier 5 take three times as long from 07:00 to 13:00, as after a crash or a closure.",
        "overrides": {"stress.day": STRESS_DAY, "stress.start_h": 7.0, "stress.end_h": 13.0, "stress.road_supplier": 5,
                      "stress.road_multiplier": 3.0},
    },
    "dock_doors": {
        "label": "Two of four dock doors out",
        "detail": "Two dock doors are out of service from 06:00 to 14:00, so the morning waves queue in the yard.",
        "overrides": {"stress.day": STRESS_DAY, "stress.start_h": 6.0, "stress.end_h": 14.0, "stress.doors_out": 2},
    },
}


def extra(name: str) -> tuple:
    """The scenario's overrides in the hashable form evaluate._setup takes."""
    return tuple(sorted(SCENARIOS[name]["overrides"].items()))
