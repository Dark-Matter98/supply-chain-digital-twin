"""Extreme-condition tests: push one input to an extreme, check the direction."""
import math

import numpy as np

from conftest import override, run_twin
from inbound_twin.costs import daily_costs
from inbound_twin.kpi import daily_kpis

CALM = {"network.travel_sigma": 0.0, "network.p_supplier_late": 0.0, "traffic.rush_multiplier": 1.0,
        "plant.unload_cv": 0.0, "plant.receiving_cv": 0.0}


def _down(tw):
    return sum(b - a for a, b, _ in tw.downtime)


def test_no_noise_and_big_buffer_never_starves(base_cfg):
    cfg = override(base_cfg, {**CALM, "inventory.ss_h": 6.0, "inventory.initial_cover_h": 12.0})
    assert _down(run_twin(cfg, days=6)) == 0


def test_no_buffer_and_heavy_noise_starves(base_cfg):
    cfg = override(base_cfg, {"inventory.ss_h": 0.0, "network.travel_sigma": 0.5,
                              "network.p_supplier_late": 0.3, "network.supplier_late_mean_min": 90})
    assert _down(run_twin(cfg, days=6)) > 0


def test_more_doors_less_yard_wait(base_cfg):
    def mean_yard_wait(n_doors):
        tw = run_twin(override(base_cfg, {"plant.n_doors": n_doors}), days=4)
        return np.mean([v.t_door - v.t_yard for v in tw.visits if not math.isnan(v.t_door)])
    assert mean_yard_wait(4) < mean_yard_wait(1)


def test_more_safety_stock_more_holding_cost(base_cfg):
    def holding(ss):
        tw = run_twin(override(base_cfg, {"inventory.ss_h": ss}), days=5)
        c = daily_costs(tw, daily_kpis(tw))
        return (c["holding_capital"] + c["holding_space"]).mean()
    assert holding(5.0) > holding(1.0)
