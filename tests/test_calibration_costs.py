"""Calibration bands for the default scenario, and the cost formulas by hand.

Bands (busy time over the 20 h operating window): the default plant is meant
to be busy but not saturated, so the Phase 3 policies have something to
fix. Doors sit lower than crews by construction: a door is held while the
truck waits for one of 2 crews, and there are 4 doors.
"""
import pytest

from conftest import override
from inbound_twin.config import CostCfg
from inbound_twin.costs import (daily_costs, disruption_cost, holding_capital,
                                premium_freight_cost, space_cost)
from inbound_twin.kpi import daily_kpis

BANDS = {"truck_util": (0.70, 0.85), "door_util": (0.35, 0.50), "crew_util": (0.50, 0.65),
         "tugger_util": (0.65, 0.80), "truck_fill": (0.60, 0.85)}


@pytest.mark.parametrize("kpi", sorted(BANDS))
def test_default_utilization_in_band(default_run_30, kpi):
    mean = daily_kpis(default_run_30)[kpi].mean()
    lo, hi = BANDS[kpi]
    assert lo <= mean <= hi, f"{kpi} = {mean:.3f}, expected [{lo}, {hi}]"


def test_cost_formulas_by_hand():
    c = CostCfg()
    # 100 unit-hours of a $10 part + 100 unit-hours of a $20 part at 20 %/yr
    assert holding_capital([6000.0, 6000.0], [10.0, 20.0], 0.20) == pytest.approx(3000 * 0.20 / 8760)
    assert space_cost(100, 1.5, days=2) == pytest.approx(300.0)
    assert premium_freight_cost(40.0, c) == pytest.approx(250 + 3.0 * 40)
    assert disruption_cost(2, 3, c) == pytest.approx(2 * 50 + 3 * 20)


def test_daily_total_is_sum_of_parts(default_run_30):
    k = daily_kpis(default_run_30)
    c = daily_costs(default_run_30, k)
    parts = ["transport", "driver", "trucks", "handling", "holding_capital", "holding_space", "line_down",
             "premium", "disruption"]
    assert (c[parts].sum(axis=1) - c["total"]).abs().max() < 1e-6
    assert (c["line_down"] == k["line_down_min"] * default_run_30.cfg.costs.line_down_per_min).all()


def test_config_rejects_bad_values(base_cfg):
    with pytest.raises(ValueError):
        override(base_cfg, {"line.prod_start_h": 23})
    with pytest.raises(ValueError):
        override(base_cfg, {"plant.no_such_field": 1})
