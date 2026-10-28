"""Driver pay by the shift, and the planned lead-time estimator against the twin."""
import numpy as np
import pytest

from conftest import run_twin
from inbound_twin.config import CostCfg
from inbound_twin.costs import driver_cost, shift_hours
from inbound_twin.kpi import shift_spans_h


def test_shift_hours_and_cost_by_hand():
    # 7.5 h is paid the 8 h minimum; 9 h has 1 h overtime; 10.25 h has 2.25 h
    n, ot = shift_hours([7.5, 9.0, 10.25], min_paid_h=8.0)
    assert n == 3 and ot == pytest.approx(3.25)
    c = CostCfg()
    assert driver_cost(n, ot, c) == pytest.approx(35.0 * (3 * 8 + 1.5 * 3.25))


def test_shift_spans_group_waves_by_shift(default_run_30):
    tw = default_run_30
    spans = shift_spans_h(tw, 5)
    # naive plan: 5 trucks x 2 shifts (04/09 early, 14/19 late)
    assert len(spans) == 2 * tw.cfg.fleet.n_trucks
    # an early shift runs from the 04:00 dispatch past the 09:00 dispatch
    assert all(s > 5.0 for s in spans)


def test_planned_lead_matches_twin_mean(base_cfg):
    """route_timing is an expected-value estimate, so it is checked against the
    twin's MEAN lead per route (the distribution is right-skewed: supplier delays
    and dock waits are rare but long, so the median sits below the mean)."""
    from inbound_twin import generate_network, naive_plan
    net = generate_network(base_cfg)
    plan = naive_plan(net, base_cfg)
    measured: dict[tuple[int, float], list[float]] = {}
    for seed in (1, 2, 3):
        tw = run_twin(base_cfg, seed=seed, days=20, net=net, plan=plan, check=False)
        for t in tw.trips:
            if t.day < 1:
                continue
            for truck, t_arrive, t_received, _ in tw.batches:
                if truck == t.truck and t.t_dispatch <= t_arrive <= t.t_end:
                    measured.setdefault((t.truck, t.t_sched % 1440), []).append(t_received - t.t_dispatch)
    for r in plan.routes:
        mean = np.mean(measured[(r.truck, r.dispatch_min)])
        assert abs(r.lead_min - mean) / mean < 0.10, (r, mean)
