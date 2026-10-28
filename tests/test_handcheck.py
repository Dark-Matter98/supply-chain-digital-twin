"""One supplier, one part, one truck, no noise: every time stamp worked out by hand.

Setup: supplier 30 road-km away at 60 km/h -> 30 min each way. Part 0 is used
once per vehicle (50 units/h at 50 JPH), 50 units/container, 2 kanban cards.
Supermarket starts with 2 containers, the line with 1 full + 1 open = 100 units.

Planned lead (plan.route_timing, expected volume 16 containers = one day):
30 drive + 15 + 0.5*16 load + 30 drive + 5 gate + 13 dock wait + 5 + 1.2*16
unload + 20 receiving = 145.2 min.
Order at 04:00 (t=240): next pickup is 04:00 next day; its load is covered
from 04:00 + 145.2 min = 06:25.2 day 1 -> 16 h + 0.42 h = 16.42 production
hours = 821 units. Position = 200 units, so order ceil(621 / 50) = 13 containers.

 240.0  dispatch
 270.0  at supplier; load 15 + 0.5*13 = 21.5 min -> pickup at 291.5
 321.5  back at the plant gate (gate 5 min) -> door and crew free at 326.5
 347.1  unloaded (5 + 1.2*13 = 20.6 min); dock out 5 min -> door free 352.1
 367.1  received into the supermarket (receiving 20 min)
 418.8  50th vehicle empties the open container (360 + 49*1.2) -> kanban signal
 420.0  tugger departs with 1 container, drops it at the line at 435.0
"""
import numpy as np
import pytest

from conftest import override, run_twin
from inbound_twin.costs import daily_costs
from inbound_twin.kpi import daily_kpis
from inbound_twin.network import Network, Part, Supplier
from inbound_twin.plan import naive_plan, order_up_to_units, production_hours


@pytest.fixture(scope="module")
def tiny(base_cfg):
    cfg = override(base_cfg, {
        "warmup_days": 0, "fleet.n_trucks": 1, "fleet.dispatch_h": [4], "fleet.shift_of_wave": [0],
        "traffic.rush_multiplier": 1.0, "plant.unload_cv": 0, "plant.receiving_cv": 0,
        "plant.background_trucks_per_h": 0, "planning.dock_wait_min": 13.0,
        "inventory.lead_margin_h": 0.0, "inventory.ss_h": 0.0, "inventory.initial_cover_h": 2.0,
    })
    net = Network(
        suppliers=(Supplier(1, 30.0, 0.0, travel_sigma=0.0, p_late=0.0),),
        parts=(Part(0, 1, usage=1, upc=50, unit_value=10.0, kanban=2),),
        dist_km=np.array([[0.0, 30.0], [30.0, 0.0]]), speed_kmh=60.0)
    return cfg, net, run_twin(cfg, days=1, net=net, plan=naive_plan(net, cfg))


def test_order_up_to_by_hand(tiny):
    cfg, net, _ = tiny
    plan = naive_plan(net, cfg)
    assert production_hours(cfg, 240.0, 1440.0 + 7 * 60) == pytest.approx(17.0)
    assert plan.routes[0].lead_min == pytest.approx(145.2)
    assert order_up_to_units(cfg, net, plan, 0, 240.0) == pytest.approx(16.42 * 50)


def test_timeline_by_hand(tiny):
    _, _, tw = tiny
    pickups = [e for e in tw.log if e[1] == "pickup"]
    assert len(pickups) == 1 and pickups[0][0] == pytest.approx(291.5) and pickups[0][3] == 13
    (v,) = tw.visits
    assert (v.t_arrive, v.t_yard, v.t_door, v.t_crew, v.t_unloaded, v.t_leave) == pytest.approx(
        (321.5, 326.5, 326.5, 326.5, 347.1, 352.1))
    (b,) = tw.batches
    assert b[1:] == pytest.approx((321.5, 367.1, 13))
    (t,) = tw.trips
    assert (t.t_dispatch, t.t_end, t.km, t.n) == pytest.approx((240.0, 352.1, 60.0, 13))
    signals = [load for when, load in tw.tugger_loads if load]
    first = next(when for when, load in tw.tugger_loads if load)
    assert first == pytest.approx(420.0) and signals[0] == 1
    drops = [e[0] for e in tw.log if e[1] == "tugger_drop"]
    assert drops[0] == pytest.approx(435.0)


def test_full_day_no_starvation(tiny):
    _, _, tw = tiny
    assert tw.vehicles_by_day[0] == 800
    assert tw.downtime == []
    s = tw.state
    assert s.consumed_units[0] == 800
    assert s.units_on_hand()[0] == 100 + 100 + 13 * 50 - 800


def test_costs_by_hand(tiny):
    cfg, _, tw = tiny
    k = daily_kpis(tw)
    c = daily_costs(tw, k).iloc[0]
    assert c["transport"] == pytest.approx(60 * cfg.costs.truck_per_km)
    # one shift spanning 1.87 h is still paid the 8 h minimum, no overtime
    assert c["driver"] == pytest.approx(8 * cfg.costs.driver_per_h)
    assert c["trucks"] == pytest.approx(cfg.costs.truck_per_day)
    assert c["handling"] == pytest.approx(13 * cfg.costs.handling_per_container)
    assert c["line_down"] == 0
