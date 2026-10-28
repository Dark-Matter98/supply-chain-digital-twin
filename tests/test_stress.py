"""Stress tests change the world only inside their window, and do what they say inside it."""
import pytest

from inbound_twin import Twin, generate_network, load_config
from inbound_twin.config import override
from inbound_twin.evaluate import ROOT, load_phase2_plan
from inbound_twin.scenario import Scenario
from inbound_twin.stress import SCENARIOS


@pytest.fixture(scope="module")
def base():
    cfg = load_config(ROOT / "config" / "base.yaml")
    net = generate_network(cfg)
    return cfg, net, load_phase2_plan(cfg, net)


def stressed(cfg, name):
    return override(cfg, SCENARIOS[name]["overrides"])


def test_outside_the_window_nothing_changes(base):
    cfg, net, _ = base
    for name in SCENARIOS:
        a, b = Scenario(cfg, net, 3), Scenario(stressed(cfg, name), net, 3)
        st = stressed(cfg, name).stress
        for t in (st.start_min - 30, st.end_min + 5, (st.day - 1) * 1440 + 600):
            for s in (5, 10):
                assert a.travel_min(0, s, t, 1) == b.travel_min(0, s, t, 1)
                assert a.supplier_delay_min(s, t) == b.supplier_delay_min(s, t)


def test_supplier_outage_holds_loads_until_it_reopens(base):
    cfg, net, _ = base
    st = stressed(cfg, "supplier_outage").stress
    sc = Scenario(stressed(cfg, "supplier_outage"), net, 3)
    t = st.start_min + 100
    assert sc.supplier_delay_min(st.supplier_down, t) == pytest.approx(st.end_min - t)


def test_road_closure_multiplies_legs_on_that_corridor(base):
    cfg, net, _ = base
    st = stressed(cfg, "road_closure").stress
    a, b = Scenario(cfg, net, 3), Scenario(stressed(cfg, "road_closure"), net, 3)
    t = st.start_min + 60
    assert b.travel_min(0, st.road_supplier, t, 2) == pytest.approx(st.road_multiplier * a.travel_min(0, st.road_supplier, t, 2))
    assert b.travel_min(0, 7, t, 2) == a.travel_min(0, 7, t, 2)


def test_doors_out_caps_trucks_at_the_dock(base):
    cfg, net, plan = base
    scfg = stressed(cfg, "dock_doors")
    st = scfg.stress
    tw = Twin(scfg, net, plan, seed=4, days=st.day + 1, check_invariants=True).run()
    open_doors = scfg.plant.n_doors - st.doors_out
    for t in range(int(st.start_min) + 30, int(st.end_min), 5):   # once the doors have been taken
        at_door = sum(1 for v in tw.visits if v.t_door <= t < v.t_leave)
        assert at_door <= open_doors
