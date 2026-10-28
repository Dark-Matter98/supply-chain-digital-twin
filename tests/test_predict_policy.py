"""Predictor by hand, the value gate, the frozen zone, and the twin's re-plan actions."""
import math

import numpy as np
import pytest

from conftest import ROOT, override
from inbound_twin import Twin, generate_network, load_config
from inbound_twin.estimate import Estimator
from inbound_twin.evaluate import load_phase2_plan
from inbound_twin.network import Network, Part, Supplier
from inbound_twin.observe import Observation, TruckObs
from inbound_twin.plan import build_plan
from inbound_twin.policy import Controller, PolicySpec
from inbound_twin.predict import Candidate, Forecast, Predictor

NAN = float("nan")


@pytest.fixture(scope="module")
def tiny(base_cfg):
    cfg = override(base_cfg, {"fleet.n_trucks": 1, "fleet.dispatch_h": [4], "fleet.shift_of_wave": [0],
                              "plant.receiving_cv": 0.0})
    net = Network((Supplier(1, 30.0, 0.0, 0.0, 0.0),), (Part(0, 1, 1, 50, 10.0, 2),),
                  np.array([[0.0, 30.0], [30.0, 0.0]]), 60.0)
    return cfg, net, build_plan(net, cfg, [(0, 0, (1,))], "tiny")


def _idle_obs(t, line_full, line_open, receiving=()):
    one = lambda v: np.array([v], dtype=np.int64)  # noqa: E731
    z = one(0)
    # day 0's 04:00 trip has run (trips_started=1): the next pickup is 04:00 tomorrow
    truck = TruckObs(0, "idle", 0.0, 0, 0, 0, NAN, None, NAN, (), {}, z.copy(), 0, -math.inf, NAN, 1)
    return Observation(t, (truck,), z, z, z, z, z, z, z, one(line_full), one(line_open), z,
                       tuple(receiving), (), "run", (), (), (), ())


def test_predictor_by_hand(tiny):
    """10:00, line holds 100 units (50/h), a 2-container batch unloaded at 09:55.

    Receiving takes exactly 20 min -> supermarket 10:15 -> tugger leaves 10:30 and
    drops at 10:45 (minute 45): 200 units in all, which last until minute 240
    (14:00). The horizon runs to 16:00, so the line is short for the last 120 min.
    Nothing runs short in the first 2 h, so P(line stops within 2 h) = 0.
    """
    cfg, net, plan = tiny
    obs = _idle_obs(600.0, 1, 50, [(595.0, np.array([2]))])
    for view in ("bayes", "det"):
        est = Estimator(cfg, net, view)
        f = Predictor(cfg, net, plan, seed=1).forecast(obs, est, [Candidate()])[0]
        assert np.all(f.line_down == 120)
        assert f.p_line == 0.0
        assert f.margin_min[0] == pytest.approx(-120.0, abs=1.5)


def test_overdue_trip_is_still_coming(tiny):
    """If the truck has not started today's 04:00 trip yet (trips_started=0), the
    predictor sends it now: its load must then arrive within the horizon."""
    cfg, net, plan = tiny
    obs = _idle_obs(600.0, 1, 50, [(595.0, np.array([2]))])
    late = obs.trucks[0].__class__(**{**obs.trucks[0].__dict__, "trips_started": 0})
    obs_late = obs.__class__(**{**obs.__dict__, "trucks": (late,)})
    est = Estimator(cfg, net, "det")
    done = Predictor(cfg, net, plan, 1).forecast(obs, est, [Candidate()])[0]
    pending = Predictor(cfg, net, plan, 1).forecast(obs_late, est, [Candidate()])[0]
    assert pending.line_down.mean() < done.line_down.mean()


def test_same_scenarios_for_every_candidate(tiny):
    cfg, net, plan = tiny
    obs = _idle_obs(600.0, 1, 50, [(595.0, np.array([2]))])
    a, b = Predictor(cfg, net, plan, 1).forecast(obs, Estimator(cfg, net), [Candidate(), Candidate()])
    assert np.array_equal(a.line_down, b.line_down)


def _forecast(cost_line_down_min, action=0.0):
    return Forecast(np.array([cost_line_down_min]), np.array([0.9]), np.array([-10.0]), action, 0.9)


@pytest.mark.parametrize("gate,cand_costs,expect", [
    ("value", [1200.0, 900.0], "cheaper"),     # executes the candidate that beats the plan
    ("value", [1200.0, 1100.0], None),         # nothing beats the plan: no action
    ("always", [1200.0, 1100.0], "cheaper"),   # acts anyway, on the best candidate
])
def test_value_gate_by_hand(tiny, gate, cand_costs, expect):
    cfg, net, plan = tiny
    ctl = Controller(cfg, net, plan, 1, PolicySpec("bayes", gate, 0.1))
    cands = [Candidate(hot_part=0, label="dearer"), Candidate(courier_supplier=1, courier_q=(1,), label="cheaper")]
    rate = cfg.costs.line_down_per_min
    fs = [_forecast(1000.0 / rate)] + [_forecast(c / rate) for c in cand_costs]

    class FakePred:
        calls = 0

        def forecast(self, obs, est, cs):
            return fs[:1] if len(cs) == 1 else fs[:len(cs)]
    ctl.pred = FakePred()
    ctl.candidates = lambda obs, parts: cands
    done = []
    ctl.execute = lambda twin, obs, c: done.append(c.label)
    ctl.assess(None, "test", _idle_obs(600.0, 1, 50))
    assert done == ([expect] if expect else [])


def test_frozen_zone(tiny):
    cfg, net, plan = tiny
    ctl = Controller(cfg, net, plan, 1, PolicySpec("bayes", "value", 0.1))
    z = np.zeros(1, dtype=np.int64)
    base = dict(status="late", status_since=0.0, node=1, leg_from=0, leg_to=0, leg_start=NAN, ping=None,
                waiting_since=0.0, remaining=(), orders={}, load=z, priority=0, last_change=-math.inf,
                plant_arrival=NAN)
    assert not ctl._frozen(TruckObs(0, **base), 100.0)
    assert ctl._frozen(TruckObs(0, **{**base, "status": "yard"}), 100.0)            # at the plant
    assert ctl._frozen(TruckObs(0, **{**base, "last_change": 80.0}), 100.0)          # changed 20 min ago
    near = {**base, "status": "travel", "leg_from": 1, "leg_to": 0, "leg_start": 70.0, "node": 1}
    assert ctl._frozen(TruckObs(0, **near), 100.0)       # 30-min leg, 30 min driven: under frozen_min left


class _Scripted:
    """Leaves a not-ready supplier when the truck already has cargo; every 5 min also
    interrupts any loaded truck still waiting (the other code path)."""

    def __init__(self, from_event: bool):
        self.from_event, self.left = from_event, []

    def on_event(self, twin, kind, info):
        if self.from_event and kind == "not_ready":
            tr = twin.state.trucks[info["truck"]]
            if tr.load.sum() > 0 and twin.leave_supplier(tr.id):
                self.left.append((tr.id, info["supplier"], twin.env.now))

    def process(self, twin):
        while not self.from_event:
            yield twin.env.timeout(5.0)
            for tr in twin.state.trucks:
                if tr.status == "late" and tr.load.sum() > 0 and twin.leave_supplier(tr.id):
                    self.left.append((tr.id, tr.node, twin.env.now))
        return
        yield  # pragma: no cover


@pytest.mark.parametrize("from_event", [True, False])
def test_leave_supplier_keeps_the_books(base_cfg, from_event):
    cfg = override(base_cfg, {"network.p_supplier_late": 0.6})
    net = generate_network(cfg)
    plan = load_phase2_plan(cfg, net)
    ctl = _Scripted(from_event)
    tw = Twin(cfg, net, plan, seed=2, days=3, controller=ctl, check_invariants=True).run()
    assert ctl.left
    lefts = [(e[0], e[2], e[4]) for e in tw.log if e[1] == "left_supplier"]
    assert len(lefts) == len(ctl.left)
    for truck, sup, t in ctl.left:   # no pickup at that stop right after leaving
        assert not any(e[1] == "pickup" and e[2] == truck and e[4] == sup and t <= e[0] <= t + 1 for e in tw.log)


def test_courier_and_hot_run_conserve_material(base_cfg):
    class Scripted:
        def on_event(self, *a):
            pass

        def process(self, twin):
            yield twin.env.timeout(1440 + 480)
            q = np.zeros(len(twin.net.parts), dtype=np.int64)
            q[list(twin.net.parts_of(1))] = 2
            twin.send_courier(1, q)
            twin.arm_hot_run(0, twin.env.now + 600)
    net = generate_network(base_cfg)
    plan = load_phase2_plan(base_cfg, net)
    tw = Twin(base_cfg, net, plan, seed=1, days=3, controller=Scripted(), check_invariants=True).run()
    assert any(e[1] == "courier_received" for e in tw.log)


def test_same_seed_same_decisions():
    cfg = load_config(ROOT / "config" / "base.yaml", ROOT / "config" / "drifting.yaml")
    cfg = override(cfg, {"inventory.ss_h": 1.0})
    net = generate_network(cfg)
    plan = load_phase2_plan(cfg, net)
    runs = []
    for _ in range(2):
        ctl = Controller(cfg, net, plan, 3, PolicySpec("bayes", "value", 0.2))
        tw = Twin(cfg, net, plan, seed=3, days=3, controller=ctl, check_invariants=True).run()
        runs.append(([(d["t"], d["label"], d["executed"]) for d in ctl.decisions], tw.actions))
    assert runs[0] == runs[1] and runs[0][0]
