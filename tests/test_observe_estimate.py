"""Observation layer (no truth leaks) and the estimator."""
import dataclasses
import math

import numpy as np
import pytest

from conftest import ROOT
from inbound_twin import Twin, generate_network, load_config
from inbound_twin.estimate import Estimator
from inbound_twin.evaluate import load_phase2_plan
from inbound_twin.observe import Observation, TruckObs, observe
from inbound_twin.scenario import corridor_path, rush_factor

TRUTH_FIELDS = {"leg_end_true"}


def test_observation_has_no_truth_fields():
    names = {f.name for f in dataclasses.fields(Observation)} | {f.name for f in dataclasses.fields(TruckObs)}
    assert not names & TRUTH_FIELDS


class _PingCheck:
    def __init__(self, cfg):
        self.cfg, self.pings = cfg, []

    def on_event(self, *a):
        pass

    def process(self, twin):
        while True:
            yield twin.env.timeout(7.0)
            for tr in observe(twin).trucks:
                if tr.ping is not None:
                    assert tr.leg_start + self.cfg.dynamic.ping_min <= tr.ping.t <= twin.env.now
                    assert 0.01 <= tr.ping.progress <= 0.99
                    self.pings.append((tr.id, tr.ping.t, tr.ping.progress))


def test_pings_are_sane_and_deterministic():
    cfg = load_config(ROOT / "config" / "base.yaml", ROOT / "config" / "drifting.yaml")
    net = generate_network(cfg)
    plan = load_phase2_plan(cfg, net)
    a, b = _PingCheck(cfg), _PingCheck(cfg)
    Twin(cfg, net, plan, seed=4, days=2, controller=a).run()
    Twin(cfg, net, plan, seed=4, days=2, controller=b).run()
    assert a.pings and a.pings == b.pings


def _obs(t, legs):
    z = np.zeros(1, dtype=np.int64)
    return Observation(t, (), z, z, z, z, z, z, z, z, z, z, (), (), "off", tuple(legs), (), (), ())


def test_filter_predictive_coverage_on_synthetic_legs():
    """Legs on one corridor drawn from the world's own process; before each leg is
    ingested, its 80 % predictive interval must cover it about 80 % of the time."""
    cfg = load_config(ROOT / "config" / "base.yaml", ROOT / "config" / "drifting.yaml")
    net = generate_network(cfg)
    est = Estimator(cfg, net, "bayes")
    rng = np.random.default_rng(0)
    s, sig = 1, net.suppliers[0].travel_sigma
    nominal = net.nominal_min(0, s)
    hits, legs = [], []
    for day in range(40):
        path = corridor_path(cfg, 11, day, s)
        for t in np.arange(day * 1440 + 300, day * 1440 + 1300, 45.0):
            offset = path[int((t % 1440) // 15)]
            dur = nominal * rush_factor(cfg, t) * math.exp(offset + sig * rng.standard_normal() - sig ** 2 / 2)
            est.ingest(_obs(t, legs))
            h = est._h(0, s)
            mu = math.log(nominal * rush_factor(cfg, t)) + float(h @ est.m) - sig ** 2 / 2
            sd = math.sqrt(float(h @ est.P @ h) + sig ** 2)
            z = (math.log(dur) - mu) / sd
            hits.append(abs(z) <= 1.2816)
            legs.append((0, 0, s, t, t + dur))
    assert 0.74 <= np.mean(hits) <= 0.86


def test_filter_is_inert_without_corridor_drift(base_cfg):
    net = generate_network(base_cfg)
    est = Estimator(base_cfg, net, "bayes")
    est.ingest(_obs(600.0, [(0, 0, 1, 500.0, 590.0)]))
    assert not est.m.any() and not est.P.any()


def test_remaining_time_mean_matches_samples():
    from inbound_twin.estimate import LegPosterior
    post = LegPosterior(mu=math.log(60), sd=0.3, elapsed=40.0)
    u = np.random.default_rng(1).random(200_000)
    samples = Estimator.sample_remaining(post, u)
    assert samples.min() >= 0
    assert samples.mean() == pytest.approx(Estimator.mean_remaining(post), rel=0.01)
