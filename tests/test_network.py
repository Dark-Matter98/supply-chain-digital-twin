import math

import numpy as np

from inbound_twin import generate_network, naive_plan
from inbound_twin.network import kanban_cards


def test_deterministic_for_seed(base_cfg):
    a, b = generate_network(base_cfg), generate_network(base_cfg)
    assert np.array_equal(a.dist_km, b.dist_km)
    assert a.parts == b.parts and a.suppliers == b.suppliers
    c = generate_network(base_cfg, seed=base_cfg.seed_network + 1)
    assert not np.array_equal(a.dist_km, c.dist_km)


def test_distances_are_sane(base_cfg):
    net = generate_network(base_cfg)
    d = net.dist_km
    nc = base_cfg.network
    assert d.shape == (nc.n_suppliers + 1,) * 2
    assert np.allclose(d, d.T) and np.allclose(np.diag(d), 0)
    radial = d[0, 1:] / nc.circuity
    assert (radial >= nc.radius_km[0] - 1e-9).all() and (radial <= nc.radius_km[1] + 1e-9).all()
    n = len(d)  # triangle inequality (Euclidean x constant factor keeps it)
    assert all(d[i, k] <= d[i, j] + d[j, k] + 1e-9 for i in range(n) for j in range(n) for k in range(n))


def test_parts_and_kanban(base_cfg):
    net = generate_network(base_cfg)
    nc = base_cfg.network
    assert len(net.parts) == nc.n_suppliers * nc.parts_per_supplier
    for p in net.parts:
        cph = base_cfg.line.jph * p.usage / p.upc
        assert 0.4 <= cph <= 1.3      # rounding of units/container moves it a little
        assert nc.unit_value_clip[0] <= p.unit_value <= nc.unit_value_clip[1]
        assert p.kanban >= 2
    assert kanban_cards(1.0 / 60, 30, 0.5) == math.ceil(1 / 60 * 45 * 1.5) + 1


def test_naive_plan_covers_every_supplier_once_per_wave(base_cfg):
    net = generate_network(base_cfg)
    plan = naive_plan(net, base_cfg)
    waves = base_cfg.fleet.dispatch_h
    assert len(plan.routes) == base_cfg.fleet.n_trucks * len(waves)
    for h in waves:
        visited = [s for r in plan.routes if r.dispatch_min == h * 60 for s in r.suppliers]
        assert sorted(visited) == list(range(1, base_cfg.network.n_suppliers + 1))
