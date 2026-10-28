"""The drifting-regime world: corridor congestion and incidents."""
import numpy as np

from conftest import ROOT, override
from inbound_twin import generate_network, load_config
from inbound_twin.scenario import STEPS_PER_DAY, Scenario, corridor_path


def test_off_by_default(base_cfg):
    assert not corridor_path(base_cfg, 1, 0, 3).any()
    net = generate_network(base_cfg)
    assert {s.p_late for s in net.suppliers} == {base_cfg.network.p_supplier_late}
    assert Scenario(base_cfg, net, 1).arc_offset(0, 3, 600.0) == 0.0


def test_ar1_statistics(base_cfg):
    cfg = override(base_cfg, {"traffic.corridor_phi": 0.9, "traffic.corridor_sigma": 0.2})
    paths = np.array([corridor_path(cfg, 7, day, 1) for day in range(400)])
    x = paths + 0.2 ** 2 / 2                      # undo the mean-one shift
    assert abs(x.mean()) < 0.02
    assert abs(x.std() - 0.2) < 0.02
    lag1 = np.corrcoef(x[:, :-1].ravel(), x[:, 1:].ravel())[0, 1]
    assert abs(lag1 - 0.9) < 0.03
    assert abs(np.exp(paths).mean() - 1.0) < 0.03  # the multiplier has mean one


def test_incident_rate_and_size(base_cfg):
    cfg = override(base_cfg, {"traffic.incidents_per_day": 0.1})
    days = 3000
    onsets, jumps = 0, []
    for day in range(days):
        p = corridor_path(cfg, 3, day, 2)
        on = p > 0
        onsets += int(on[0]) + int(np.sum(on[1:] & ~on[:-1]))
        jumps += list(p[on])
    assert abs(onsets / days - 0.1) < 0.015      # overlapping incidents can merge; small effect
    assert min(jumps) >= np.log(1.4) - 1e-9


def test_drifting_overlay_loads(base_cfg):
    cfg = load_config(ROOT / "config" / "base.yaml", ROOT / "config" / "drifting.yaml")
    assert cfg.traffic.corridor_sigma > 0 and cfg.network.p_late_spread > 0
    net = generate_network(cfg)
    base = generate_network(base_cfg)
    assert np.array_equal(net.dist_km, base.dist_km)          # same plant and suppliers
    assert len({round(s.p_late, 6) for s in net.suppliers}) > 1
    assert len(paths := corridor_path(cfg, 1, 0, 1)) == STEPS_PER_DAY and paths.std() > 0
