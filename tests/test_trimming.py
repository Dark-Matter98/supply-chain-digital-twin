"""Capacity trimming keeps every part's end-of-cycle safety cover even.

Two suppliers share a truck. Supplier 1 is picked twice a day (waves 04:00 and
14:00), so its order must last about 10 production hours; supplier 2 is picked
every wave. The old rule trimmed whichever part had the most raw cover after
the order, which is always the less frequent one.
"""
import numpy as np

from conftest import override
from inbound_twin.network import Network, Part, Supplier
from inbound_twin.plan import build_plan, order_quantities, order_up_to_units
from inbound_twin.state import TwinState

RATE = 50.0   # units per production hour for both parts (usage 1 x 50 JPH)
UPC = 10


def _setup(base_cfg, capacity):
    cfg = override(base_cfg, {"fleet.n_trucks": 1, "fleet.capacity": capacity,
                              "inventory.ss_h": 2.0, "inventory.initial_cover_h": 2.0})
    net = Network(
        suppliers=(Supplier(1, 20.0, 0.0, 0.0, 0.0), Supplier(2, 0.0, 20.0, 0.0, 0.0)),
        parts=(Part(0, 1, usage=1, upc=UPC, unit_value=10.0, kanban=2),
               Part(1, 2, usage=1, upc=UPC, unit_value=10.0, kanban=2)),
        dist_km=np.array([[0, 20, 20], [20, 0, 28.3], [20, 28.3, 0]], dtype=float), speed_kmh=60.0)
    plan = build_plan(net, cfg, [(0, 0, (1, 2)), (0, 1, (2,)), (0, 2, (1, 2)), (0, 3, (2,))], "test")
    return cfg, net, plan, TwinState.initial(cfg, net), plan.routes[0]


def _order(cfg, net, plan, state, route):
    q = order_quantities(state, net, cfg, plan, route, route.dispatch_min)
    return {0: int(q[1][0]), 1: int(q[2][1])}


def test_trimming_evens_out_safety_cover(base_cfg):
    cfg, net, plan, state, route = _setup(base_cfg, capacity=500)
    full = _order(cfg, net, plan, state, route)                 # untrimmed needs
    assert full[0] > full[1] > 10                               # the twice-a-day part needs more

    cap = sum(full.values()) - 10                               # force 10 containers of trimming
    cfg, net, plan, state, route = _setup(base_cfg, capacity=cap)
    got = _order(cfg, net, plan, state, route)
    assert sum(got.values()) == cap
    assert got[0] < full[0] and got[1] < full[1]                # both parts give some up

    t, pos = route.dispatch_min, state.inventory_position_units()
    left = {p: (pos[p] + got[p] * UPC - (order_up_to_units(cfg, net, plan, p, t) - 2.0 * RATE)) / RATE
            for p in (0, 1)}
    assert abs(left[0] - left[1]) <= UPC / RATE + 1e-9          # within one container of each other
