"""Material balance and resource invariants.

The 30-day fixture runs with check_invariants=True, so Twin.assert_invariants()
already ran every 5 simulated minutes (non-negative stock, exact material
balance, kanban cards conserved, resource counts within capacity). These tests
re-check the end state and the interval records independently.
"""
import math

import numpy as np


def test_material_balance_end_state(default_run_30):
    s = default_run_30.state
    lhs = s.initial_units + s.picked * s.upc
    rhs = s.consumed_units + s.units_on_hand()
    assert np.array_equal(lhs, rhs)
    assert s.picked.sum() > 0 and s.consumed_units.sum() > 0


def test_consumption_matches_vehicles_built(default_run_30):
    s = default_run_30.state
    assert np.array_equal(s.consumed_units, s.usage * s.vehicles_built)
    assert s.vehicles_built == default_run_30.vehicles_by_day.sum()


def _max_concurrency(intervals):
    events = sorted([(a, 1) for a, b in intervals] + [(b, -1) for a, b in intervals])
    level = peak = 0
    for _, dv in events:          # at equal times the -1 sorts first: release before acquire
        level += dv
        peak = max(peak, level)
    return peak


def test_resource_capacities_from_records(default_run_30):
    tw = default_run_30
    horizon = tw.days * 1440.0
    end = lambda t: horizon if math.isnan(t) else t  # noqa: E731
    doors = [(v.t_door, end(v.t_leave)) for v in tw.visits if not math.isnan(v.t_door)]
    crews = [(v.t_crew, end(v.t_unloaded)) for v in tw.visits if not math.isnan(v.t_crew)]
    assert _max_concurrency(doors) <= tw.cfg.plant.n_doors
    assert _max_concurrency(crews) <= tw.cfg.plant.n_crews
    assert max(n for _, n in tw.tugger_loads) <= tw.cfg.tugger.capacity
    assert max(t.n for t in tw.trips) <= tw.cfg.fleet.capacity
    for k in range(tw.cfg.fleet.n_trucks):     # a truck runs one trip at a time
        assert _max_concurrency([(t.t_dispatch, t.t_end) for t in tw.trips if t.truck == k]) <= 1


def test_visit_timestamps_ordered(default_run_30):
    for v in default_run_30.visits:
        times = [t for t in (v.t_arrive, v.t_yard, v.t_door, v.t_crew, v.t_unloaded, v.t_leave)
                 if not math.isnan(t)]
        assert times == sorted(times)


def test_line_never_runs_outside_production_window(default_run_30):
    tw = default_run_30
    start, end = tw.cfg.line.prod_start_h * 60, tw.cfg.line.prod_end_h * 60
    max_per_day = round((end - start) / tw.cfg.line.takt_min)
    assert (tw.vehicles_by_day <= max_per_day).all()
    for a, b, _ in tw.downtime:
        day0 = (a // 1440) * 1440
        assert start - 1e-6 <= a - day0 and b - day0 <= end + 1e-6


def test_mid_run_snapshot_is_complete_and_independent(base_cfg):
    from inbound_twin import Twin, generate_network, naive_plan
    net = generate_network(base_cfg)
    tw = Twin(base_cfg, net, naive_plan(net, base_cfg), seed=2, days=2)
    taken = {}

    def capture():
        yield tw.env.timeout(1440 + 11 * 60)   # day 1, 11:00: trucks out, line running
        taken["snap"] = tw.state.snapshot()

    tw.env.process(capture())
    tw.run()
    snap = taken["snap"]
    assert np.array_equal(snap.initial_units + snap.picked * snap.upc,
                          snap.consumed_units + snap.units_on_hand())
    assert snap.in_transit.sum() + snap.on_order.sum() > 0      # caught the plant mid-flow
    snap.supermarket[:] = -1
    snap.trucks[0].load[:] = 99
    assert (tw.state.supermarket >= 0).all() and tw.state.trucks[0].load.max() < 99


def test_same_seed_is_deterministic(base_cfg):
    from conftest import run_twin
    from inbound_twin.kpi import daily_kpis
    a = daily_kpis(run_twin(base_cfg, seed=3, days=4, check=False))
    b = daily_kpis(run_twin(base_cfg, seed=3, days=4, check=False))
    c = daily_kpis(run_twin(base_cfg, seed=4, days=4, check=False))
    assert a.equals(b)
    assert not a.equals(c)
