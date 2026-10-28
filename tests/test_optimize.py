"""Phase 2 optimizer: model data, optimality on a small case, feasibility, determinism."""
from itertools import permutations

import pytest

from conftest import override, run_twin
from inbound_twin import generate_network, naive_plan
from inbound_twin.kpi import daily_kpis
from inbound_twin.optimize import (WaveSolver, frequency_options, optimize_plan, planning_cost,
                                   solve_wave, wave_model)
from inbound_twin.plan import visit_volumes, wave_index


def test_frequency_options():
    assert frequency_options(4) == ((0, 1, 2, 3), (0, 2), (1, 3), (0,), (1,), (2,), (3,))


def test_visit_volumes_by_hand(base_cfg):
    """One part at 1 container per production hour (usage 1, 50 units/container, 50 JPH).
    Loads land at 08:00 and 18:00. The 08:00 load covers 08-18 (10 h), the 18:00
    load covers 18-22 + 06-08 (6 h)."""
    from inbound_twin.network import Network, Part, Supplier
    import numpy as np
    net = Network((Supplier(1, 30.0, 0.0, 0.0, 0.0),), (Part(0, 1, 1, 50, 10.0, 2),),
                  np.array([[0.0, 30.0], [30.0, 0.0]]), 60.0)
    v = visit_volumes(net, base_cfg, 1, {0: 8 * 60.0, 2: 18 * 60.0})
    assert v == pytest.approx({0: 10.0, 2: 6.0})


def _partitions(items, max_blocks):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in _partitions(rest, max_blocks):
        for i in range(len(part)):
            yield part[:i] + [[first, *part[i]]] + part[i + 1:]
        if len(part) < max_blocks:
            yield [[first], *part]


def _brute_force(m, n_vehicles):
    """Exact optimum of the same integer model: every partition, every visit order."""
    best = None
    idx = list(range(1, len(m.nodes)))
    for blocks in _partitions(idx, n_vehicles):
        total = 0
        for block in blocks:
            if sum(m.demand[i] for i in block) > m.capacity:
                break
            options = []
            for order in permutations(block):
                path = (0, *order, 0)
                if sum(m.time[a][b] for a, b in zip(path, path[1:])) <= m.time_budget:
                    options.append(sum(m.cost[a][b] for a, b in zip(path, path[1:])))
            if not options:
                break
            total += min(options) + m.trip_cost
        else:
            best = total if best is None else min(best, total)
    return best


def test_ortools_matches_brute_force_on_small_case(base_cfg):
    cfg = override(base_cfg, {"network.n_suppliers": 6, "network.parts_per_supplier": 1,
                              "fleet.n_trucks": 3})
    net = generate_network(cfg)
    demands = {1: 5.0, 2: 6.5, 3: 7.0, 4: 8.0, 5: 9.5, 6: 10.0}
    for wave in (0, 2):
        m = wave_model(net, cfg, wave, demands)
        exact = _brute_force(m, 3)
        sol = solve_wave(net, cfg, wave, demands, 3, "full")
        assert exact is not None and sol is not None
        assert sol.cost == pytest.approx(exact / 100)


def _check_feasible(net, cfg, wave, demands, sol):
    m = wave_model(net, cfg, wave, demands)
    pos = {s: i for i, s in enumerate(m.nodes)}
    seen = [s for r in sol.routes for s in r]
    assert sorted(seen) == sorted(demands)                  # every supplier exactly once
    for r in sol.routes:
        path = (0, *(pos[s] for s in r), 0)
        assert sum(m.demand[pos[s]] for s in r) <= m.capacity
        assert sum(m.time[a][b] for a, b in zip(path, path[1:])) <= m.time_budget


def test_default_waves_feasible_and_deterministic(base_cfg):
    net = generate_network(base_cfg)
    demands = WaveSolver(net, base_cfg).demands({s.id: (0, 1, 2, 3) for s in net.suppliers})
    for wave, dem in demands.items():
        a = solve_wave(net, base_cfg, wave, dem, base_cfg.fleet.n_trucks, "fast")
        b = solve_wave(net, base_cfg, wave, dem, base_cfg.fleet.n_trucks, "fast")
        assert a is not None and a == b
        _check_feasible(net, base_cfg, wave, dem, a)


def test_planning_km_matches_twin_for_naive(base_cfg):
    net = generate_network(base_cfg)
    plan = naive_plan(net, base_cfg)
    tw = run_twin(base_cfg, seed=1, days=4, net=net, plan=plan, check=False)
    assert planning_cost(net, base_cfg, plan)["km"] == pytest.approx(daily_kpis(tw)["km"].mean())


@pytest.fixture(scope="module")
def small_optimized(base_cfg):
    cfg = override(base_cfg, {"network.n_suppliers": 6, "fleet.n_trucks": 3,
                              "planning.solution_limit": 200})
    net = generate_network(cfg)
    return cfg, net, optimize_plan(net, cfg, max_sweeps=2)


def test_optimized_plan_respects_its_frequencies(small_optimized):
    cfg, net, res = small_optimized
    visits: dict[int, list[int]] = {}
    for r in res.plan.routes:
        for s in r.suppliers:
            visits.setdefault(s, []).append(wave_index(cfg, r.dispatch_min))
    assert {s: tuple(sorted(w)) for s, w in visits.items()} == res.freq
    assert all(r.lead_min > 0 for r in res.plan.routes)
    assert res.plan.fleet <= cfg.fleet.n_trucks
    assert res.planning["total"] <= planning_cost(net, cfg, naive_plan(net, cfg))["total"] + 1e-6


def test_optimized_plan_runs_in_twin(small_optimized):
    cfg, net, res = small_optimized
    tw = run_twin(cfg, seed=2, days=5, net=net, plan=res.plan, check=True)
    assert tw.vehicles_by_day[1:].min() > 0
