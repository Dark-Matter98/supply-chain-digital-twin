"""Phase 2: static milk-run optimization.

Decisions for the daily template:
  - pickup frequency per supplier (which dispatch waves visit it),
  - routes per wave (which truck visits which suppliers, in what order),
  - fleet size.

Routing per wave is a capacitated VRP with a route-duration limit, solved with
OR-Tools: savings construction, then greedy descent (fast) or guided local
search stopped by a solution limit (full). Visits are optional at a huge
penalty, so a solve never stalls and a dropped visit signals infeasibility.
Results are deterministic. Frequencies come from a greedy local search on a
deterministic planning cost; fleet size from enumeration. This is a heuristic
decomposition, not a joint optimum.

The planning cost leaves out line-down: it is the deterministic promise that
the twin then checks under uncertainty.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from .config import Config
from .costs import driver_cost, shift_hours
from .network import Network
from .plan import (DEFAULT_LEAD_MIN, Plan, arrival_minute, build_plan, plan_volumes,
                   route_timing, visit_volumes, wave_index)
from .scenario import rush_factor

TIME_SCALE = 10     # OR-Tools works in integers: tenths of a minute,
COST_SCALE = 100    # cents,
LOAD_SCALE = 10     # and tenths of a container (expected volumes are fractional)
INFEASIBLE = 1e7    # search score floor for frequency assignments with no feasible routing
DROP_COST = 10**9   # cents for leaving a visit out: far above any real route cost (~1e5)


@dataclass(frozen=True)
class WaveSolution:
    routes: tuple[tuple[int, ...], ...]  # supplier sequences, one per truck used
    cost: float                          # solver objective, USD


@dataclass
class PlanResult:
    plan: Plan
    freq: dict[int, tuple[int, ...]]
    planning: dict[str, float]
    log: list[str] = field(default_factory=list)


def frequency_options(n_waves: int) -> tuple[tuple[int, ...], ...]:
    """Wave sets a supplier can be visited on: every wave, every other wave, one wave."""
    opts = [tuple(range(n_waves))]
    if n_waves % 2 == 0 and n_waves > 2:
        opts += [tuple(range(0, n_waves, 2)), tuple(range(1, n_waves, 2))]
    opts += [(w,) for w in range(n_waves)]
    return tuple(opts)


def visit_demands(net: Network, cfg: Config, supplier: int, waves: tuple[int, ...]) -> dict[int, float]:
    """Expected containers per visit, using the default lead (routes are not known yet)."""
    return visit_volumes(net, cfg, supplier, {w: arrival_minute(cfg, w, DEFAULT_LEAD_MIN) for w in waves})


def route_capacity(cfg: Config) -> float:
    """Containers a route may be planned to carry (capacity x max_fill)."""
    return cfg.fleet.capacity * cfg.planning.max_fill


def wave_budget_min(cfg: Config, wave: int) -> float:
    """Minutes from dispatch until the door must be released: before the truck's
    next wave (or the end of the operating window), minus the planning slack."""
    disp = cfg.fleet.dispatch_h
    nxt = disp[wave + 1] * 60.0 if wave + 1 < len(disp) else disp[0] * 60.0 + 1440.0
    end = min(nxt, cfg.plant.operating_window_h[1] * 60.0)
    return end - disp[wave] * 60.0 - cfg.planning.slack_min


def avg_rush(cfg: Config, t0: float, t1: float) -> float:
    return float(np.mean([rush_factor(cfg, t) for t in np.arange(t0, t1, 1.0)]))


@dataclass(frozen=True)
class WaveModel:
    """The integer data OR-Tools sees for one wave (node 0 = plant)."""
    nodes: tuple[int, ...]
    cost: tuple[tuple[int, ...], ...]   # cents: km x $/km + driving minutes x driver $/h
    time: tuple[tuple[int, ...], ...]   # tenths of a minute: driving + service at the from-node
    demand: tuple[int, ...]             # tenths of a container
    capacity: int                       # tenths of a container
    time_budget: int                    # tenths of a minute for driving + service
    trip_cost: int                      # cents per truck used: the fixed part of the plant visit


def wave_model(net: Network, cfg: Config, wave: int, demands: dict[int, float]) -> WaveModel:
    so, pc, c = cfg.supplier_ops, cfg.plant, cfg.costs
    suppliers = sorted(demands)
    dispatch = cfg.fleet.dispatch_h[wave] * 60.0
    budget = wave_budget_min(cfg, wave)
    rush = avg_rush(cfg, dispatch, dispatch + budget)
    # Unload time is linear in containers, so each pickup carries its own share
    # of it; only the per-trip part of the plant visit is subtracted up front.
    plant_fixed = pc.gate_min + cfg.planning.dock_wait_min + pc.unload_setup_min + pc.dock_out_min
    nodes = (0, *suppliers)
    service = [0.0] + [so.load_fixed_min
                       + (so.load_per_container_min + pc.unload_per_container_min) * demands[s]
                       + net.suppliers[s - 1].p_late * cfg.network.supplier_late_mean_min
                       for s in suppliers]
    n = len(nodes)
    drive = [[net.nominal_min(nodes[i], nodes[j]) * rush for j in range(n)] for i in range(n)]
    return WaveModel(
        nodes=nodes,
        cost=tuple(tuple(round((float(net.dist_km[nodes[i], nodes[j]]) * c.truck_per_km
                                + drive[i][j] / 60.0 * c.driver_per_h) * COST_SCALE) for j in range(n))
                   for i in range(n)),
        time=tuple(tuple(round((drive[i][j] + service[i]) * TIME_SCALE) for j in range(n))
                   for i in range(n)),
        demand=(0, *(round(demands[s] * LOAD_SCALE) for s in suppliers)),
        capacity=int(math.floor(route_capacity(cfg) * LOAD_SCALE + 1e-9)),
        time_budget=int((budget - plant_fixed) * TIME_SCALE),
        trip_cost=round(plant_fixed / 60.0 * c.driver_per_h * COST_SCALE))


def _guard(started: float, limit_s: float, wave: int) -> None:
    """Refuse results that a time limit cut short: they would depend on CPU speed.
    (OR-Tools reports the same status for time and solution limits, so use the clock.)"""
    if time.perf_counter() - started >= 0.95 * limit_s:
        raise RuntimeError(f"wave {wave}: hit the {limit_s:.0f} s safety time limit; the result "
                           f"would depend on CPU speed, so it is not used")


def _solve_model(m: WaveModel, n_vehicles: int, metaheuristic, solution_limit: int | None,
                 limit_s: float, wave: int) -> WaveSolution | None:
    """One OR-Tools solve with every visit optional at DROP_COST. Returns None if
    the best solution found still drops a visit."""
    nodes, n = m.nodes, len(m.nodes)
    manager = pywrapcp.RoutingIndexManager(n, n_vehicles, 0)
    model = pywrapcp.RoutingModel(manager)

    def node(index: int) -> int:
        return manager.IndexToNode(index)

    cost_cb = model.RegisterTransitCallback(lambda i, j: m.cost[node(i)][node(j)])
    model.SetArcCostEvaluatorOfAllVehicles(cost_cb)
    model.SetFixedCostOfAllVehicles(m.trip_cost)
    time_cb = model.RegisterTransitCallback(lambda i, j: m.time[node(i)][node(j)])
    model.AddDimension(time_cb, 0, m.time_budget, True, "Time")
    load_cb = model.RegisterUnaryTransitCallback(lambda i: m.demand[node(i)])
    model.AddDimensionWithVehicleCapacity(load_cb, 0, [m.capacity] * n_vehicles, True, "Load")
    # Optional visits keep the model always feasible, so construction never
    # stalls on a tight wave (it used to search for 3-25 s); a dropped visit in
    # the answer is how infeasibility shows up.
    for k in range(1, n):
        model.AddDisjunction([manager.NodeToIndex(k)], DROP_COST)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.SAVINGS
    params.local_search_metaheuristic = metaheuristic
    if solution_limit:
        params.solution_limit = solution_limit
    params.time_limit.FromMilliseconds(int(limit_s * 1000))
    started = time.perf_counter()
    sol = model.SolveWithParameters(params)
    _guard(started, limit_s, wave)
    if sol is None:
        return None
    routes, served = [], 0
    for v in range(n_vehicles):
        idx, seq = model.Start(v), []
        idx = sol.Value(model.NextVar(idx))
        while not model.IsEnd(idx):
            seq.append(nodes[node(idx)])
            idx = sol.Value(model.NextVar(idx))
        if seq:
            routes.append(tuple(seq))
            served += len(seq)
    if served < n - 1:
        return None
    return WaveSolution(tuple(routes), sol.ObjectiveValue() / COST_SCALE)


def solve_wave(net: Network, cfg: Config, wave: int, demands: dict[int, float],
               n_vehicles: int, mode: str = "full") -> WaveSolution | None:
    """Capacitated VRP with a duration limit for one wave. None if no feasible
    routing was found (a heuristic verdict, not a proof).

    mode "fast": greedy descent to a local optimum (about 0.01 s). If that drops
                 a visit, guided local search gets a chance before "infeasible".
    mode "full": guided local search stopped by planning.solution_limit (2-3 s).
    Both are deterministic; the time limits are only a safety net that raises.
    """
    if mode not in ("fast", "full"):
        raise ValueError(f"mode must be 'fast' or 'full', got {mode!r}")
    if not demands:
        return WaveSolution((), 0.0)
    m = wave_model(net, cfg, wave, demands)
    if max(m.demand) > m.capacity:
        return None
    meta = routing_enums_pb2.LocalSearchMetaheuristic
    if mode == "fast":
        sol = _solve_model(m, n_vehicles, meta.GREEDY_DESCENT, None, cfg.planning.fast_time_limit_s, wave)
        if sol is not None:
            return sol
    return _solve_model(m, n_vehicles, meta.GUIDED_LOCAL_SEARCH, cfg.planning.solution_limit,
                        cfg.planning.time_limit_s, wave)


class WaveSolver:
    """Solves the waves of a frequency assignment, caching each wave's VRP."""

    def __init__(self, net: Network, cfg: Config):
        self.net, self.cfg = net, cfg
        self.cache: dict[tuple, WaveSolution | None] = {}
        self.solves = 0

    def demands(self, freq: dict[int, tuple[int, ...]]) -> dict[int, dict[int, float]]:
        out: dict[int, dict[int, float]] = {w: {} for w in range(len(self.cfg.fleet.dispatch_h))}
        for s, waves in freq.items():
            for w, v in visit_demands(self.net, self.cfg, s, waves).items():
                out[w][s] = v
        return out

    def solve(self, freq, n_vehicles: int, mode: str) -> dict[int, WaveSolution] | None:
        out = {}
        for w, dem in self.demands(freq).items():
            key = (w, tuple(sorted((s, round(v, 6)) for s, v in dem.items())), n_vehicles, mode)
            if key not in self.cache:
                self.cache[key] = solve_wave(self.net, self.cfg, w, dem, n_vehicles, mode)
                self.solves += 1
            if self.cache[key] is None:
                return None
            out[w] = self.cache[key]
        return out


def assemble(net: Network, cfg: Config, freq, waves: dict[int, WaveSolution], name: str) -> Plan:
    """Give each wave's routes to trucks and build the plan.

    Within a driver shift, a truck that already ran an earlier wave is reused
    first (no extra paid shift), and reused trucks get the shortest routes so
    the overtime they add is as small as possible.
    """
    demands = WaveSolver(net, cfg).demands(freq)
    triples = []
    for shift in sorted(set(cfg.fleet.shift_of_wave)):
        for w in [w for w in sorted(waves) if cfg.fleet.shift_of_wave[w] == shift]:
            dispatch = cfg.fleet.dispatch_h[w] * 60.0
            dur = {seq: route_timing(net, cfg, seq, dispatch, demands[w]).door_release_min
                   for seq in waves[w].routes}
            # Shortest routes take the lowest truck numbers, which an earlier
            # wave of the same shift has already used.
            for truck, seq in enumerate(sorted(waves[w].routes, key=lambda r: (dur[r], r))):
                triples.append((truck, w, seq))
    return build_plan(net, cfg, triples, name)


def planning_cost(net: Network, cfg: Config, plan: Plan) -> dict[str, float]:
    """Deterministic daily cost the plan promises (no line-down). Approximations:
    holding uses production-time averages and full lead time in transit; space
    sums each part's own peak, so it is an upper bound."""
    c = cfg.costs
    triples = [(r.truck, wave_index(cfg, r.dispatch_min), r.suppliers) for r in plan.routes]
    vol = plan_volumes(net, cfg, triples, [r.lead_min for r in plan.routes])
    km = 0.0
    spans: dict[tuple[int, int], list[float]] = {}
    for (truck, w, seq), r in zip(triples, plan.routes, strict=True):
        t = route_timing(net, cfg, seq, r.dispatch_min, {s: vol[(s, w)] for s in seq})
        km += t.km
        lo_hi = spans.setdefault((truck, cfg.fleet.shift_of_wave[w]), [math.inf, -math.inf])
        lo_hi[0] = min(lo_hi[0], r.dispatch_min)
        lo_hi[1] = max(lo_hi[1], r.dispatch_min + t.door_release_min)
    n_shifts, ot_h = shift_hours([(hi - lo) / 60.0 for lo, hi in spans.values()], c.driver_min_paid_h)

    positions, capital = 0.0, 0.0
    leads = {(s, wave_index(cfg, r.dispatch_min)): r.lead_min for r in plan.routes for s in r.suppliers}
    for p in net.parts:
        rate = p.usage * cfg.line.jph
        visits = {w: v for (s, w), v in vol.items() if s == p.supplier}
        share = rate / p.upc / sum(cfg.line.jph * q.usage / q.upc for q in net.parts if q.supplier == p.supplier)
        g = np.array([v * share / (rate / p.upc) for v in visits.values()])  # production hours per visit
        cycle = rate * float((g ** 2).sum()) / (2 * g.sum())
        pipeline = sum(v * share * p.upc * leads[(p.supplier, w)] / 1440.0 for w, v in visits.items())
        on_hand = rate * cfg.inventory.ss_h + cycle + pipeline + (p.kanban - 0.5) * p.upc
        capital += on_hand * p.unit_value * c.carrying_rate_per_year / 365.0
        positions += (rate * cfg.inventory.ss_h + rate * g.max()) / p.upc
    positions += sum(p.kanban for p in net.parts)
    containers = sum(cfg.line.jph * p.usage / p.upc for p in net.parts) * cfg.line.prod_hours

    out = {
        "transport": km * c.truck_per_km,
        "driver": driver_cost(n_shifts, ot_h, c),
        "trucks": plan.fleet * c.truck_per_day,
        "handling": containers * c.handling_per_container,
        "holding_capital": capital,
        "holding_space": math.ceil(positions) * c.space_per_position_day,
    }
    out["total"] = sum(out.values())
    out.update({"km": km, "shifts": n_shifts, "overtime_h": ot_h, "fleet": plan.fleet,
                "trips": len(plan.routes), "visits": sum(len(r.suppliers) for r in plan.routes)})
    return out


def best_fleet(net: Network, cfg: Config, solver: WaveSolver, freq, name: str,
               log: list[str]) -> PlanResult | None:
    """Try fleet sizes from the configured maximum down while every wave stays feasible."""
    best = None
    for fleet in range(cfg.fleet.n_trucks, 0, -1):
        waves = solver.solve(freq, fleet, "full")
        if waves is None:
            log.append(f"fleet {fleet}: infeasible")
            break
        plan = assemble(net, cfg, freq, waves, name)
        cost = planning_cost(net, cfg, plan)
        log.append(f"fleet {fleet}: uses {plan.fleet} trucks, planning cost ${cost['total']:,.0f}/day")
        if best is None or cost["total"] < best.planning["total"] - 1e-6:
            best = PlanResult(plan, dict(freq), cost)
    if best is not None:
        best.log = log
    return best


def routing_only_plan(net: Network, cfg: Config) -> PlanResult:
    """Every supplier on every wave (as the naive plan); only routes and fleet change."""
    every = frequency_options(len(cfg.fleet.dispatch_h))[0]
    freq = {s.id: every for s in net.suppliers}
    result = best_fleet(net, cfg, WaveSolver(net, cfg), freq, "routing", [])
    if result is None:
        raise RuntimeError("no feasible routing with every supplier on every wave")
    return result


def optimize_plan(net: Network, cfg: Config, max_sweeps: int = 3, name: str = "optimized") -> PlanResult:
    """Frequencies by greedy local search (fast solves), then fleet size (full solves)."""
    solver = WaveSolver(net, cfg)
    opts = frequency_options(len(cfg.fleet.dispatch_h))
    freq = {s.id: opts[0] for s in net.suppliers}
    n_trucks = cfg.fleet.n_trucks
    log: list[str] = []

    cap = route_capacity(cfg) * n_trucks

    def evaluate(f) -> float:
        waves = solver.solve(f, n_trucks, "fast")
        if waves is None:
            # Infeasible: rank by how far over capacity the waves are, so the
            # search can climb toward feasibility one supplier at a time.
            over = sum(max(0.0, sum(d.values()) - cap) for d in solver.demands(f).values())
            return INFEASIBLE + over * 1e4
        return planning_cost(net, cfg, assemble(net, cfg, f, waves, "eval"))["total"]

    best = evaluate(freq)
    log.append("start (every wave): " + ("infeasible" if best >= INFEASIBLE else f"${best:,.0f}/day"))
    for sweep in range(max_sweeps):
        improved = False
        for s in sorted(freq):
            for opt in opts:
                if opt == freq[s]:
                    continue
                trial = {**freq, s: opt}
                cost = evaluate(trial)
                if cost < best - 1e-6:
                    shown = "still infeasible" if cost >= INFEASIBLE else f"${cost:,.0f}/day"
                    log.append(f"sweep {sweep}: supplier {s} {freq[s]} -> {opt}: {shown}")
                    freq, best, improved = trial, cost, True
        if not improved:
            break
    log.append(f"frequency search done after {solver.solves} wave solves")
    result = best_fleet(net, cfg, solver, freq, name, log)
    if result is None:
        raise RuntimeError("optimized frequencies have no feasible fleet")
    return result
