"""Milk-run plans, route timing, and the (R,S) pickup-quantity rule.

A plan is a daily template of routes. Each route carries its planned lead time
(dispatch until the load is in the supermarket), computed by route_timing()
from expected volumes, so the (R,S) rule orders against the lead time of the
plan actually being run, not one global constant.

The naive baseline: suppliers are split into angular sectors (sweep), each
truck owns one sector, visits it nearest-neighbour first, and runs every wave.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config
from .network import Network
from .scenario import rush_factor
from .state import TwinState


@dataclass(frozen=True)
class Route:
    truck: int
    dispatch_min: float         # minutes after midnight, repeated every day
    suppliers: tuple[int, ...]  # visit order
    lead_min: float             # planned minutes from dispatch until the load is in the supermarket


@dataclass(frozen=True)
class Plan:
    routes: tuple[Route, ...]
    name: str = "naive"

    @property
    def fleet(self) -> int:
        return len({r.truck for r in self.routes})


@dataclass(frozen=True)
class RouteTiming:
    km: float
    load: float              # expected containers
    door_release_min: float  # after dispatch: the truck is free again
    lead_min: float          # after dispatch: the load is in the supermarket


def production_hours(cfg: Config, t0: float, t1: float) -> float:
    """Production hours inside the absolute window [t0, t1] (minutes)."""
    start, end = cfg.line.prod_start_h * 60.0, cfg.line.prod_end_h * 60.0
    total = 0.0
    for day in range(int(t0 // 1440), int(t1 // 1440) + 1):
        lo, hi = max(t0, day * 1440.0 + start), min(t1, day * 1440.0 + end)
        total += max(0.0, hi - lo)
    return total / 60.0


def wave_index(cfg: Config, dispatch_min: float) -> int:
    return min(range(len(cfg.fleet.dispatch_h)),
               key=lambda w: abs(cfg.fleet.dispatch_h[w] * 60.0 - dispatch_min % 1440.0))


def visit_volumes(net: Network, cfg: Config, supplier: int,
                  arrivals: dict[int, float]) -> dict[int, float]:
    """Expected containers picked at each visit of one supplier.

    arrivals maps wave -> minute of day the load is covered from (dispatch +
    planned lead + margin). In (R,S) steady state the order placed at visit k is
    S(t_k) - IP(t_k) = consumption between this load's arrival and the next
    load's arrival, so the volume is the production hours in that window times
    the supplier's container rate.
    """
    waves = sorted(arrivals)
    per_h = sum(cfg.line.jph * p.usage / p.upc for p in net.parts if p.supplier == supplier)
    out = {}
    for k, w in enumerate(waves):
        a = 1440.0 + arrivals[w]
        nxt = arrivals[waves[k + 1]] + 1440.0 if k + 1 < len(waves) else arrivals[waves[0]] + 2880.0
        out[w] = per_h * production_hours(cfg, a, nxt)
    return out


def arrival_minute(cfg: Config, wave: int, lead_min: float) -> float:
    return cfg.fleet.dispatch_h[wave] * 60.0 + lead_min + cfg.inventory.lead_margin_h * 60.0


def route_timing(net: Network, cfg: Config, suppliers: tuple[int, ...], dispatch_min: float,
                 volumes: dict[int, float]) -> RouteTiming:
    """Deterministic expected timing of one route, mirroring the twin's steps.

    Travel uses the rush factor at each leg's departure (as the twin does) with
    mean-one noise; each stop adds the expected not-ready delay; the plant adds
    gate, the planned dock wait, the mean unload, and receiving.
    """
    so, pc, nc = cfg.supplier_ops, cfg.plant, cfg.network
    t, node, km, load = dispatch_min, 0, 0.0, 0.0
    for s in suppliers:
        t += net.nominal_min(node, s) * rush_factor(cfg, t)
        km += float(net.dist_km[node, s])
        t += net.suppliers[s - 1].p_late * nc.supplier_late_mean_min
        t += so.load_fixed_min + so.load_per_container_min * volumes[s]
        load += volumes[s]
        node = s
    t += net.nominal_min(node, 0) * rush_factor(cfg, t)
    km += float(net.dist_km[node, 0])
    unloaded = (t + pc.gate_min + cfg.planning.dock_wait_min
                + pc.unload_setup_min + pc.unload_per_container_min * load)
    return RouteTiming(km, load, unloaded + pc.dock_out_min - dispatch_min,
                       unloaded + pc.receiving_mean_min - dispatch_min)


DEFAULT_LEAD_MIN = 240.0  # starting guess for the lead / volume fixed point


def plan_volumes(net: Network, cfg: Config, routes: list[tuple[int, int, tuple[int, ...]]],
                 leads: list[float]) -> dict[tuple[int, int], float]:
    """(supplier, wave) -> expected containers, given each route's planned lead."""
    arrivals: dict[int, dict[int, float]] = {}
    for (_, w, seq), lead in zip(routes, leads, strict=True):
        for s in seq:
            arrivals.setdefault(s, {})[w] = arrival_minute(cfg, w, lead)
    return {(s, w): v for s, arr in arrivals.items()
            for w, v in visit_volumes(net, cfg, s, arr).items()}


def build_plan(net: Network, cfg: Config, routes: list[tuple[int, int, tuple[int, ...]]],
               name: str) -> Plan:
    """Turn (truck, wave, supplier sequence) triples into a Plan with planned lead times.

    Lead time depends on volume (loading and unloading) and volume depends on
    lead time (the window it covers), so the two are iterated to a fixed point.
    Volumes move about 1.7 min of lead per container, so it settles in a few rounds.
    """
    leads = [DEFAULT_LEAD_MIN] * len(routes)
    for _ in range(20):
        vol = plan_volumes(net, cfg, routes, leads)
        new = [route_timing(net, cfg, seq, cfg.fleet.dispatch_h[w] * 60.0,
                            {s: vol[(s, w)] for s in seq}).lead_min for _, w, seq in routes]
        done = max(abs(a - b) for a, b in zip(new, leads, strict=True)) < 0.01
        leads = new
        if done:
            break
    out = [Route(truck, cfg.fleet.dispatch_h[w] * 60.0, tuple(seq), lead)
           for (truck, w, seq), lead in zip(routes, leads, strict=True)]
    out.sort(key=lambda r: (r.dispatch_min, r.truck))
    return Plan(tuple(out), name)


def nearest_neighbour(net: Network, suppliers: list[int]) -> tuple[int, ...]:
    order, node, left = [], 0, list(suppliers)
    while left:
        nxt = min(left, key=lambda s: (net.dist_km[node, s], s))
        order.append(nxt)
        left.remove(nxt)
        node = nxt
    return tuple(order)


def naive_plan(net: Network, cfg: Config) -> Plan:
    by_angle = sorted((s.id for s in net.suppliers), key=net.angle)
    sectors = np.array_split(np.array(by_angle), cfg.fleet.n_trucks)
    routes = []
    for truck, sector in enumerate(sectors):
        seq = nearest_neighbour(net, [int(s) for s in sector])
        for w in range(len(cfg.fleet.dispatch_h)):
            routes.append((truck, w, seq))
    return build_plan(net, cfg, routes, "naive")


def next_visit(plan: Plan, supplier: int, t_now: float) -> tuple[float, Route]:
    """The next scheduled route that picks up this supplier, and its absolute dispatch minute."""
    routes = sorted((r for r in plan.routes if supplier in r.suppliers), key=lambda r: r.dispatch_min)
    day0 = (t_now // 1440) * 1440.0
    for r in routes:
        if day0 + r.dispatch_min > t_now + 1e-9:
            return day0 + r.dispatch_min, r
    return day0 + 1440.0 + routes[0].dispatch_min, routes[0]


def order_up_to_units(cfg: Config, net: Network, plan: Plan, part: int, t_now: float) -> float:
    """Periodic-review order-up-to level S for one part, in units.

    S covers consumption from now until the NEXT pickup of this supplier has
    reached the supermarket (its dispatch + its route's planned lead time),
    plus the safety stock. Both are measured in production hours, so nights
    and uneven waves are handled exactly.
    """
    p = net.parts[part]
    t_next, route = next_visit(plan, p.supplier, t_now)
    t_cover_end = t_next + route.lead_min + cfg.inventory.lead_margin_h * 60.0
    hours = production_hours(cfg, t_now, t_cover_end) + cfg.inventory.ss_h
    return p.usage * cfg.line.jph * hours


def order_quantities(state: TwinState, net: Network, cfg: Config, plan: Plan,
                     route: Route, t_now: float) -> dict[int, np.ndarray]:
    """Containers to pick at each supplier of the route, capped by truck capacity.

    If the route needs more than one truckload, containers are trimmed one at
    a time so that the safety cover each part has left at the END of its cycle
    stays as even as possible. Each step trims the part that would keep the most
    safety hours after losing one more container. (Comparing raw hours of cover
    after the order would always trim the least frequently picked parts first,
    because their orders must last longest.)
    """
    position = state.inventory_position_units()
    upc = state.upc
    rate_h = state.usage * cfg.line.jph

    route_parts = [p for s in route.suppliers for p in net.parts_of(s)]
    target = {p: order_up_to_units(cfg, net, plan, p, t_now) for p in route_parts}
    # S is a float built from hours; the tolerance stops float noise such as
    # 13.000000001 containers from rounding up to 14.
    need = {p: max(0, math.ceil((target[p] - position[p]) / upc[p] - 1e-9)) for p in route_parts}

    def safety_left_h(p: int, containers: int) -> float:
        cycle_need = target[p] - cfg.inventory.ss_h * rate_h[p]   # units used until the next load lands
        return (position[p] + containers * upc[p] - cycle_need) / rate_h[p]

    while sum(need.values()) > cfg.fleet.capacity:
        after = {p: safety_left_h(p, need[p] - 1) for p in route_parts if need[p] > 0}
        trim = max(after, key=lambda p: (after[p], p))
        need[trim] -= 1

    out = {}
    for s in route.suppliers:
        q = np.zeros(len(net.parts), dtype=np.int64)
        for p in net.parts_of(s):
            q[p] = need[p]
        out[s] = q
    return out
