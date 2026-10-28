"""The observation layer: what a TMS / WMS would actually see.

The controller, estimator and predictor only ever get an Observation. True
leg end times, the corridor congestion path and future supplier delays stay
inside the twin. GPS pings are generated from the truth with keyed noise:
every ping_min while a truck drives, it reports the fraction of the leg done.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .rng import Kind, keyed_rng


@dataclass(frozen=True)
class Ping:
    t: float
    progress: float           # reported fraction of the current leg completed


@dataclass(frozen=True)
class TruckObs:
    id: int
    status: str               # idle | travel | late | load | gate | yard | door_wait | unload | dock_out
    status_since: float
    node: int                 # last node reached
    leg_from: int
    leg_to: int
    leg_start: float          # nan when not driving
    ping: Ping | None         # latest ping of the leg in progress
    waiting_since: float      # nan unless waiting at a supplier that is not ready
    remaining: tuple[int, ...]  # stops after the current destination
    orders: dict              # supplier -> containers per part, for stops not yet picked up
    load: np.ndarray          # containers on board
    priority: int
    last_change: float
    plant_arrival: float      # gate arrival of the current plant visit, nan if not at the plant
    trips_started: int = 0    # scheduled trips this truck has started since day 0


@dataclass(frozen=True)
class Observation:
    t: float
    trucks: tuple[TruckObs, ...]
    on_order: np.ndarray
    in_plant: np.ndarray
    staging: np.ndarray
    supermarket: np.ndarray
    tugger: np.ndarray
    forklift: np.ndarray
    courier: np.ndarray
    line_full: np.ndarray
    line_open: np.ndarray
    pending: np.ndarray        # kanban signals waiting for the tugger, per part
    receiving: tuple           # (unloaded at, containers per part) for batches not yet received
    hot_parts: tuple[int, ...]
    line_status: str
    legs: tuple                # completed legs (truck, from, to, start, end), all so far
    stops: tuple               # supplier arrivals (supplier, time, was not ready), all so far
    plant_waits: tuple[float, ...]  # recent gate-to-unload-start waits of milk-run trucks
    couriers: tuple            # in flight: (dispatched, supplier, containers per part, picked up at)


def latest_ping(twin, truck) -> Ping | None:
    """The newest GPS ping of the leg in progress, or None if none was sent yet."""
    if math.isnan(truck.leg_start):
        return None
    dt = twin.cfg.dynamic.ping_min
    k = int((twin.env.now - truck.leg_start) // dt)
    if k < 1:
        return None
    t_ping = truck.leg_start + k * dt
    if t_ping >= truck.leg_end_true:
        return None
    day = int(truck.leg_start // 1440)
    noise = keyed_rng(twin.seed, Kind.PING, day, truck.id, truck.leg_seq, k).normal(0.0, twin.cfg.dynamic.gps_sigma)
    true_progress = (t_ping - truck.leg_start) / (truck.leg_end_true - truck.leg_start)
    return Ping(t_ping, float(np.clip(true_progress + noise, 0.01, 0.99)))


def observe(twin, wait_days: float = 3.0) -> Observation:
    s, now = twin.state, twin.env.now
    arrivals = {}
    for v in twin.visits:
        if v.kind == "milk" and math.isnan(v.t_leave):
            arrivals[v.key] = v.t_arrive
    trucks = tuple(
        TruckObs(t.id, t.status, t.status_since, t.node, t.leg_from, t.leg_to, t.leg_start, latest_ping(twin, t),
                 t.waiting_since, tuple(t.remaining), {k: q.copy() for k, q in t.orders.items()},
                 t.load.copy(), t.priority, t.last_change, arrivals.get(t.id, math.nan),
                 twin._trip_seq[t.id])
        for t in s.trucks)
    waits = tuple(v.t_crew - v.t_arrive for v in twin.visits
                  if v.kind == "milk" and not math.isnan(v.t_crew) and v.t_arrive >= now - wait_days * 1440)
    return Observation(
        t=now, trucks=trucks, on_order=s.on_order.copy(), in_plant=s.in_plant.copy(),
        staging=s.staging.copy(), supermarket=s.supermarket.copy(), tugger=s.tugger.copy(),
        forklift=s.forklift.copy(), courier=s.courier.copy(), line_full=s.line_full.copy(),
        line_open=s.line_open.copy(), pending=s.pending_signals(),
        receiving=tuple((t, b.copy()) for t, b in twin.in_receiving.values()),
        hot_parts=tuple(sorted(twin.hot_parts)), line_status=s.line_status,
        legs=tuple(twin.legs), stops=tuple(twin.stop_log), plant_waits=waits,
        couriers=tuple((c[0], c[1], c[2].copy(), c[3]) for c in twin.couriers.values()))
