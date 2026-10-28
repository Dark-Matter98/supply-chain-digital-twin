"""Exogenous randomness: traffic, supplier readiness, handling times, other carriers.

Each draw is keyed by the physical event it describes (see rng.py), so the
same seed gives the same world no matter which plan is being run.
"""
from __future__ import annotations

import math

import numpy as np

from .config import Config
from .network import Network
from .rng import Kind, keyed_rng, lognormal_mean_cv

TRAVEL_BUCKET_MIN = 15  # departures in the same 15 min on the same arc share a draw
CORRIDOR_STEP_MIN = 15  # corridor congestion changes in 15-min steps
STEPS_PER_DAY = 1440 // CORRIDOR_STEP_MIN


def corridor_path(cfg: Config, seed: int, day: int, supplier: int) -> np.ndarray:
    """One day of a supplier corridor's log travel-multiplier offset, per 15-min step.

    AR(1) congestion with stationary std corridor_sigma, shifted by -var/2 so the
    multiplier has mean one, plus incidents: Poisson-many per day, each adding
    log(U(incident_multiplier)) for an exponential duration. Zero when off.
    """
    tc = cfg.traffic
    path = np.zeros(STEPS_PER_DAY)
    if tc.corridor_sigma > 0:
        rng = keyed_rng(seed, Kind.CORRIDOR, day, supplier)
        v = tc.corridor_sigma ** 2
        x = rng.normal(0.0, tc.corridor_sigma)
        innov = tc.corridor_sigma * math.sqrt(1.0 - tc.corridor_phi ** 2)
        for k in range(STEPS_PER_DAY):
            path[k] = x - v / 2
            x = tc.corridor_phi * x + rng.normal(0.0, innov)
    if tc.incidents_per_day > 0:
        rng = keyed_rng(seed, Kind.INCIDENT, day, supplier)
        lo_h, hi_h = cfg.plant.operating_window_h
        for _ in range(int(rng.poisson(tc.incidents_per_day))):
            start = rng.uniform(lo_h * 60.0, hi_h * 60.0)
            end = start + rng.exponential(tc.incident_mean_min)
            jump = math.log(rng.uniform(*tc.incident_multiplier))
            k0, k1 = int(start // CORRIDOR_STEP_MIN), min(STEPS_PER_DAY, int(math.ceil(end / CORRIDOR_STEP_MIN)))
            path[k0:k1] += jump
    return path


def rush_factor(cfg: Config, t_min: float) -> float:
    """Time-of-day travel multiplier at an absolute minute."""
    h = (t_min % 1440.0) / 60.0
    for lo, hi in cfg.traffic.rush_hours:
        if lo <= h < hi:
            return cfg.traffic.rush_multiplier
    return 1.0


class Scenario:
    def __init__(self, cfg: Config, net: Network, seed: int):
        self.cfg = cfg
        self.net = net
        self.seed = seed
        self._paths: dict[tuple[int, int], np.ndarray] = {}
        tc = cfg.traffic
        self.drifting = tc.corridor_sigma > 0 or tc.incidents_per_day > 0

    def tod_multiplier(self, t_min: float) -> float:
        return rush_factor(self.cfg, t_min)

    def corridor_offset(self, supplier: int, t_min: float) -> float:
        """Log travel-multiplier offset of a supplier's corridor at an absolute minute."""
        if not self.drifting or supplier == 0:
            return 0.0
        day = int(t_min // 1440)
        key = (day, supplier)
        if key not in self._paths:
            self._paths[key] = corridor_path(self.cfg, self.seed, day, supplier)
        return float(self._paths[key][int((t_min % 1440) // CORRIDOR_STEP_MIN)])

    def arc_offset(self, i: int, j: int, t_min: float) -> float:
        """An arc's corridor offset: the mean over its supplier endpoints (the plant has none)."""
        ends = [n for n in (i, j) if n != 0]
        return sum(self.corridor_offset(n, t_min) for n in ends) / len(ends) if ends else 0.0

    def travel_min(self, i: int, j: int, t_depart: float, truck: int) -> float:
        """Travel time of one leg, fixed at departure: rush factor x corridor congestion x noise."""
        nominal = self.net.nominal_min(i, j)
        if nominal == 0.0:
            return 0.0
        sigma = max(self.net.sigma(i), self.net.sigma(j))
        noise = 1.0
        if sigma > 0:
            day = int(t_depart // 1440)
            bucket = int((t_depart % 1440) // TRAVEL_BUCKET_MIN)
            z = keyed_rng(self.seed, Kind.TRAVEL, day, i, j, truck, bucket).standard_normal()
            noise = math.exp(sigma * z - sigma * sigma / 2)  # mean-one multiplier
        if self.drifting:
            noise *= math.exp(self.arc_offset(i, j, t_depart))
        st = self.cfg.stress
        if st.road_supplier and st.road_supplier in (i, j) and st.active(t_depart):
            noise *= st.road_multiplier
        return nominal * self.tod_multiplier(t_depart) * noise

    def supplier_delay_min(self, supplier: int, t_arrive: float) -> float:
        st = self.cfg.stress
        if supplier == st.supplier_down and st.active(t_arrive):
            return st.end_min - t_arrive          # nothing ships until the supplier is back
        p_late = self.net.suppliers[supplier - 1].p_late
        if p_late <= 0:
            return 0.0
        day = int(t_arrive // 1440)
        hour = int((t_arrive % 1440) // 60)
        rng = keyed_rng(self.seed, Kind.SUPPLIER_LATE, day, supplier, hour)
        if rng.random() < p_late:
            return float(rng.exponential(self.cfg.network.supplier_late_mean_min))
        return 0.0

    def unload_min(self, n_containers: int, day: int, truck_key: int, seq: int) -> float:
        pc = self.cfg.plant
        mean = pc.unload_setup_min + pc.unload_per_container_min * n_containers
        rng = keyed_rng(self.seed, Kind.UNLOAD, day, truck_key, seq)
        return lognormal_mean_cv(rng, mean, pc.unload_cv)

    def receiving_min(self, day: int, truck_key: int, seq: int) -> float:
        pc = self.cfg.plant
        rng = keyed_rng(self.seed, Kind.RECEIVING, day, truck_key, seq)
        return lognormal_mean_cv(rng, pc.receiving_mean_min, pc.receiving_cv)

    def background_arrivals(self, day: int) -> list[tuple[float, int]]:
        """Other carriers' trucks for one day: (absolute arrival minute, containers)."""
        pc = self.cfg.plant
        if pc.background_trucks_per_h <= 0:
            return []
        rng = keyed_rng(self.seed, Kind.BACKGROUND, day)
        lo_h, hi_h = pc.operating_window_h
        n = int(rng.poisson(pc.background_trucks_per_h * (hi_h - lo_h)))
        times = sorted(rng.uniform(lo_h * 60.0, hi_h * 60.0, n))
        sizes = rng.integers(pc.background_containers[0], pc.background_containers[1] + 1, n)
        return [(day * 1440.0 + float(t), int(k)) for t, k in zip(times, sizes)]
