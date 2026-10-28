"""Estimation: a Kalman filter on corridor congestion, and supplier readiness.

Corridor model, shared with the world (scenario.py) except for incidents:
  - each supplier corridor has a log travel-multiplier offset z_s(t), AR(1) in
    15-min steps around a stationary mean;
  - an arc's offset is the mean of its supplier endpoints' offsets;
  - a leg's log duration = log(nominal x rush) + arc offset + leg noise
    (sigma of the arc, shifted so the noise multiplier has mean one).
Incidents (jumps) are not in the filter's model; their average effect is
folded into the stationary mean and variance, and the filter sees the rest as
large innovations. Filter parameters come from config, as if fitted on history.

Supplier readiness: Beta-Bernoulli on each supplier's chance of not being ready,
from a common prior; a delay already running is exponential (memoryless).

Views, for the ablation:
  bayes  - filter mean and covariance, per-corridor leg sigma, ping-sharpened legs
  det    - the same means, no spread (point estimates)
  fixed  - the same means, one pooled variance that never updates
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from .config import Config
from .network import Network
from .observe import Observation, TruckObs
from .scenario import CORRIDOR_STEP_MIN, rush_factor

READINESS_PRIOR_STRENGTH = 20.0   # pseudo-observations behind the common p_late prior
MIN_PING_PROGRESS = 0.1           # earlier pings say too little about the leg's duration
VIEWS = ("bayes", "det", "fixed")


def _incident_moments(cfg: Config) -> tuple[float, float]:
    """Mean and variance an incident process adds to a corridor's offset, on average."""
    tc = cfg.traffic
    if tc.incidents_per_day <= 0:
        return 0.0, 0.0
    lo, hi = np.log(tc.incident_multiplier)
    ej = (hi * math.exp(hi) - lo * math.exp(lo) - (math.exp(hi) - math.exp(lo))) / (math.exp(hi) - math.exp(lo)) \
        if hi > lo else lo                                   # E[log U] for U uniform on [a, b]
    grid = np.linspace(*tc.incident_multiplier, 2001)
    ej2 = float(np.mean(np.log(grid) ** 2))
    window = (cfg.plant.operating_window_h[1] - cfg.plant.operating_window_h[0]) * 60.0
    p_active = min(1.0, tc.incidents_per_day * tc.incident_mean_min / window)
    return p_active * ej, p_active * ej2 - (p_active * ej) ** 2


@dataclass
class LegPosterior:
    mu: float       # mean of log duration
    sd: float       # sd of log duration
    elapsed: float  # minutes already driven (duration is known to exceed this)


class Estimator:
    def __init__(self, cfg: Config, net: Network, view: str = "bayes"):
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}")
        self.cfg, self.net, self.view = cfg, net, view
        tc = cfg.traffic
        n = len(net.suppliers)
        inc_mean, inc_var = _incident_moments(cfg)
        self.phi = tc.corridor_phi
        self.v = tc.corridor_sigma ** 2 + inc_var          # stationary variance of an offset
        self.mu_inf = -tc.corridor_sigma ** 2 / 2 + inc_mean
        self.m = np.full(n, self.mu_inf)
        self.P = np.eye(n) * self.v
        self.t = 0.0
        self.leg_sigma = np.array([s.travel_sigma for s in net.suppliers])
        self.pooled_sigma = float(np.sqrt(np.mean(self.leg_sigma ** 2)))
        p0 = cfg.network.p_supplier_late
        self.a = np.full(n, p0 * READINESS_PRIOR_STRENGTH)
        self.b = np.full(n, (1 - p0) * READINESS_PRIOR_STRENGTH)
        self._legs_seen = 0
        self._stops_seen = 0

    # ----------------------------------------------------------- filter core
    def _h(self, i: int, j: int) -> np.ndarray:
        h = np.zeros(len(self.m))
        ends = [n for n in (i, j) if n != 0]
        for n in ends:
            h[n - 1] += 1.0 / len(ends)
        return h

    def arc_sigma(self, i: int, j: int) -> float:
        if self.view == "fixed":
            return self.pooled_sigma
        return max(self.net.sigma(i), self.net.sigma(j))

    def _decay(self, minutes: float) -> float:
        return self.phi ** (max(0.0, minutes) / CORRIDOR_STEP_MIN) if self.phi > 0 else (1.0 if minutes <= 0 else 0.0)

    def _propagate(self, m, P, minutes):
        a = self._decay(minutes)
        return self.mu_inf + a * (m - self.mu_inf), a * a * P + (1 - a * a) * self.v * np.eye(len(m))

    def _update(self, m, P, h, y, R):
        S = float(h @ P @ h) + R
        if S <= 0:
            return m, P
        K = P @ h / S
        return m + K * (y - h @ m), P - np.outer(K, h @ P)

    def _leg_obs(self, i, j, t_dep, log_dur, extra_R=0.0):
        """Observation of the corridor offsets from one leg's (log) duration."""
        sig = self.arc_sigma(i, j)
        y = log_dur - math.log(self.net.nominal_min(i, j) * rush_factor(self.cfg, t_dep)) + sig * sig / 2
        lag = max(0.0, self.t - t_dep)             # out-of-sequence: the state has drifted since t_dep
        R = sig * sig + extra_R + self.v * (1 - self._decay(lag) ** 2)
        return self._h(i, j), y, R

    def ingest(self, obs: Observation) -> None:
        """Learn from legs completed and supplier arrivals since the last call."""
        if self.v > 0:
            for truck, i, j, t0, t1 in obs.legs[self._legs_seen:]:
                if t0 > self.t:
                    self.m, self.P = self._propagate(self.m, self.P, t0 - self.t)
                    self.t = t0
                h, y, R = self._leg_obs(i, j, t0, math.log(t1 - t0))
                self.m, self.P = self._update(self.m, self.P, h, y, R)
            if obs.t > self.t:
                self.m, self.P = self._propagate(self.m, self.P, obs.t - self.t)
                self.t = obs.t
        self._legs_seen = len(obs.legs)
        for sup, _, late in obs.stops[self._stops_seen:]:
            self.a[sup - 1] += late
            self.b[sup - 1] += 1 - late
        self._stops_seen = len(obs.stops)

    def corridor_now(self, obs: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Offsets at obs.t, sharpened by the latest ping of every leg in progress
        (a temporary copy: pings are never committed, so they are not counted twice)."""
        m, P = self.m.copy(), self.P.copy()
        if self.v > 0 and self.view != "fixed":
            for tr in obs.trucks:
                if tr.ping is None or tr.ping.progress < MIN_PING_PROGRESS:
                    continue
                log_dur = math.log((tr.ping.t - tr.leg_start) / tr.ping.progress)
                h, y, R = self._leg_obs(tr.leg_from, tr.leg_to, tr.leg_start, log_dur,
                                        extra_R=(self.cfg.dynamic.gps_sigma / tr.ping.progress) ** 2)
                m, P = self._update(m, P, h, y, R)
        if self.view == "fixed":
            P = np.eye(len(m)) * self.v
        return m, P

    # ------------------------------------------------------------- legs
    def leg_posterior(self, tr: TruckObs, now: float) -> LegPosterior:
        """Log-duration posterior of the leg a truck is driving: committed filter prior x its ping."""
        i, j, t0 = tr.leg_from, tr.leg_to, tr.leg_start
        sig = self.arc_sigma(i, j)
        h = self._h(i, j)
        base = math.log(self.net.nominal_min(i, j) * rush_factor(self.cfg, t0))
        mu = base + float(h @ self.m) - sig * sig / 2
        var = (float(h @ self.P @ h) if self.view != "fixed" else self.v * float(h @ h)) + sig * sig
        if tr.ping is not None and tr.ping.progress >= MIN_PING_PROGRESS:
            mu_p = math.log((tr.ping.t - t0) / tr.ping.progress)
            var_p = (self.cfg.dynamic.gps_sigma / tr.ping.progress) ** 2
            post_var = 1.0 / (1.0 / var + 1.0 / var_p)
            mu_post = post_var * (mu / var + mu_p / var_p)
            if self.view == "fixed":       # keep the live mean, drop the sharpened spread
                mu, var = mu_post, var
            else:
                mu, var = mu_post, post_var
        return LegPosterior(mu, math.sqrt(max(var, 1e-12)), max(now - t0, 1e-6))

    @staticmethod
    def sample_remaining(post: LegPosterior, u: np.ndarray) -> np.ndarray:
        """Remaining minutes of the leg for uniforms u, given it has not ended yet."""
        z_c = (math.log(post.elapsed) - post.mu) / post.sd
        lo = float(ndtr(z_c))
        q = lo + (1 - lo) * np.clip(u, 1e-9, 1 - 1e-9)
        dur = np.exp(post.mu + post.sd * ndtri(np.clip(q, 1e-12, 1 - 1e-12)))
        return np.maximum(dur - post.elapsed, 0.0)

    @staticmethod
    def mean_remaining(post: LegPosterior) -> float:
        """E[duration | duration > elapsed] - elapsed, for the point-estimate view."""
        z_c = (math.log(post.elapsed) - post.mu) / post.sd
        tail = 1.0 - float(ndtr(z_c))
        if tail < 1e-9:
            return 1.0
        mean = math.exp(post.mu + post.sd ** 2 / 2) * (1.0 - float(ndtr(z_c - post.sd))) / tail
        return max(mean - post.elapsed, 0.0)

    def p_late(self) -> np.ndarray:
        return self.a / (self.a + self.b)
