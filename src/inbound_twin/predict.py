"""Fast predictor: vectorized Monte Carlo of material reaching the line.

From an Observation and the estimator, for the current plan and any candidate
change, simulate N scenarios of the next horizon_h hours:
  trucks  finish the current leg (ping-sharpened posterior), wait at suppliers
          (readiness model), load, drive the remaining legs (filter predictive),
          wait at the dock (recently observed gate-to-unload waits), unload and
          receive; then their next scheduled trips, with expected volumes
  plant   batches in receiving, couriers in flight, hot runs
  line    per part, stock reaches the line one tugger wait after the supermarket
          (hot_run_min with a hot run); a part is short whenever cumulative
          consumption exceeds what has reached the line
Line-down per scenario = production minutes in which any part is short. It
ignores that a stopped line also stops consuming; gate 2 checks the result.

The "det" view runs one scenario at expected values (point estimates).
Common random numbers: each draw comes from a stream keyed by (seed, call,
what it is), so the current plan and every candidate see the same scenarios.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr, ndtri

from .config import Config
from .estimate import Estimator
from .network import Network
from .observe import Observation, TruckObs
from .plan import Plan, plan_volumes, wave_index
from .rng import Kind, keyed_rng
from .scenario import CORRIDOR_STEP_MIN, rush_factor

AT_PLANT_WAITING = ("gate", "yard", "door_wait")


@dataclass(frozen=True)
class Candidate:
    """A change to the current plan. The empty Candidate is "change nothing"."""
    truck: int = -1
    remaining: tuple[int, ...] | None = None  # new stop order after the current one; dropped stops are deferred
    priority: bool = False                    # first in line for a door at the next plant visit
    courier_supplier: int = -1
    courier_q: tuple[int, ...] | None = None  # containers per part
    hot_part: int = -1
    leave_now: bool = False                   # stop waiting at a not-ready supplier, without its load
    label: str = "no change"

    def stops_changed(self, obs: Observation) -> int:
        if self.truck < 0:
            return 0
        n = int(self.leave_now)
        if self.remaining is not None:
            old = obs.trucks[self.truck].remaining
            n += sum(1 for a, b in zip(old, self.remaining) if a != b) + abs(len(old) - len(self.remaining))
        return n

    @property
    def is_change(self) -> bool:
        return (self.remaining is not None or self.priority or self.courier_supplier >= 0
                or self.hot_part >= 0 or self.leave_now)


@dataclass
class Forecast:
    line_down: np.ndarray   # production minutes short, per scenario, within the horizon
    p_starve: np.ndarray    # per part: share of scenarios short within trigger_h
    margin_min: np.ndarray  # per part, first scenario: least slack over the horizon, in minutes of use
    action_cost: float      # premium + disruption + km change, USD
    p_line: float = 0.0     # share of scenarios in which the line stops within trigger_h

    def expected_cost(self, line_down_per_min: float) -> float:
        return float(self.line_down.mean()) * line_down_per_min + self.action_cost


class _Draws:
    """Keyed random streams for one predictor call; repeat keys return the same numbers."""

    def __init__(self, seed: int, day: int, call: int, n: int, point: bool):
        self.seed, self.day, self.call, self.n, self.point = seed, day, call, n, point
        self._cache: dict[tuple, np.ndarray] = {}

    def uniform(self, *key: int) -> np.ndarray:
        if key not in self._cache:
            rng = keyed_rng(self.seed, Kind.PREDICT, self.day, self.call, *key)
            self._cache[key] = rng.random(self.n)
        return self._cache[key]

    def normal(self, *key: int) -> np.ndarray:
        return ndtri(np.clip(self.uniform(*key), 1e-12, 1 - 1e-12))

    def lognormal(self, mean, cv: float, *key: int) -> np.ndarray:
        mean = np.broadcast_to(np.asarray(mean, dtype=float), (self.n,))
        if self.point or cv <= 0:
            return mean.copy()
        s2 = math.log1p(cv * cv)
        return mean * np.exp(math.sqrt(s2) * self.normal(*key) - s2 / 2)

    def delay(self, p_late: float, mean: float, *key: int) -> np.ndarray:
        """Not-ready delay at a future stop: late with p_late, then exponential."""
        if self.point:
            return np.full(self.n, p_late * mean)
        u = self.uniform(*key, 0)
        v = self.uniform(*key, 1)
        return np.where(u < p_late, -mean * np.log(np.clip(v, 1e-12, 1.0)), 0.0)

    def residual_exp(self, mean: float, *key: int) -> np.ndarray:
        if self.point:
            return np.full(self.n, mean)
        return -mean * np.log(np.clip(self.uniform(*key), 1e-12, 1.0))

    def residual_lognormal(self, mean: float, cv: float, elapsed: float, *key: int) -> np.ndarray:
        """Remaining time of a lognormal activity that has run `elapsed` already."""
        if self.point or cv <= 0:
            return np.full(self.n, max(mean - elapsed, 0.0))
        s2 = math.log1p(cv * cv)
        mu, sd = math.log(mean) - s2 / 2, math.sqrt(s2)
        lo = float(ndtr((math.log(max(elapsed, 1e-6)) - mu) / sd))
        q = lo + (1 - lo) * self.uniform(*key)
        return np.maximum(np.exp(mu + sd * ndtri(np.clip(q, 1e-12, 1 - 1e-12))) - elapsed, 0.0)

    def empirical_residual(self, values: np.ndarray, elapsed: float, fallback: float, *key: int) -> np.ndarray:
        """Remaining part of a wait drawn from recent observed waits longer than elapsed."""
        pool = values[values > elapsed] if values.size else values
        if pool.size == 0:
            return np.full(self.n, max(fallback - elapsed, 0.0))
        if self.point:
            return np.full(self.n, float(pool.mean()) - elapsed)
        idx = np.minimum((self.uniform(*key) * pool.size).astype(int), pool.size - 1)
        return np.sort(pool)[idx] - elapsed


class Predictor:
    def __init__(self, cfg: Config, net: Network, plan: Plan, seed: int):
        self.cfg, self.net, self.plan, self.seed = cfg, net, plan, seed
        self.rate_min = np.array([p.usage for p in net.parts], dtype=float) * cfg.line.jph / 60.0
        self.upc = np.array([p.upc for p in net.parts], dtype=float)
        self.parts_of = {s.id: list(net.parts_of(s.id)) for s in net.suppliers}
        triples = [(r.truck, wave_index(cfg, r.dispatch_min), r.suppliers) for r in plan.routes]
        vol = plan_volumes(net, cfg, triples, [r.lead_min for r in plan.routes])
        self.plan_q = {}   # (supplier, wave) -> expected containers per part of that supplier
        for (s, w), v in vol.items():
            per_h = {p: self.rate_min[p] * 60 / self.upc[p] for p in self.parts_of[s]}
            total = sum(per_h.values())
            self.plan_q[(s, w)] = {p: v * per_h[p] / total for p in per_h}
        self.routes_of = {}
        for r in sorted(plan.routes, key=lambda r: r.dispatch_min):
            self.routes_of.setdefault(r.truck, []).append(r)
        self.rush_tab = np.array([rush_factor(cfg, float(m)) for m in range(1440)])
        self.calls = 0

    # ---------------------------------------------------------------- pieces
    def _leg(self, d: _Draws, z0, i: int, j: int, t_dep: np.ndarray, now: float, est: Estimator, *key) -> np.ndarray:
        """Duration of a future leg departing at t_dep (per scenario)."""
        nominal = self.net.nominal_min(i, j)
        if nominal == 0:
            return np.zeros(d.n)
        rush = self.rush_tab[(np.broadcast_to(t_dep, (d.n,)) % 1440).astype(int)]
        h = est._h(i, j)
        sig = est.arc_sigma(i, j)
        if d.point:
            return nominal * rush * np.exp(float(h @ est.m))
        ahead = np.maximum(np.broadcast_to(t_dep, (d.n,)) - now, 0.0)
        a = est.phi ** (ahead / CORRIDOR_STEP_MIN) if est.phi > 0 else (ahead <= 0).astype(float)
        state = est.mu_inf + a * ((z0 - est.mu_inf) @ h)
        extra = np.sqrt(np.maximum((1 - a * a) * est.v * float(h @ h), 0.0)) * d.normal(*key, 1)
        noise = sig * d.normal(*key, 2) - sig * sig / 2
        return nominal * rush * np.exp(state + extra + noise)

    def _plant(self, d: _Draws, t_arrive: np.ndarray, load: float, priority: bool, waits: np.ndarray, *key):
        """(unloaded, door released) after arriving at the gate."""
        pc = self.cfg.plant
        if priority:
            wait = np.full(d.n, pc.gate_min)
        else:
            wait = d.empirical_residual(waits, 0.0, pc.gate_min + self.cfg.planning.dock_wait_min, *key, 1)
        unload = d.lognormal(pc.unload_setup_min + pc.unload_per_container_min * load, pc.unload_cv, *key, 2)
        unloaded = t_arrive + wait + unload
        return unloaded, unloaded + pc.dock_out_min

    def _receive(self, d: _Draws, unloaded: np.ndarray, *key) -> np.ndarray:
        pc = self.cfg.plant
        return unloaded + d.lognormal(pc.receiving_mean_min, pc.receiving_cv, *key)

    # ------------------------------------------------------------ one plan
    def _truck(self, obs, est, d, z0, tr: TruckObs, cand: Candidate, deferred_out: dict, arrivals: list):
        """Simulate one truck's current trip under the candidate; returns when it is free."""
        cfg, so, now = self.cfg, self.cfg.supplier_ops, obs.t
        remaining = list(tr.remaining)
        priority = tr.priority == -1
        if cand.truck == tr.id:
            if cand.remaining is not None:
                for s in remaining:
                    if s not in cand.remaining:
                        deferred_out[s] = deferred_out.get(s, 0) + tr.orders[s]
                remaining = list(cand.remaining)
            priority = priority or cand.priority
        p_late = est.p_late()
        key = (tr.id, 0)
        load = tr.load.astype(float).copy()
        load_vec = np.tile(load, (d.n, 1))

        if tr.status == "idle":
            return np.full(d.n, now)
        at_plant_now = tr.status == "arrived" and tr.node == 0
        if tr.status in AT_PLANT_WAITING or tr.status in ("unload", "dock_out") or at_plant_now:
            waits = np.asarray(obs.plant_waits, dtype=float)
            if tr.status == "dock_out":
                return np.full(d.n, tr.status_since + cfg.plant.dock_out_min)
            if tr.status == "unload":
                pc = cfg.plant
                unloaded = now + d.residual_lognormal(pc.unload_setup_min + pc.unload_per_container_min * load.sum(),
                                                      pc.unload_cv, now - tr.status_since, *key, 3)
            else:
                elapsed = 0.0 if at_plant_now else now - tr.plant_arrival
                wait = (np.full(d.n, max(cfg.plant.gate_min - elapsed, 0.0)) if priority else
                        d.empirical_residual(waits, elapsed, cfg.plant.gate_min + cfg.planning.dock_wait_min, *key, 4))
                unload = d.lognormal(cfg.plant.unload_setup_min + cfg.plant.unload_per_container_min * load.sum(),
                                     cfg.plant.unload_cv, *key, 5)
                unloaded = now + wait + unload
            arrivals.append((self._receive(d, unloaded, *key, 6), load))
            return unloaded + cfg.plant.dock_out_min

        # On the road or at a supplier: finish the current stop, then the rest.
        if tr.status == "travel":
            post = est.leg_posterior(tr, now)
            if d.point:
                t = np.full(d.n, now + est.mean_remaining(post))
            else:
                t = now + est.sample_remaining(post, d.uniform(*key, 7))
            node = tr.leg_to
            stop_pending = node != 0
        else:   # "arrived", "late" or "load" at supplier tr.node
            node, t, stop_pending = tr.node, np.full(d.n, now), True

        def visit(s: int, t_arr: np.ndarray, waiting_now: bool, loading_since: float | None, k: int):
            q = tr.orders.get(s)
            n_q = float(q.sum()) if q is not None else 0.0
            if waiting_now:
                delay = d.residual_exp(cfg.network.supplier_late_mean_min, 100 + s)   # shared with couriers
            elif loading_since is not None:
                delay = np.zeros(d.n)
            else:
                delay = d.delay(float(p_late[s - 1]), cfg.network.supplier_late_mean_min, 200 + s, k)
            load_time = so.load_fixed_min + so.load_per_container_min * n_q
            if loading_since is not None:
                load_time = max(load_time - (now - loading_since), 0.0)
            if q is not None:
                load_vec[:] += q
            return t_arr + delay + load_time

        waiting = tr.status == "late" or (tr.status == "arrived" and not math.isnan(tr.waiting_since))
        leave = cand.truck == tr.id and cand.leave_now and waiting
        if leave:            # this stop's order waits for the supplier's next pickup
            deferred_out[node] = deferred_out.get(node, 0) + tr.orders[node]
        elif stop_pending:
            t = visit(node, t, waiting, tr.status_since if tr.status == "load" else None, 0)
        for k, s in enumerate(remaining, start=1):
            t = t + self._leg(d, z0, node, s, t, now, est, tr.id, 10 + k)
            t = visit(s, t, False, None, k)
            node = s
        if node != 0:
            t = t + self._leg(d, z0, node, 0, t, now, est, tr.id, 30)
        total = float(load_vec[0].sum())
        unloaded, free = self._plant(d, t, total, priority, np.asarray(obs.plant_waits, dtype=float), tr.id, 40)
        arrivals.append((self._receive(d, unloaded, tr.id, 41), load_vec))
        return free

    def _future_trips(self, obs, est, d, z0, truck: int, started: int, free: np.ndarray, deferred: dict,
                      arrivals: list):
        """Scheduled trips of this truck not started yet that are due within the horizon.

        The truck runs its schedule in order, so the first `started` trips are done or
        under way; a trip whose time has passed but has not started (the truck was
        still busy) leaves as soon as the truck is free.
        """
        cfg, so, now = self.cfg, self.cfg.supplier_ops, obs.t
        end = now + cfg.dynamic.horizon_h * 60.0
        p_late = est.p_late()
        routes = self.routes_of.get(truck, [])
        if not routes:
            return free
        last_day = int(end // 1440) + 1
        trips = [(day * 1440.0 + r.dispatch_min, r) for day in range(last_day + 1) for r in routes]
        for n_trip, (t_sched, r) in enumerate(trips):
            if n_trip < started:
                continue
            if t_sched >= end:
                break
            w = wave_index(cfg, r.dispatch_min)
            t = np.maximum(free, t_sched)
            node = 0
            load = np.zeros(len(self.upc))
            # Keys: (truck, kind, trip, stop) with kind 1000-4000, apart from the current trip's keys.
            for k, s in enumerate(r.suppliers):
                t = t + self._leg(d, z0, node, s, t, now, est, truck, 1000, n_trip, k)
                q = np.zeros(len(self.upc))
                for p, v in self.plan_q.get((s, w), {}).items():
                    q[p] = v
                if s in deferred:
                    q += deferred.pop(s)
                t = t + d.delay(float(p_late[s - 1]), cfg.network.supplier_late_mean_min, 300 + s, n_trip, k)
                t = t + so.load_fixed_min + so.load_per_container_min * float(q.sum())
                load += q
                node = s
            t = t + self._leg(d, z0, node, 0, t, now, est, truck, 2000, n_trip)
            unloaded, free = self._plant(d, t, float(load.sum()), False, np.asarray(obs.plant_waits, dtype=float),
                                         truck, 3000, n_trip)
            arrivals.append((self._receive(d, unloaded, truck, 4000, n_trip), load))
        return free

    def _courier(self, obs, est, d, z0, s: int, q: np.ndarray, t_disp: float, t_pick: float, key: int, arrivals):
        cfg, so, now = self.cfg, self.cfg.supplier_ops, obs.t
        waiting_trucks = [tr for tr in obs.trucks if tr.status == "late" and tr.node == s]
        if not math.isnan(t_pick):
            t = np.full(d.n, max(t_pick, now))
        else:
            t = np.full(d.n, t_disp)
            t = t + self._leg(d, z0, 0, s, t, now, est, 500 + key, 1)
            t = np.maximum(t, now)
            if waiting_trucks:
                t = t + d.residual_exp(cfg.network.supplier_late_mean_min, 100 + s)
            else:
                t = t + d.delay(float(est.p_late()[s - 1]), cfg.network.supplier_late_mean_min, 400 + s, key)
            t = t + so.load_fixed_min + so.load_per_container_min * float(q.sum())
        t = t + self._leg(d, z0, s, 0, t, now, est, 500 + key, 2) + cfg.dynamic.courier_receiving_min
        arrivals.append((np.maximum(t, now), q.astype(float)))

    def _line(self, obs, d, arrivals, hot: set) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg, now = self.cfg, obs.t
        H = int(cfg.dynamic.horizon_h * 60)
        trig = int(cfg.dynamic.trigger_h * 60)
        minutes = now + np.arange(H)
        tod = minutes % 1440
        prod = (tod >= cfg.line.prod_start_h * 60) & (tod < cfg.line.prod_end_h * 60)
        cum_prod = np.concatenate([[0], np.cumsum(prod)])[1:]      # production minutes by the end of minute k
        loop = cfg.tugger.loop_min
        n_parts = len(self.upc)
        short_any = np.zeros((d.n, H), dtype=bool)
        p_starve = np.zeros(n_parts)
        margin = np.full(n_parts, np.inf)
        base_all = (obs.line_full * self.upc + obs.line_open
                    + (obs.supermarket + obs.tugger + obs.forklift) * self.upc)
        for p in range(n_parts):
            demand = self.rate_min[p] * cum_prod
            if base_all[p] >= demand[-1]:
                margin[p] = (base_all[p] - demand[-1]) / self.rate_min[p]
                continue
            avail = np.full((d.n, H + 1), 0.0)
            avail[:, 0] = base_all[p]
            rows = np.arange(d.n)
            for t_sm, q in arrivals:
                qp = q[:, p] if q.ndim == 2 else np.full(d.n, q[p])
                if not np.any(qp > 0):
                    continue
                if p in hot:
                    t_line = t_sm + cfg.dynamic.hot_run_min
                else:
                    t_line = np.ceil(t_sm / loop) * loop + loop / 2
                idx = np.clip(np.ceil(t_line - now).astype(int), 0, H)
                np.add.at(avail, (rows, idx), qp * self.upc[p])
            have = np.cumsum(avail, axis=1)[:, :H]
            gap = have - demand
            short = (gap < 0) & prod          # only production minutes can stop the line
            short_any |= short
            p_starve[p] = float(short[:, :trig].any(axis=1).mean())
            margin[p] = float(gap[0].min()) / self.rate_min[p]
        line_down = short_any.sum(axis=1).astype(float)
        p_line = float(short_any[:, :trig].any(axis=1).mean())
        return line_down, p_starve, margin, p_line

    # ---------------------------------------------------------------- public
    def forecast(self, obs: Observation, est: Estimator, candidates: list[Candidate]) -> list[Forecast]:
        cfg = self.cfg
        point = est.view == "det"
        n = 1 if point else cfg.dynamic.n_scenarios
        self.calls += 1
        d = _Draws(self.seed, int(obs.t // 1440), self.calls, n, point)
        m, P = est.corridor_now(obs)
        if point or est.v <= 0:
            z0 = np.tile(m, (n, 1))
        else:
            w, V = np.linalg.eigh((P + P.T) / 2)
            root = V * np.sqrt(np.clip(w, 0.0, None))
            eps = np.column_stack([d.normal(900, k) for k in range(len(m))])
            z0 = m + eps @ root.T
        out = []
        for cand in candidates:
            arrivals: list = []
            deferred: dict = {}
            for tr in obs.trucks:
                free = self._truck(obs, est, d, z0, tr, cand, deferred, arrivals)
                self._future_trips(obs, est, d, z0, tr.id, tr.trips_started, free, deferred, arrivals)
            for t_unloaded, batch in obs.receiving:
                pc = cfg.plant
                elapsed = obs.t - t_unloaded
                rem = d.residual_lognormal(pc.receiving_mean_min, pc.receiving_cv, elapsed, 700, int(t_unloaded))
                arrivals.append((obs.t + rem, batch.astype(float)))
            for k, (t_disp, s, q, t_pick) in enumerate(obs.couriers):
                self._courier(obs, est, d, z0, s, q, t_disp, t_pick, k, arrivals)
            cost = 0.0
            if cand.courier_supplier >= 0:
                q = np.array(cand.courier_q, dtype=float)
                self._courier(obs, est, d, z0, cand.courier_supplier, q, obs.t, math.nan, 99, arrivals)
                cost += cfg.costs.premium_freight_fixed + cfg.costs.premium_freight_per_km * 2 * float(
                    self.net.dist_km[0, cand.courier_supplier])
            if cand.is_change:
                c = cfg.costs
                cost += c.disruption_per_change + c.disruption_per_stop * cand.stops_changed(obs)
            if cand.remaining is not None:
                tr = obs.trucks[cand.truck]
                cost += (self._route_km(tr.leg_to if tr.status == "travel" else tr.node, cand.remaining)
                         - self._route_km(tr.leg_to if tr.status == "travel" else tr.node, tr.remaining)) \
                    * cfg.costs.truck_per_km
            hot = set(obs.hot_parts) | ({cand.hot_part} if cand.hot_part >= 0 else set())
            line_down, p_starve, margin, p_line = self._line(obs, d, arrivals, hot)
            out.append(Forecast(line_down, p_starve, margin, cost, p_line))
        return out

    def _route_km(self, start: int, stops) -> float:
        km, node = 0.0, start
        for s in stops:
            km += float(self.net.dist_km[node, s])
            node = s
        return km + float(self.net.dist_km[node, 0])
