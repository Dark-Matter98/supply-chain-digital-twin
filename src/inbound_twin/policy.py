"""Re-planning policies: a trigger decides when to look, a gate decides when to act.

Triggers
  never     the static plan runs untouched
  periodic  assess every periodic_min
  event     assess on every event: leg done, supplier not ready, arrival at the
            plant, receipt, line stop
  det       every check_min, project stock with point estimates; assess when any
            part's projected slack drops below m minutes
  fixed     every check_min; assess when P(line stops within trigger_h) > tau,
            using one pooled, never-updated variance
  bayes     the same with the Kalman filter's variance and ping-sharpened legs

Gates
  value     execute the best candidate only if its expected cost, disruption
            included, is below the current plan's (both scored on the same scenarios)
  always    execute the best candidate whenever an assessment finds parts at risk

An assessment builds candidates for the most at-risk parts:
  - the truck carrying the part, or about to pick it up: go to the plant after
    the current stop, or move the part's supplier to the front (then the rest,
    or straight to the plant), optionally with a priority door;
  - premium freight of premium_cover_h of the part from its supplier;
  - a hot run of the part from the supermarket to the line.
Road changes respect the frozen zone (at the plant or under frozen_min from it)
and at least min_change_gap_min between changes to the same truck.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config
from .estimate import Estimator
from .network import Network
from .observe import Observation, TruckObs, observe
from .plan import Plan
from .predict import Candidate, Predictor

TRIGGERS = ("never", "periodic", "event", "det", "fixed", "bayes")
GATES = ("value", "always")
EVENT_KINDS = ("leg_done", "not_ready", "arrive_plant", "received", "line_stop")
PLANT_STATUSES = ("gate", "yard", "door_wait", "unload", "dock_out")


@dataclass(frozen=True)
class PolicySpec:
    trigger: str
    gate: str = "value"
    threshold: float = 0.0   # tau for fixed/bayes, margin m (minutes) for det

    def __post_init__(self):
        if self.trigger not in TRIGGERS or self.gate not in GATES:
            raise ValueError(f"bad policy {self}")

    @property
    def name(self) -> str:
        if self.trigger == "never":
            return "never"
        arg = {"det": f"m={self.threshold:g}", "fixed": f"tau={self.threshold:g}",
               "bayes": f"tau={self.threshold:g}"}.get(self.trigger, "")
        return f"{self.trigger}{'(' + arg + ')' if arg else ''}+{self.gate}"


class Controller:
    def __init__(self, cfg: Config, net: Network, plan: Plan, seed: int, spec: PolicySpec):
        self.cfg, self.net, self.spec = cfg, net, spec
        view = {"det": "det", "fixed": "fixed"}.get(spec.trigger, "bayes")
        self.est = Estimator(cfg, net, view)
        self.pred = Predictor(cfg, net, plan, seed)
        self.decisions: list[dict] = []
        self.fires: list[float] = []      # times the trigger asked for an assessment
        self.checks = 0
        self.rate_h = np.array([p.usage for p in net.parts], dtype=float) * cfg.line.jph

    # ------------------------------------------------------------- triggers
    def on_event(self, twin, kind: str, info: dict) -> None:
        if self.spec.trigger == "event" and kind in EVENT_KINDS:
            self.fires.append(twin.env.now)
            self.assess(twin, f"event:{kind}")

    def process(self, twin):
        spec, dyn = self.spec, self.cfg.dynamic
        if spec.trigger in ("never", "event"):
            return
            yield   # pragma: no cover  (makes this a generator)
        step = dyn.periodic_min if spec.trigger == "periodic" else dyn.check_min
        while True:
            yield twin.env.timeout(step)
            if spec.trigger == "periodic":
                self.fires.append(twin.env.now)
                self.assess(twin, "periodic")
            else:
                self.risk_check(twin)

    def risk_check(self, twin) -> None:
        obs = observe(twin)
        self.est.ingest(obs)
        if obs.line_status == "starved":
            return
        self.checks += 1
        f = self.pred.forecast(obs, self.est, [Candidate()])[0]
        if self.spec.trigger == "det":
            fire = bool((f.margin_min < self.spec.threshold).any())
        else:
            fire = f.p_line > self.spec.threshold
        if fire:
            self.fires.append(obs.t)
            self.assess(twin, "risk", obs)

    # ----------------------------------------------------------- assessment
    def at_risk(self, f) -> list[int]:
        k = self.cfg.dynamic.max_risk_parts
        if self.est.view == "det":
            order = np.argsort(f.margin_min)
            return [int(p) for p in order[:k] if f.margin_min[p] < self.cfg.dynamic.horizon_h * 60]
        order = np.argsort(-f.p_starve)
        return [int(p) for p in order[:k] if f.p_starve[p] > 0]

    def assess(self, twin, why: str, obs: Observation | None = None) -> None:
        if obs is None:
            obs = observe(twin)
            self.est.ingest(obs)
        base = self.pred.forecast(obs, self.est, [Candidate()])[0]
        parts = self.at_risk(base)
        row = {"t": obs.t, "why": why, "parts": tuple(parts), "n_candidates": 0, "executed": False,
               "label": "", "base_cost": float("nan"), "best_cost": float("nan")}
        if parts:
            cands = self.candidates(obs, parts)
            row["n_candidates"] = len(cands)
            if cands:
                # Score the current plan and every candidate in ONE call: same scenarios.
                fs = self.pred.forecast(obs, self.est, [Candidate(), *cands])
                rate = self.cfg.costs.line_down_per_min
                costs = [f.expected_cost(rate) for f in fs]
                best = int(np.argmin(costs[1:])) + 1
                row.update(label=cands[best - 1].label, base_cost=costs[0], best_cost=costs[best])
                act = costs[best] < costs[0] - 1e-9 if self.spec.gate == "value" else True
                if act:
                    self.execute(twin, obs, cands[best - 1])
                    row["executed"] = True
        self.decisions.append(row)

    def _frozen(self, tr: TruckObs, now: float) -> bool:
        if tr.status in PLANT_STATUSES or tr.status == "idle" or (tr.status == "arrived" and tr.node == 0):
            return True
        if now - tr.last_change < self.cfg.dynamic.min_change_gap_min:
            return True
        if tr.status == "travel" and tr.leg_to == 0:
            post = self.est.leg_posterior(tr, now)
            return Estimator.mean_remaining(post) < self.cfg.dynamic.frozen_min
        return False

    def candidates(self, obs: Observation, parts: list[int]) -> list[Candidate]:
        out: dict[tuple, Candidate] = {}
        dyn = self.cfg.dynamic
        for p in parts:
            s = self.net.parts[p].supplier
            for tr in obs.trucks:
                carries = tr.load[p] > 0 or (s in tr.orders and (tr.leg_to == s or tr.node == s))
                upcoming = s in tr.remaining
                if not (carries or upcoming):
                    continue
                waiting = tr.status == "late" or (tr.status == "arrived" and not math.isnan(tr.waiting_since))
                if waiting and tr.load[p] > 0 and not self._frozen(tr, obs.t):
                    self._add(out, Candidate(truck=tr.id, leave_now=True, remaining=(),
                                             label=f"truck {tr.id}: leave supplier {tr.node} now, go to plant"))
                    if tr.remaining:
                        self._add(out, Candidate(truck=tr.id, leave_now=True, remaining=tuple(tr.remaining),
                                                 label=f"truck {tr.id}: leave supplier {tr.node} now, continue"))
                if not self._frozen(tr, obs.t):
                    if carries and tr.remaining:
                        self._add(out, Candidate(truck=tr.id, remaining=(),
                                                 label=f"truck {tr.id}: plant after this stop"))
                        self._add(out, Candidate(truck=tr.id, remaining=(), priority=True,
                                                 label=f"truck {tr.id}: plant after this stop + priority door"))
                    if upcoming:
                        rest = tuple(x for x in tr.remaining if x != s)
                        if tr.remaining[0] != s:
                            self._add(out, Candidate(truck=tr.id, remaining=(s, *rest),
                                                     label=f"truck {tr.id}: supplier {s} first"))
                        if rest:
                            self._add(out, Candidate(truck=tr.id, remaining=(s,),
                                                     label=f"truck {tr.id}: supplier {s}, then plant"))
                if tr.priority == 0 and tr.status in ("travel", "late", "load", "gate"):
                    self._add(out, Candidate(truck=tr.id, priority=True, label=f"truck {tr.id}: priority door"))
            q = np.zeros(len(self.net.parts), dtype=int)
            q[p] = max(1, math.ceil(self.rate_h[p] * dyn.premium_cover_h / self.net.parts[p].upc))
            self._add(out, Candidate(courier_supplier=s, courier_q=tuple(int(x) for x in q),
                                     label=f"courier: part {p} from supplier {s}"))
            if p not in obs.hot_parts:
                self._add(out, Candidate(hot_part=p, label=f"hot run: part {p}"))
        return list(out.values())

    @staticmethod
    def _add(out: dict, c: Candidate) -> None:
        out.setdefault((c.truck, c.remaining, c.priority, c.courier_supplier, c.courier_q, c.hot_part,
                        c.leave_now), c)

    def execute(self, twin, obs: Observation, c: Candidate) -> None:
        now = obs.t
        if c.truck >= 0:
            truck = twin.state.trucks[c.truck]
            if c.leave_now:
                twin.leave_supplier(c.truck)
            if c.remaining is not None:
                changed = c.stops_changed(obs)
                twin.defer_stops(c.truck, [s for s in truck.remaining if s not in c.remaining])
                twin.resequence(c.truck, list(c.remaining))
                truck.last_change = now
                twin.record_action("reroute", truck=c.truck, stops_changed=changed)
            if c.priority:
                twin.prioritize_door(c.truck)
                if c.remaining is None:
                    twin.record_action("priority", truck=c.truck)
        if c.courier_supplier >= 0:
            q = np.array(c.courier_q, dtype=np.int64)
            twin.send_courier(c.courier_supplier, q)
            premium = (self.cfg.costs.premium_freight_fixed + self.cfg.costs.premium_freight_per_km * 2
                       * float(self.net.dist_km[0, c.courier_supplier]))
            twin.record_action("courier", supplier=c.courier_supplier, part=int(np.argmax(q)), premium=premium)
        if c.hot_part >= 0:
            twin.arm_hot_run(c.hot_part, now + self.cfg.dynamic.horizon_h * 60)
            twin.record_action("hot_run", part=c.hot_part)
