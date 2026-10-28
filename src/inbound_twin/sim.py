"""The twin: a SimPy discrete-event model of the inbound line-feed system.

A part prevents a line stop only once it is AT the line, so the model follows
every container: supplier -> truck -> gate -> yard -> door (wait for a crew,
unload, dock out) -> staging -> receiving -> supermarket -> tugger -> line-side
rack -> consumed. Other carriers' trucks share the doors and crews.

All state lives in TwinState (see state.py). Generator locals hold only
timing, so a snapshot of TwinState is a complete picture of the plant.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import simpy

from .config import Config
from .network import Network
from .plan import Plan, Route, order_quantities
from .scenario import Scenario
from .state import LOCATIONS, TruckState, TwinState

BACKGROUND_KEY = 1000  # truck keys >= this belong to other carriers
COURIER_KEY = 2000     # premium-freight couriers
SAMPLE_MIN = 5.0
EPS = 1e-6
NAN = float("nan")


@dataclass
class Visit:
    """One truck's pass through the dock."""
    key: int
    day: int
    kind: str              # "milk" | "background"
    n: int                 # containers on board
    t_arrive: float        # at the gate
    t_yard: float = NAN    # gate done, waiting for a door
    t_door: float = NAN    # got a door, waiting for a crew
    t_crew: float = NAN    # unloading starts
    t_unloaded: float = NAN
    t_leave: float = NAN   # door released


@dataclass
class Trip:
    truck: int
    day: int
    t_sched: float
    t_dispatch: float
    t_end: float
    km: float
    n: int                 # containers picked up


class Twin:
    def __init__(self, cfg: Config, net: Network, plan: Plan, seed: int, days: int,
                 check_invariants: bool = False, controller=None):
        self.cfg, self.net, self.plan, self.seed, self.days = cfg, net, plan, seed, days
        self.check_invariants = check_invariants
        self.controller = controller
        self.scen = Scenario(cfg, net, seed)
        self.env = simpy.Environment()
        self.state = TwinState.initial(cfg, net)
        self.doors = simpy.PriorityResource(self.env, capacity=cfg.plant.n_doors)
        self.crews = simpy.Resource(self.env, capacity=cfg.plant.n_crews)
        self.truck_res = [simpy.Resource(self.env, capacity=1) for _ in range(cfg.fleet.n_trucks)]
        self._replenish = self.env.event()
        self._trip_seq = [0] * cfg.fleet.n_trucks

        n_parts = len(net.parts)
        self.visits: list[Visit] = []
        self.trips: list[Trip] = []
        self.segments: list[tuple[int, str, float, float]] = []  # road segments of milk-run trucks
        self.batches: list[tuple[int, float, float, int]] = []   # (truck, t_arrive, t_received, n)
        self.downtime: list[tuple[float, float, tuple[int, ...]]] = []
        self.tugger_loads: list[tuple[float, int]] = []
        self.yard_trace: list[tuple[float, int]] = [(0.0, 0)]
        self.log: list[tuple[float, str, int, int, int]] = []    # (t, event, entity, value, info)
        self.vehicles_by_day = np.zeros(days, dtype=np.int64)
        self.area = {loc: np.zeros((days, n_parts)) for loc in LOCATIONS}  # unit-minutes
        self.peak_positions = np.zeros(days, dtype=np.int64)  # supermarket + staging containers
        self._t_last = 0.0
        self._samples: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
        self.samples: dict[str, np.ndarray] = {}
        # Phase 3
        self.legs: list[tuple[int, int, int, float, float]] = []  # completed (truck, from, to, start, end)
        self.actions: list[dict] = []        # executed re-planning actions and their costs
        self.hot_parts: dict[int, float] = {}  # part -> time the armed hot run expires
        self.stop_log: list[tuple[int, float, bool]] = []  # (supplier, arrival, was it not ready)
        self.in_receiving: dict[tuple[int, int], tuple[float, np.ndarray]] = {}  # unloaded, not received
        self._couriers = 0
        self._trip_proc: dict[int, simpy.Process] = {}   # trucks waiting at a not-ready supplier
        self._leave_now: dict[int, bool] = {}
        self.couriers: dict[int, list] = {}  # in flight: id -> [dispatched, supplier, q, picked up at]
        self.courier_log: list[tuple[int, int, float, float, float, float, int]] = []  # done, for replays:
        #                   (id, supplier, dispatched, at supplier, picked up, received, containers)

    # ------------------------------------------------------------------ run
    def run(self) -> "Twin":
        env = self.env
        env.process(self._line())
        env.process(self._tugger())
        env.process(self._sampler())
        if self.controller is not None:
            env.process(self.controller.process(self))
        st = self.cfg.stress
        if 0 <= st.day < self.days:
            for _ in range(st.doors_out):
                env.process(self._door_out())
        for day in range(self.days):
            for route in self.plan.routes:
                env.process(self._trip(route, day))
            for i, (t, n) in enumerate(self.scen.background_arrivals(day)):
                env.process(self._background(day, i, t, n))
        env.run(until=self.days * 1440.0)
        self._integrate(self.days * 1440.0)
        t, sm, line, pipe = zip(*self._samples)
        self.samples = {"t": np.array(t), "supermarket": np.array(sm),
                        "line": np.array(line), "pipeline": np.array(pipe)}
        return self

    # ------------------------------------------------- holding-cost integrals
    def _loc_units(self) -> dict[str, np.ndarray]:
        s = self.state
        return {"in_transit": s.in_transit * s.upc, "in_plant": s.in_plant * s.upc,
                "staging": s.staging * s.upc, "supermarket": s.supermarket * s.upc,
                "tugger": s.tugger * s.upc, "line": s.line_units(),
                "courier": s.courier * s.upc, "forklift": s.forklift * s.upc}

    def _integrate(self, now: float) -> None:
        t = self._t_last
        if now <= t:
            return
        units = self._loc_units()
        while t < now:
            d = int(t // 1440)
            if d >= self.days:
                break
            t_next = min(now, (d + 1) * 1440.0)
            for loc, v in units.items():
                self.area[loc][d] += v * (t_next - t)
            t = t_next
        self._t_last = now

    @contextmanager
    def _mutate(self):
        """Wrap every inventory change: integrate the old levels, then note the peak."""
        self._integrate(self.env.now)
        yield
        d = min(int(self.env.now // 1440), self.days - 1)
        s = self.state
        self.peak_positions[d] = max(self.peak_positions[d], int((s.supermarket + s.staging).sum()))

    def _log(self, event: str, entity: int, value: int = 0, info: int = -1) -> None:
        self.log.append((self.env.now, event, entity, value, info))

    # ------------------------------------------------------------ processes
    def _segment(self, truck: TruckState, kind: str, duration: float):
        if duration <= 0:
            return
        truck.status = kind
        truck.status_since = t0 = self.env.now
        try:
            yield self.env.timeout(duration)
        finally:   # also record a segment cut short by an interrupt
            self.segments.append((truck.id, kind, t0, self.env.now))

    def _drive(self, truck: TruckState, frm: int, to: int):
        """One leg. The twin knows its true end (for GPS pings); the observation layer does not."""
        tt = self.scen.travel_min(frm, to, self.env.now, truck.id)
        truck.leg_from, truck.leg_to, truck.leg_start = frm, to, self.env.now
        truck.leg_end_true = self.env.now + tt
        truck.leg_seq += 1
        yield from self._segment(truck, "travel", tt)
        if tt > 0:
            self.legs.append((truck.id, frm, to, truck.leg_start, self.env.now))
        truck.leg_start = truck.leg_end_true = math.nan
        truck.node = to
        truck.status, truck.status_since = "arrived", self.env.now
        self._emit("leg_done", truck=truck.id, node=to)

    def _trip(self, route: Route, day: int):
        env, s, cfg = self.env, self.state, self.cfg
        t_sched = day * 1440.0 + route.dispatch_min
        yield env.timeout(t_sched - env.now)
        tr = route.truck
        truck = s.trucks[tr]
        with self.truck_res[tr].request() as req:
            yield req                              # waits if the previous trip ran late
            t0 = env.now
            seq = self._trip_seq[tr]
            self._trip_seq[tr] += 1
            orders = order_quantities(s, self.net, cfg, self.plan, route, t0)
            with self._mutate():
                for q in orders.values():
                    s.on_order += q
            self._log("dispatch", tr, int(sum(q.sum() for q in orders.values())), int(t0 - t_sched))

            # The rest of the trip lives in TruckState so a controller can change
            # it between stops; the stop being driven to is committed.
            truck.remaining = list(route.suppliers)
            truck.orders = dict(orders)
            truck.priority = 0
            km, node, so = 0.0, 0, cfg.supplier_ops
            while truck.remaining:
                sup = truck.remaining.pop(0)
                km += float(self.net.dist_km[node, sup])
                yield from self._drive(truck, node, sup)
                node = sup
                delay = self.scen.supplier_delay_min(sup, env.now)
                self.stop_log.append((sup, env.now, delay > 0))
                if delay > 0:
                    truck.waiting_since = env.now
                    self._trip_proc[tr] = env.active_process
                    try:
                        self._emit("not_ready", truck=tr, supplier=sup)
                        if self._leave_now.pop(tr, False):   # decided inside this very process
                            raise simpy.Interrupt("leave")
                        yield from self._segment(truck, "late", delay)
                    except simpy.Interrupt:
                        # The controller sent the truck on without this load; the
                        # order goes back to the (R,S) rule for the next pickup.
                        truck.waiting_since = math.nan
                        with self._mutate():
                            s.on_order -= truck.orders.pop(sup)
                        self._log("left_supplier", tr, 0, sup)
                        continue
                    finally:
                        self._trip_proc.pop(tr, None)
                    truck.waiting_since = math.nan
                q = truck.orders[sup]
                yield from self._segment(
                    truck, "load", so.load_fixed_min + so.load_per_container_min * int(q.sum()))
                with self._mutate():
                    s.on_order -= q
                    s.in_transit += q
                    s.picked += q
                    truck.load += q
                    del truck.orders[sup]
                self._log("pickup", tr, int(q.sum()), sup)
            km += float(self.net.dist_km[node, 0])
            yield from self._drive(truck, node, 0)

            n = int(truck.load.sum())
            if n > 0:
                t_arrive = env.now
                with self._mutate():
                    s.in_transit -= truck.load
                    s.in_plant += truck.load

                def unloaded():
                    batch = truck.load.copy()
                    with self._mutate():
                        s.in_plant -= batch
                        s.staging += batch
                        truck.load[:] = 0
                    self._log("unloaded", tr, int(batch.sum()))
                    self.in_receiving[(tr, seq)] = (env.now, batch)
                    env.process(self._receive(tr, day, seq, batch, t_arrive))

                yield from self._dock(tr, day, seq, n, "milk", truck, unloaded)
            truck.status = "idle"
            truck.status_since = env.now
            self.trips.append(Trip(tr, day, t_sched, t0, env.now, km, n))

    def _background(self, day: int, idx: int, t: float, n: int):
        yield self.env.timeout(t - self.env.now)
        yield from self._dock(BACKGROUND_KEY + idx, day, 0, n, "background")

    def _dock(self, key: int, day: int, seq: int, n: int, kind: str,
              truck: TruckState | None = None, on_unloaded=None):
        env, s, pc = self.env, self.state, self.cfg.plant

        def status(x: str) -> None:
            if truck is not None:
                truck.status = x
                truck.status_since = env.now

        v = Visit(key, day, kind, n, t_arrive=env.now)
        self.visits.append(v)
        if truck is not None:
            self._emit("arrive_plant", truck=truck.id)
        status("gate")
        yield env.timeout(pc.gate_min)
        v.t_yard = env.now
        entry = (day, key)
        s.yard.append(entry)
        self.yard_trace.append((env.now, len(s.yard)))
        status("yard")
        with self.doors.request(priority=truck.priority if truck is not None else 0) as door:
            yield door
            s.yard.remove(entry)
            self.yard_trace.append((env.now, len(s.yard)))
            v.t_door = env.now
            s.doors_busy += 1
            status("door_wait")
            with self.crews.request() as crew:
                yield crew
                v.t_crew = env.now
                s.crews_busy += 1
                status("unload")
                yield env.timeout(self.scen.unload_min(n, day, key, seq))
                s.crews_busy -= 1
                v.t_unloaded = env.now
            if on_unloaded is not None:
                on_unloaded()
            status("dock_out")
            yield env.timeout(pc.dock_out_min)
            s.doors_busy -= 1
            v.t_leave = env.now

    def _door_out(self):
        """Stress test: a dock door out of service. It takes the next door that frees up at
        the window's start, ahead of any waiting truck, and gives it back at the end."""
        st = self.cfg.stress
        yield self.env.timeout(st.start_min - self.env.now)
        with self.doors.request(priority=-100) as door:
            yield door
            self.state.doors_busy += 1
            self._log("door_out", 0)
            yield self.env.timeout(max(0.0, st.end_min - self.env.now))
            self.state.doors_busy -= 1
            self._log("door_back", 0)

    def _receive(self, truck: int, day: int, seq: int, batch: np.ndarray, t_arrive: float):
        yield self.env.timeout(self.scen.receiving_min(day, truck, seq))
        s = self.state
        with self._mutate():
            s.staging -= batch
            s.supermarket += batch
        self.batches.append((truck, t_arrive, self.env.now, int(batch.sum())))
        self.in_receiving.pop((truck, seq), None)
        self._log("received", truck, int(batch.sum()))
        self._after_receipt(batch)
        self._emit("received", truck=truck)

    def _tugger(self):
        env, s, tc = self.env, self.state, self.cfg.tugger
        while True:
            kept, n = [], 0
            with self._mutate():
                while s.kanban:
                    t_sig, p = s.kanban.popleft()
                    if n < tc.capacity and s.supermarket[p] > 0:
                        s.supermarket[p] -= 1
                        s.tugger[p] += 1
                        n += 1
                    else:
                        kept.append((t_sig, p))   # stays pending, order preserved
                s.kanban.extend(kept)
            self.tugger_loads.append((env.now, n))
            yield env.timeout(tc.loop_min / 2)
            if n:
                with self._mutate():
                    s.line_full += s.tugger
                    s.tugger[:] = 0
                self._log("tugger_drop", 0, n)
                fired, self._replenish = self._replenish, env.event()
                fired.succeed()
            yield env.timeout(tc.loop_min / 2)

    def _short_parts(self) -> np.ndarray:
        s = self.state
        return np.flatnonzero(s.line_open + s.line_full * s.upc < s.usage)

    def _consume(self) -> None:
        s, now = self.state, self.env.now
        for p in range(len(s.upc)):
            need = int(s.usage[p])
            while need:
                if s.line_open[p] == 0:            # open the next full container
                    s.line_full[p] -= 1
                    s.line_open[p] = s.upc[p]
                take = min(need, int(s.line_open[p]))
                s.line_open[p] -= take
                need -= take
                if s.line_open[p] == 0:            # empty container: card goes back
                    s.kanban.append((now, p))
                    if p in self.hot_parts:
                        self._try_hot_run(p)
        s.consumed_units += s.usage

    def _line(self):
        env, s, lc = self.env, self.state, self.cfg.line
        start, end = lc.prod_start_h * 60.0, lc.prod_end_h * 60.0
        while True:
            now = env.now
            day0 = (now // 1440) * 1440.0
            tod = now - day0
            if tod < start - EPS:
                s.line_status = "off"
                yield env.timeout(day0 + start - now)
                continue
            if tod >= end - EPS:
                s.line_status = "off"
                yield env.timeout(day0 + 1440.0 + start - now)
                continue
            short = self._short_parts()
            if short.size:
                t0, parts = now, tuple(int(p) for p in short)
                s.line_status = "starved"
                self._log("line_stop", parts[0], len(parts))
                self._emit("line_stop", parts=parts)
                window_end = day0 + end
                while short.size and env.now < window_end - EPS:
                    yield self._replenish | env.timeout(window_end - env.now)
                    short = self._short_parts()
                self.downtime.append((t0, env.now, parts))
                self._log("line_resume", parts[0], int(round(env.now - t0)))
                continue
            s.line_status = "run"
            with self._mutate():
                self._consume()
            s.vehicles_built += 1
            self.vehicles_by_day[int(now // 1440)] += 1
            yield env.timeout(lc.takt_min)

    def _sampler(self):
        while True:
            s = self.state
            self._samples.append((self.env.now, s.supermarket * s.upc, s.line_units(),
                                  (s.in_transit + s.in_plant + s.staging) * s.upc))
            if self.check_invariants:
                self.assert_invariants()
            yield self.env.timeout(SAMPLE_MIN)

    # ------------------------------------------------- Phase 3: events and actions
    def _emit(self, kind: str, **info) -> None:
        """Tell the controller (if any) that something it may react to happened."""
        if self.controller is not None:
            self.controller.on_event(self, kind, info)

    def record_action(self, kind: str, *, truck: int = -1, supplier: int = -1, part: int = -1,
                      stops_changed: int = 0, premium: float = 0.0) -> None:
        c = self.cfg.costs
        self.actions.append({"t": self.env.now, "day": int(self.env.now // 1440), "kind": kind,
                             "truck": truck, "supplier": supplier, "part": part,
                             "premium": premium,
                             "disruption": c.disruption_per_change + c.disruption_per_stop * stops_changed})

    def defer_stops(self, truck_id: int, suppliers) -> int:
        """Drop future stops from a trip; their orders go back to the (R,S) rule."""
        truck, s = self.state.trucks[truck_id], self.state
        n = 0
        with self._mutate():
            for sup in list(suppliers):
                if sup in truck.remaining:
                    truck.remaining.remove(sup)
                    s.on_order -= truck.orders.pop(sup)
                    n += 1
        return n

    def leave_supplier(self, truck_id: int) -> bool:
        """Stop waiting at a supplier that is not ready; the truck moves on without that load."""
        proc = self._trip_proc.get(truck_id)
        if proc is None or math.isnan(self.state.trucks[truck_id].waiting_since):
            return False
        if proc is self.env.active_process:   # SimPy forbids a process interrupting itself
            self._leave_now[truck_id] = True
        else:
            proc.interrupt("leave")
        return True

    def resequence(self, truck_id: int, order) -> None:
        truck = self.state.trucks[truck_id]
        if sorted(order) != sorted(truck.remaining):
            raise ValueError("a re-sequence must keep the same remaining stops")
        truck.remaining = list(order)

    def prioritize_door(self, truck_id: int) -> None:
        self.state.trucks[truck_id].priority = -1

    def send_courier(self, supplier: int, q: np.ndarray) -> None:
        self._couriers += 1
        self.env.process(self._courier(supplier, q.copy(), self._couriers))

    def _courier(self, supplier: int, q: np.ndarray, cid: int):
        """Premium freight: a van fetches q from one supplier and it is received express."""
        env, s, cfg = self.env, self.state, self.cfg
        key = COURIER_KEY + cid
        self.couriers[cid] = [env.now, supplier, q.copy(), math.nan]
        with self._mutate():
            s.on_order += q
        yield env.timeout(self.scen.travel_min(0, supplier, env.now, key))
        self.couriers[cid].append(env.now)   # at the supplier (index 4; observe() reads 0-3)
        yield env.timeout(self.scen.supplier_delay_min(supplier, env.now))
        yield env.timeout(cfg.supplier_ops.load_fixed_min + cfg.supplier_ops.load_per_container_min * int(q.sum()))
        with self._mutate():
            s.on_order -= q
            s.courier += q
            s.picked += q
        self.couriers[cid][3] = env.now
        yield env.timeout(self.scen.travel_min(supplier, 0, env.now, key))
        yield env.timeout(cfg.dynamic.courier_receiving_min)
        with self._mutate():
            s.courier -= q
            s.supermarket += q
        t_disp, _, _, t_pick, t_at = self.couriers.pop(cid)
        self.courier_log.append((cid, supplier, t_disp, t_at, t_pick, env.now, int(q.sum())))
        self._log("courier_received", supplier, int(q.sum()))
        self._after_receipt(q)
        self._emit("received", truck=key)

    def arm_hot_run(self, part: int, until: float) -> None:
        """Send the part's next containers straight to the line, skipping the tugger loop."""
        self.hot_parts[part] = until
        self._try_hot_run(part)

    def _after_receipt(self, batch: np.ndarray) -> None:
        for p in [p for p in self.hot_parts if batch[p] > 0]:
            self._try_hot_run(p)

    def _try_hot_run(self, p: int) -> None:
        s = self.state
        if self.hot_parts.get(p, -1.0) < self.env.now:
            self.hot_parts.pop(p, None)
            return
        n = min(sum(1 for _, q in s.kanban if q == p), int(s.supermarket[p]))
        if n == 0:
            return
        with self._mutate():
            taken, kept = 0, []
            for sig in s.kanban:            # fill this part's oldest signals first
                if sig[1] == p and taken < n:
                    taken += 1
                else:
                    kept.append(sig)
            s.kanban.clear()
            s.kanban.extend(kept)
            s.supermarket[p] -= n
            s.forklift[p] += n
        del self.hot_parts[p]
        self.env.process(self._hot_run(p, n))

    def _hot_run(self, p: int, n: int):
        yield self.env.timeout(self.cfg.dynamic.hot_run_min)
        s = self.state
        with self._mutate():
            s.forklift[p] -= n
            s.line_full[p] += n
        self._log("hot_run", p, n)
        fired, self._replenish = self._replenish, self.env.event()
        fired.succeed()

    # ------------------------------------------------------------ invariants
    def assert_invariants(self) -> None:
        s, cfg = self.state, self.cfg
        for name in ("on_order", "in_transit", "in_plant", "staging", "supermarket",
                     "tugger", "line_full", "line_open", "courier", "forklift"):
            assert (getattr(s, name) >= 0).all(), f"negative {name} at t={self.env.now}"
        balance = s.initial_units + s.picked * s.upc - s.consumed_units - s.units_on_hand()
        assert not balance.any(), f"material balance off at t={self.env.now}: {balance}"
        cards = s.line_full + (s.line_open > 0) + s.pending_signals() + s.tugger + s.forklift
        assert np.array_equal(cards, s.kanban_cards), f"kanban cards off at t={self.env.now}"
        assert s.doors_busy <= cfg.plant.n_doors
        assert s.crews_busy <= cfg.plant.n_crews
        assert s.tugger.sum() <= cfg.tugger.capacity
        assert (s.line_full + (s.line_open > 0) <= s.kanban_cards).all()
        for t in s.trucks:   # every order still on a truck's list is counted as on order
            assert all((q >= 0).all() for q in t.orders.values())
        assert all(t.load.sum() <= cfg.fleet.capacity for t in s.trucks)
