"""TwinState: every piece of simulation state lives here, not in generator locals.

That makes snapshot() a plain deep copy, which the Phase 3 predictor reads.
Inventory is counted in containers per part, except the open container at the
line, which is counted in units.
"""
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .network import Network

# Locations integrated over time for holding cost (see Twin._loc_units).
LOCATIONS = ("in_transit", "in_plant", "staging", "supermarket", "tugger", "line", "courier", "forklift")


@dataclass
class TruckState:
    id: int
    status: str = "idle"  # idle | travel | late | load | gate | yard | door_wait | unload | dock_out
    node: int = 0
    load: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Phase 3: the trip in progress, which a controller may change between stops
    remaining: list = field(default_factory=list)   # suppliers still to visit, in order
    orders: dict = field(default_factory=dict)      # supplier -> containers per part still to pick
    leg_from: int = 0
    leg_to: int = 0
    leg_start: float = math.nan
    leg_seq: int = 0              # legs driven so far (keys the GPS ping noise)
    leg_end_true: float = math.nan  # TRUTH: only the twin and its ping generator may read this
    waiting_since: float = math.nan  # at a supplier that is not ready
    priority: int = 0             # door priority for the next plant visit (-1 = first)
    status_since: float = 0.0     # when the current status began
    last_change: float = -math.inf  # last executed re-plan for this truck


@dataclass
class TwinState:
    upc: np.ndarray            # units per container, per part
    usage: np.ndarray          # units per vehicle, per part
    kanban_cards: np.ndarray   # cards per part
    on_order: np.ndarray       # released to the supplier, not yet picked up
    in_transit: np.ndarray     # on a truck on the road
    in_plant: np.ndarray       # on a truck at the gate, in the yard or at a door
    staging: np.ndarray        # unloaded, waiting for receiving
    supermarket: np.ndarray    # received, full containers
    tugger: np.ndarray         # on the tugger
    line_full: np.ndarray      # unopened containers at the line
    line_open: np.ndarray      # units left in the open container (0 = none open)
    courier: np.ndarray = None   # on a premium-freight courier (picked, not yet received)
    forklift: np.ndarray = None  # on a hot run from the supermarket to the line
    kanban: deque = field(default_factory=deque)  # pending signals (time, part), FIFO
    trucks: list[TruckState] = field(default_factory=list)
    yard: list = field(default_factory=list)      # trucks waiting for a door
    doors_busy: int = 0
    crews_busy: int = 0
    line_status: str = "off"   # off | run | starved
    vehicles_built: int = 0
    consumed_units: np.ndarray = None
    picked: np.ndarray = None  # containers picked up from suppliers, cumulative
    initial_units: np.ndarray = None

    @classmethod
    def initial(cls, cfg: Config, net: Network) -> "TwinState":
        n = len(net.parts)
        upc = np.array([p.upc for p in net.parts], dtype=np.int64)
        usage = np.array([p.usage for p in net.parts], dtype=np.int64)
        cards = np.array([p.kanban for p in net.parts], dtype=np.int64)
        rate_h = usage * cfg.line.jph
        zeros = lambda: np.zeros(n, dtype=np.int64)  # noqa: E731
        st = cls(
            upc=upc, usage=usage, kanban_cards=cards,
            on_order=zeros(), in_transit=zeros(), in_plant=zeros(), staging=zeros(),
            supermarket=np.array(
                [math.ceil(rate_h[p] * cfg.inventory.initial_cover_h / upc[p]) for p in range(n)],
                dtype=np.int64),
            tugger=zeros(),
            line_full=cards - 1,          # one card is on the open container
            line_open=upc.copy(),
            trucks=[TruckState(i, load=zeros()) for i in range(cfg.fleet.n_trucks)],
            consumed_units=zeros(), picked=zeros(), courier=zeros(), forklift=zeros(),
        )
        st.initial_units = st.units_on_hand()
        return st

    def units_on_hand(self) -> np.ndarray:
        """Units per part physically owned by the plant (excludes on_order)."""
        containers = (self.in_transit + self.in_plant + self.staging + self.supermarket
                      + self.tugger + self.line_full + self.courier + self.forklift)
        return containers * self.upc + self.line_open

    def inventory_position_units(self) -> np.ndarray:
        """On hand plus on order, per part: what the (R,S) rule orders against."""
        return self.units_on_hand() + self.on_order * self.upc

    def line_units(self) -> np.ndarray:
        return self.line_full * self.upc + self.line_open

    def pending_signals(self) -> np.ndarray:
        counts = np.zeros(len(self.upc), dtype=np.int64)
        for _, p in self.kanban:
            counts[p] += 1
        return counts

    def snapshot(self) -> "TwinState":
        return copy.deepcopy(self)
