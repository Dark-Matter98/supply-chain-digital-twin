"""Typed configuration loaded from YAML.

Every value in config/base.yaml is illustrative (a synthetic plant), not data
from any real company. Defaults below mirror base.yaml so tests can build a
config without reading a file.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LineCfg:
    jph: float = 50.0
    prod_start_h: float = 6.0
    prod_end_h: float = 22.0

    @property
    def takt_min(self) -> float:
        return 60.0 / self.jph

    @property
    def prod_hours(self) -> float:
        return self.prod_end_h - self.prod_start_h


@dataclass(frozen=True)
class NetworkCfg:
    n_suppliers: int = 15
    parts_per_supplier: int = 2
    radius_km: tuple[float, float] = (10.0, 65.0)
    circuity: float = 1.3
    speed_kmh: float = 60.0
    usage_choices: tuple[int, ...] = (1, 1, 1, 2, 4)
    containers_per_hour: tuple[float, float] = (0.5, 1.1)
    unit_value_median: float = 40.0
    unit_value_sigma: float = 1.2
    unit_value_clip: tuple[float, float] = (2.0, 1500.0)
    travel_sigma: float = 0.15
    travel_sigma_spread: float = 0.0
    p_supplier_late: float = 0.05
    p_late_spread: float = 0.0     # lognormal spread of p_late across suppliers (0 = all equal)
    supplier_late_mean_min: float = 40.0


@dataclass(frozen=True)
class FleetCfg:
    n_trucks: int = 5
    capacity: int = 26
    dispatch_h: tuple[float, ...] = (4.0, 9.0, 14.0, 19.0)
    shift_of_wave: tuple[int, ...] = (0, 0, 1, 1)   # driver shift that runs each wave


@dataclass(frozen=True)
class SupplierOpsCfg:
    load_fixed_min: float = 15.0
    load_per_container_min: float = 0.5


@dataclass(frozen=True)
class PlantCfg:
    gate_min: float = 5.0
    n_doors: int = 4
    n_crews: int = 2
    unload_setup_min: float = 5.0
    unload_per_container_min: float = 1.2
    unload_cv: float = 0.2
    dock_out_min: float = 5.0
    receiving_mean_min: float = 20.0
    receiving_cv: float = 0.3
    operating_window_h: tuple[float, float] = (4.0, 24.0)
    background_trucks_per_h: float = 1.5
    background_containers: tuple[int, int] = (10, 26)

    @property
    def window_min(self) -> float:
        return (self.operating_window_h[1] - self.operating_window_h[0]) * 60.0


@dataclass(frozen=True)
class TuggerCfg:
    capacity: int = 16
    loop_min: float = 30.0
    kanban_margin: float = 0.5


@dataclass(frozen=True)
class TrafficCfg:
    rush_multiplier: float = 1.35
    rush_hours: tuple[tuple[float, float], ...] = ((7.0, 9.0), (16.0, 18.0))
    corridor_phi: float = 0.0       # AR(1) coefficient of corridor congestion per 15-min step
    corridor_sigma: float = 0.0     # stationary std of a corridor's log travel multiplier (0 = off)
    incidents_per_day: float = 0.0  # per corridor (0 = off)
    incident_multiplier: tuple[float, float] = (1.4, 2.2)
    incident_mean_min: float = 90.0


@dataclass(frozen=True)
class DynamicCfg:
    """Phase 3: observation, estimation, prediction and the re-planning controller."""
    ping_min: float = 10.0          # GPS ping interval while driving
    gps_sigma: float = 0.03         # noise on reported leg progress (fraction of the leg)
    check_min: float = 15.0         # risk-check interval for the stockout triggers
    periodic_min: float = 30.0      # assessment interval for the periodic trigger
    horizon_h: float = 6.0          # predictor horizon
    trigger_h: float = 2.0          # P(starve within this) for the probabilistic triggers
    n_scenarios: int = 200          # predictor Monte Carlo scenarios
    frozen_min: float = 30.0        # trucks this close to the plant cannot be re-routed
    min_change_gap_min: float = 60.0  # minimum time between executed changes per truck
    hot_run_min: float = 10.0       # forklift run supermarket -> line, bypassing the tugger
    courier_receiving_min: float = 10.0
    premium_cover_h: float = 2.0    # a courier brings this many production hours of the part
    max_risk_parts: int = 3         # candidates are built for at most this many at-risk parts


@dataclass(frozen=True)
class InventoryCfg:
    lead_margin_h: float = 0.0   # added to each route's planned lead time
    ss_h: float = 2.0
    initial_cover_h: float = 8.0


@dataclass(frozen=True)
class PlanningCfg:
    max_fill: float = 1.0          # plan routes to this share of truck capacity
    slack_min: float = 15.0        # route must end this long before the truck's next wave
    dock_wait_min: float = 13.0    # planned wait for a door and crew (Phase 1 measured)
    solution_limit: int = 2000     # OR-Tools guided local search, final solves
    time_limit_s: float = 120.0    # safety net for final solves; hitting it raises an error
    fast_time_limit_s: float = 60.0  # safety net for frequency-search solves


@dataclass(frozen=True)
class CostCfg:
    truck_per_km: float = 1.20
    truck_per_day: float = 250.0   # lease per truck in the plan's fleet
    driver_per_h: float = 35.0
    driver_min_paid_h: float = 8.0  # a driver shift is paid at least this long
    overtime_factor: float = 1.5    # pay multiplier beyond driver_min_paid_h of shift span
    handling_per_container: float = 4.0
    carrying_rate_per_year: float = 0.20
    space_per_position_day: float = 1.50
    line_down_per_min: float = 5000.0
    premium_freight_fixed: float = 250.0
    premium_freight_per_km: float = 3.0
    disruption_per_change: float = 50.0
    disruption_per_stop: float = 20.0


@dataclass(frozen=True)
class StressCfg:
    """A stress test: one disruption during one window of one day. Off when day < 0."""
    day: int = -1
    start_h: float = 0.0
    end_h: float = 0.0
    supplier_down: int = 0          # this supplier cannot load trucks or couriers in the window
    road_supplier: int = 0          # legs to or from this supplier slow down in the window
    road_multiplier: float = 1.0
    doors_out: int = 0              # dock doors out of service in the window

    @property
    def start_min(self) -> float:
        return self.day * 1440.0 + self.start_h * 60.0

    @property
    def end_min(self) -> float:
        return self.day * 1440.0 + self.end_h * 60.0

    def active(self, t_min: float) -> bool:
        return self.day >= 0 and self.start_min <= t_min < self.end_min


@dataclass(frozen=True)
class Config:
    seed_network: int = 7
    warmup_days: int = 1
    line: LineCfg = field(default_factory=LineCfg)
    network: NetworkCfg = field(default_factory=NetworkCfg)
    fleet: FleetCfg = field(default_factory=FleetCfg)
    supplier_ops: SupplierOpsCfg = field(default_factory=SupplierOpsCfg)
    plant: PlantCfg = field(default_factory=PlantCfg)
    tugger: TuggerCfg = field(default_factory=TuggerCfg)
    traffic: TrafficCfg = field(default_factory=TrafficCfg)
    inventory: InventoryCfg = field(default_factory=InventoryCfg)
    planning: PlanningCfg = field(default_factory=PlanningCfg)
    dynamic: DynamicCfg = field(default_factory=DynamicCfg)
    costs: CostCfg = field(default_factory=CostCfg)
    stress: StressCfg = field(default_factory=StressCfg)


def _tupleize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize(v) for v in value)
    return value


def _build(cls: type, data: dict[str, Any]) -> Any:
    base = cls()
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        current = getattr(base, f.name)
        if is_dataclass(current):
            kwargs[f.name] = _build(type(current), data[f.name] or {})
        else:
            kwargs[f.name] = _tupleize(data[f.name])
    return replace(base, **kwargs)


def validate(cfg: Config) -> Config:
    """Reject configs that cannot describe a working plant."""
    errors = []
    if not 0 <= cfg.line.prod_start_h < cfg.line.prod_end_h <= 24:
        errors.append("production window must satisfy 0 <= start < end <= 24")
    if cfg.line.jph <= 0:
        errors.append("jph must be positive")
    lo, hi = cfg.network.radius_km
    if not 0 < lo <= hi:
        errors.append("radius_km must satisfy 0 < lo <= hi")
    clo, chi = cfg.network.containers_per_hour
    if not 0 < clo <= chi:
        errors.append("containers_per_hour must satisfy 0 < lo <= hi")
    if not 0 <= cfg.network.p_supplier_late <= 1:
        errors.append("p_supplier_late must be in [0, 1]")
    if cfg.fleet.n_trucks < 1 or cfg.fleet.capacity < 1:
        errors.append("need at least one truck with positive capacity")
    if cfg.fleet.n_trucks > cfg.network.n_suppliers:
        errors.append("naive plan needs at least one supplier per truck")
    if any(not 0 <= h < 24 for h in cfg.fleet.dispatch_h):
        errors.append("dispatch_h must be in [0, 24)")
    if list(cfg.fleet.dispatch_h) != sorted(cfg.fleet.dispatch_h):
        errors.append("dispatch_h must be sorted")
    if len(cfg.fleet.shift_of_wave) != len(cfg.fleet.dispatch_h):
        errors.append("shift_of_wave needs one entry per dispatch wave")
    if not 0 < cfg.planning.max_fill <= 1:
        errors.append("planning.max_fill must be in (0, 1]")
    if cfg.planning.slack_min < 0 or cfg.planning.dock_wait_min < 0:
        errors.append("planning slack and dock wait must be non-negative")
    tc = cfg.traffic
    if not 0 <= tc.corridor_phi < 1 or tc.corridor_sigma < 0 or tc.incidents_per_day < 0:
        errors.append("corridor_phi must be in [0, 1); corridor_sigma, incidents_per_day >= 0")
    if not 1 <= tc.incident_multiplier[0] <= tc.incident_multiplier[1]:
        errors.append("incident_multiplier must satisfy 1 <= lo <= hi")
    if cfg.dynamic.n_scenarios < 1 or cfg.dynamic.ping_min <= 0 or cfg.dynamic.check_min <= 0:
        errors.append("dynamic: n_scenarios, ping_min and check_min must be positive")
    if cfg.plant.n_doors < 1 or cfg.plant.n_crews < 1 or cfg.tugger.capacity < 1:
        errors.append("doors, crews and tugger capacity must be >= 1")
    wlo, whi = cfg.plant.operating_window_h
    if not 0 <= wlo < whi <= 24:
        errors.append("operating_window_h must satisfy 0 <= start < end <= 24")
    blo, bhi = cfg.plant.background_containers
    if not 1 <= blo <= bhi:
        errors.append("background_containers must satisfy 1 <= lo <= hi")
    if cfg.inventory.ss_h < 0 or cfg.inventory.lead_margin_h < 0 or cfg.inventory.initial_cover_h < 0:
        errors.append("inventory hours must be non-negative")
    st = cfg.stress
    if st.day >= 0:
        if not 0 <= st.start_h < st.end_h <= 24:
            errors.append("stress window must satisfy 0 <= start_h < end_h <= 24")
        if not 0 <= st.supplier_down <= cfg.network.n_suppliers or not 0 <= st.road_supplier <= cfg.network.n_suppliers:
            errors.append("stress suppliers must be 0 (none) or a supplier id")
        if not 0 <= st.doors_out < cfg.plant.n_doors:
            errors.append("stress doors_out must leave at least one door")
        if st.road_multiplier < 1:
            errors.append("stress road_multiplier must be >= 1")
    if errors:
        raise ValueError("; ".join(errors))
    return cfg


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, *overlays: str | Path) -> Config:
    """Load a YAML config; each overlay YAML is deep-merged on top (e.g. a regime)."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    for extra in overlays:
        with open(extra, encoding="utf-8") as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})
    return validate(_build(Config, data))


def override(cfg: Config, changes: dict[str, Any]) -> Config:
    """Return a copy with dotted-path changes, e.g. {"plant.n_doors": 2}."""
    for path, value in changes.items():
        section, _, name = path.partition(".")
        if not name:
            cfg = replace(cfg, **{section: _tupleize(value)})
            continue
        sub = getattr(cfg, section)
        if name not in {f.name for f in fields(sub)}:
            raise ValueError(f"unknown parameter {path}")
        cfg = replace(cfg, **{section: replace(sub, **{name: _tupleize(value)})})
    return validate(cfg)
