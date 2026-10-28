"""Synthetic supply network: one plant (node 0), suppliers, part numbers."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Config
from .rng import Kind, keyed_rng


@dataclass(frozen=True)
class Supplier:
    id: int              # node index, 1..n (0 is the plant)
    x_km: float
    y_km: float
    travel_sigma: float  # lognormal sigma of travel noise on legs touching it
    p_late: float        # chance the load is not ready on arrival


@dataclass(frozen=True)
class Part:
    id: int              # 0..P-1
    supplier: int
    usage: int           # units per vehicle
    upc: int             # units per container
    unit_value: float    # USD per unit
    kanban: int          # line-side cards = max containers at the line


@dataclass(frozen=True, eq=False)
class Network:
    suppliers: tuple[Supplier, ...]
    parts: tuple[Part, ...]
    dist_km: np.ndarray  # road km, (n+1) x (n+1), node 0 = plant
    speed_kmh: float

    def nominal_min(self, i: int, j: int) -> float:
        return float(self.dist_km[i, j]) / self.speed_kmh * 60.0

    def sigma(self, node: int) -> float:
        return 0.0 if node == 0 else self.suppliers[node - 1].travel_sigma

    def parts_of(self, supplier: int) -> tuple[int, ...]:
        return tuple(p.id for p in self.parts if p.supplier == supplier)

    def angle(self, supplier: int) -> float:
        s = self.suppliers[supplier - 1]
        return math.atan2(s.y_km, s.x_km) % (2 * math.pi)


def kanban_cards(rate_containers_per_min: float, loop_min: float, margin: float) -> int:
    """Line-side cards: cover the worst replenishment wait, plus the open container.

    A signal waits up to one loop for the next tugger departure, then half a
    loop for the drop at the line, so the replenishment time is 1.5 loops.
    """
    cover = rate_containers_per_min * 1.5 * loop_min * (1.0 + margin)
    return max(2, math.ceil(cover) + 1)


def generate_network(cfg: Config, seed: int | None = None) -> Network:
    nc = cfg.network
    rng = keyed_rng(cfg.seed_network if seed is None else seed, Kind.NETWORK)
    n = nc.n_suppliers

    angles = rng.uniform(0.0, 2 * math.pi, n)
    radii = rng.uniform(nc.radius_km[0], nc.radius_km[1], n)
    xy = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    pts = np.vstack([[0.0, 0.0], xy])
    dist = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2) * nc.circuity
    sigmas = nc.travel_sigma * np.exp(nc.travel_sigma_spread * rng.standard_normal(n))
    p_late = np.full(n, nc.p_supplier_late)
    if nc.p_late_spread > 0:   # its own keyed stream, so networks without a spread don't change
        z = keyed_rng(cfg.seed_network if seed is None else seed, Kind.NETWORK_EXTRA, 1).standard_normal(n)
        s2 = nc.p_late_spread ** 2
        p_late = np.clip(nc.p_supplier_late * np.exp(nc.p_late_spread * z - s2 / 2), 0.0, 0.5)

    suppliers = tuple(
        Supplier(i + 1, float(xy[i, 0]), float(xy[i, 1]), float(sigmas[i]), float(p_late[i]))
        for i in range(n)
    )

    parts = []
    log_med = math.log(nc.unit_value_median)
    for s in suppliers:
        for _ in range(nc.parts_per_supplier):
            usage = int(rng.choice(nc.usage_choices))
            target_cph = rng.uniform(*nc.containers_per_hour)
            upc = max(1, round(cfg.line.jph * usage / target_cph))
            value = float(np.clip(rng.lognormal(log_med, nc.unit_value_sigma), *nc.unit_value_clip))
            rate_cpm = cfg.line.jph * usage / upc / 60.0
            k = kanban_cards(rate_cpm, cfg.tugger.loop_min, cfg.tugger.kanban_margin)
            parts.append(Part(len(parts), s.id, usage, upc, round(value, 2), k))

    return Network(suppliers, tuple(parts), dist, nc.speed_kmh)
