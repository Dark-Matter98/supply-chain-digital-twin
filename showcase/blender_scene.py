"""Build the 3D twin in Blender from a replay and render it.

Run with Blender's own Python (it only reads the replay JSON, never the twin package):

    blender -b --factory-startup -P showcase/blender_scene.py -- \
        --replay showcase/web/data/replay_R2_s115_d10.json --stills outputs/showcase/stills

Units: 1 Blender unit = 1 km on the map. The plant is drawn about 8 x 5 units, far larger
than life, so that its inside (dock, supermarket, tugger loop, line) can be shown. Every
position comes from the replay at time t; this script only interpolates between records.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import random
import sys
from pathlib import Path

import bpy
from mathutils import Vector

# ------------------------------------------------------------------ style
PALETTE = {
    "ground": "#18202b", "ground2": "#1d2734", "tree": "#2b4a3c", "road": "#3a4658", "ring": "#46546a",
    "site": "#c9d1db", "site_roof": "#9aa6b5", "floor": "#d9dee5", "wall": "#aeb8c5",
    "truck": "#eef1f5", "trailer": "#c3cbd6", "bg_truck": "#7c8796", "rack": "#5d6878",
    "stock": "#8fb3d9", "car": "#e9edf2", "conveyor": "#4a5568",
    "accent": "#35c2ff", "amber": "#f2a33a", "red": "#ff4d4d", "green": "#3ddc84", "off": "#55606e",
}
ROUTE_COLORS = ["#4fb3ff", "#ff8a4c", "#3ddc84", "#ffd23f", "#ff7ab6"]   # one per truck, plan overlays


def lin(hex_: str) -> tuple[float, float, float, float]:
    """sRGB hex to linear RGBA, which is what Blender materials take."""
    def ch(c):
        c = int(c, 16) / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    h = hex_.lstrip("#")
    return ch(h[0:2]), ch(h[2:4]), ch(h[4:6]), 1.0


def mix(a, b, f):
    f = max(0.0, min(1.0, f))
    return tuple(x + (y - x) * f for x, y in zip(a, b))


def material(name: str, color: str, emit: float = 0.0, rough: float = 0.75):
    m = bpy.data.materials.new(name)
    if m.node_tree is None:
        m.use_nodes = True
    bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        bsdf = m.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        out = next(n for n in m.node_tree.nodes if n.type == "OUTPUT_MATERIAL")
        m.node_tree.links.new(bsdf.outputs[0], out.inputs[0])
    bsdf.inputs["Base Color"].default_value = lin(color)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Emission Color"].default_value = lin(color)
    bsdf.inputs["Emission Strength"].default_value = emit
    m.diffuse_color = lin(color)
    return m


def set_mat(m, rgba, emit: float) -> None:
    bsdf = next(n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Emission Color"].default_value = rgba
    bsdf.inputs["Emission Strength"].default_value = emit


# ------------------------------------------------------------------ geometry
COLL = None


def box_vf(sx, sy, sz, cx=0.0, cy=0.0, cz=0.0):
    """A box resting on z = cz, centred on (cx, cy)."""
    x0, x1, y0, y1, z0, z1 = cx - sx / 2, cx + sx / 2, cy - sy / 2, cy + sy / 2, cz, cz + sz
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    f = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    return v, f


def cone_vf(r, h, cx, cy, n=6):
    v = [(cx + r * math.cos(2 * math.pi * k / n), cy + r * math.sin(2 * math.pi * k / n), 0.0) for k in range(n)]
    v.append((cx, cy, h))
    f = [tuple(range(n - 1, -1, -1))] + [(k, (k + 1) % n, n) for k in range(n)]
    return v, f


def pin_vf(r, h, n=4):
    """An upside-down pyramid: a map pin whose tip is at the object's origin."""
    v = [(r * math.cos(2 * math.pi * k / n + math.pi / 4), r * math.sin(2 * math.pi * k / n + math.pi / 4), h)
         for k in range(n)] + [(0.0, 0.0, 0.0)]
    f = [tuple(range(n))] + [((k + 1) % n, k, n) for k in range(n)]
    return v, f


def strip_vf(p: Vector, q: Vector, w: float, z: float):
    d = (q - p).normalized()
    n = Vector((-d.y, d.x)) * (w / 2)
    v = [(p.x + n.x, p.y + n.y, z), (q.x + n.x, q.y + n.y, z), (q.x - n.x, q.y - n.y, z), (p.x - n.x, p.y - n.y, z)]
    return v, [(3, 2, 1, 0)]


def merge(parts):
    verts, faces = [], []
    for v, f in parts:
        o = len(verts)
        verts += v
        faces += [tuple(i + o for i in face) for face in f]
    return verts, faces


def obj(name, vf, mat, loc=(0, 0, 0), parent=None):
    me = bpy.data.meshes.new(name)
    me.from_pydata(vf[0], [], vf[1])
    me.update()
    me.materials.append(mat)
    ob = bpy.data.objects.new(name, me)
    ob.location = loc
    COLL.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def empty(name, loc=(0, 0, 0)):
    ob = bpy.data.objects.new(name, None)
    ob.location = loc
    COLL.objects.link(ob)
    return ob


# ------------------------------------------------------------------ layout
# Plant footprint x in [-4, 4], y in [-2.5, 2.5]; a ring road around it, the yard to the south.
RING = [Vector((-6.0, -6.5)), Vector((6.0, -6.5)), Vector((6.0, 4.5)), Vector((-6.0, 4.5))]
RING_C = Vector((0.0, -1.0))
GATE = Vector((-4.5, -6.5))
GATE_IN = Vector((-4.5, -5.9))
DOOR_X = (-2.4, -0.8, 0.8, 2.4)
DOOR_Y = -3.35
YARD = [Vector((-3.6 + 1.05 * k, -5.0)) for k in range(6)]
PARK = [Vector((1.4 + 0.95 * k, -5.0)) for k in range(5)]
SM_ORIGIN, SM_DX, SM_DY = Vector((-3.55, -1.15)), 0.27, 0.55     # supermarket lanes, 10 x 3
# The assembly line runs along the north wall, facing a camera in the south; the tugger loop
# leaves the supermarket and runs along the line-side racks (half a loop = the drop).
LINE_Y, LINE_X0, LINE_X1 = 1.85, -0.75, 3.75
RACK_Y = 1.4
LOOP = [Vector((-0.95, 0.25)), Vector((-0.95, 1.05)), Vector((3.8, 1.05)), Vector((3.8, -0.55)),
        Vector((-0.95, -0.55)), Vector((-0.95, 0.25))]
SOUTH, EAST = -math.pi / 2, 0.0


def sm_pos(p: int) -> Vector:
    return SM_ORIGIN + Vector((SM_DX * (p % 10), SM_DY * (p // 10)))


def rack_pos(p: int, n: int) -> Vector:
    return Vector((LINE_X0 + 0.1 + (LINE_X1 - LINE_X0 - 0.2) * p / (n - 1), RACK_Y))


def perim_s(p: Vector) -> float:
    """Distance along the ring road to the point of the ring nearest p."""
    s, best = 0.0, None
    for a, b in zip(RING, RING[1:] + RING[:1]):
        seg = b - a
        t = max(0.0, min(1.0, (p - a).dot(seg) / seg.length_squared))
        d = (a + seg * t - p).length
        if best is None or d < best[0]:
            best = (d, s + t * seg.length)
        s += seg.length
    return best[1]


RING_L = sum((b - a).length for a, b in zip(RING, RING[1:] + RING[:1]))


def perim_pt(s: float) -> Vector:
    s %= RING_L
    for a, b in zip(RING, RING[1:] + RING[:1]):
        L = (b - a).length
        if s <= L:
            return a + (b - a) * (s / L)
        s -= L
    return RING[0].copy()


def ring_path(p: Vector, q: Vector) -> list[Vector]:
    """Points along the ring road from p to q, the short way round, corners included."""
    s1, s2 = perim_s(p), perim_s(q)
    fwd = (s2 - s1) % RING_L
    step = 1 if fwd <= RING_L / 2 else -1
    dist = fwd if step == 1 else RING_L - fwd
    corners, acc = [], 0.0
    for a, b in zip(RING, RING[1:] + RING[:1]):
        corners.append(acc)
        acc += (b - a).length
    pts = [p]
    hits = sorted(((c - s1) * step) % RING_L for c in corners)
    for h in hits:
        if 1e-6 < h < dist - 1e-6:
            pts.append(perim_pt(s1 + step * h))
    pts.append(q)
    return pts


def ring_entry(target: Vector) -> Vector:
    """Where the straight road from a supplier meets the ring road."""
    d = (target - RING_C).normalized()
    best = None
    for a, b in zip(RING, RING[1:] + RING[:1]):
        e = b - a
        den = d.x * e.y - d.y * e.x
        if abs(den) < 1e-9:
            continue
        w = a - RING_C
        t = (w.x * e.y - w.y * e.x) / den
        u = (w.x * d.y - w.y * d.x) / den
        if t > 0 and 0 <= u <= 1 and (best is None or t < best[0]):
            best = (t, RING_C + d * t)
    return best[1]


# ------------------------------------------------------------------ replay helpers
def interp(xs, ys, t):
    if t <= xs[0]:
        return ys[0]
    if t >= xs[-1]:
        return ys[-1]
    i = bisect.bisect_right(xs, t) - 1
    f = (t - xs[i]) / (xs[i + 1] - xs[i])
    return ys[i] + (ys[i + 1] - ys[i]) * f


def along(pts: list[Vector], f: float) -> tuple[Vector, float]:
    """Point and heading at fraction f of a polyline's length."""
    lens = [(b - a).length for a, b in zip(pts, pts[1:])]
    total = sum(lens) or 1.0
    d = max(0.0, min(1.0, f)) * total
    for (a, b), L in zip(zip(pts, pts[1:]), lens):
        if d <= L or L == lens[-1]:
            g = (b - a)
            return a + g * (d / L if L else 0.0), math.atan2(g.y, g.x)
        d -= L
    g = pts[-1] - pts[-2]
    return pts[-1].copy(), math.atan2(g.y, g.x)


class World:
    def __init__(self, rep: dict):
        self.rep = rep
        self.node = {0: Vector((0.0, 0.0))}
        for s in rep["network"]["suppliers"]:
            self.node[s["id"]] = Vector((s["x_km"], s["y_km"]))
        self.entry = {k: ring_entry(v) for k, v in self.node.items() if k}
        self.n_parts = len(rep["network"]["parts"])
        line = rep["meta"]["line"]
        self.prod = (line["prod_start_h"] * 60.0, line["prod_end_h"] * 60.0)
        self.takt = line["takt_min"]

    def site_spot(self, s: int) -> Vector:
        return self.node[s] + Vector((1.05, -0.75))

    def leg_path(self, i: int, j: int) -> list[Vector]:
        if i == 0:
            return [GATE_IN, GATE] + ring_path(GATE, self.entry[j])[1:] + [self.site_spot(j)]
        if j == 0:
            return [self.site_spot(i), self.entry[i]] + ring_path(self.entry[i], GATE)[1:] + [GATE_IN]
        return [self.site_spot(i), self.site_spot(j)]

    def line_status(self, run: dict, t: float) -> str:
        if not self.prod[0] <= t < self.prod[1]:
            return "off"
        return "stopped" if any(s["t0"] <= t < s["t1"] for s in run["line_stops"]) else "running"

    def running_minutes(self, run: dict, t: float) -> float:
        lo, hi = self.prod[0], min(max(t, self.prod[0]), self.prod[1])
        stopped = sum(max(0.0, min(s["t1"], hi) - max(s["t0"], lo)) for s in run["line_stops"])
        return max(0.0, hi - lo - stopped)


# ------------------------------------------------------------------ the scene
class Scene:
    def __init__(self, rep: dict):
        self.w = World(rep)
        self.rep = rep
        self._legs = self.road_legs()
        self.mats = {k: material(k, v) for k, v in PALETTE.items()}
        self.mats["accent_glow"] = material("accent_glow", PALETTE["accent"], emit=4.0)
        self.mats["amber_glow"] = material("amber_glow", PALETTE["amber"], emit=3.0)
        self.build_ground()
        self.build_roads()
        self.build_plans()
        self.build_sites()
        self.build_plant()
        self.build_vehicles()
        self.build_lights_camera()

    # -- static world
    def build_ground(self):
        obj("ground", box_vf(900, 900, 0.02, cz=-0.03), self.mats["ground"])
        rnd = random.Random(7)
        trees = []
        while len(trees) < 3200:
            p = Vector((rnd.uniform(-130, 120), rnd.uniform(-70, 150)))
            if -8 < p.x < 8 and -9 < p.y < 7:
                continue
            if any((p - n).length < 3.2 for n in self.w.node.values()):
                continue
            if self.near_road(p, 0.9):
                continue
            r = rnd.uniform(0.35, 0.7)
            trees.append(cone_vf(r, r * rnd.uniform(1.6, 2.4), p.x, p.y))
        obj("trees", merge(trees), self.mats["tree"])

    def road_legs(self) -> list[tuple[int, int]]:
        legs = set()
        for run in self.rep["runs"].values():
            for tr in run["trucks"]:
                for x in tr["track"]:
                    if x["state"] == "travel":
                        legs.add(tuple(sorted((x["from"], x["to"]))))
            for c in run["couriers"]:
                legs.add((0, c["supplier"]))
        return sorted(legs)   # only roads that were driven; plans are drawn as overlays (build_plans)

    def leg_line(self, i: int, j: int) -> tuple[Vector, Vector]:
        if i == 0:
            return self.w.entry[j], self.w.node[j]
        return self.w.node[i], self.w.node[j]

    def near_road(self, p: Vector, tol: float) -> bool:
        for i, j in getattr(self, "_legs", []):
            a, b = self.leg_line(i, j)
            ab = b - a
            t = max(0.0, min(1.0, (p - a).dot(ab) / ab.length_squared))
            if (a + ab * t - p).length < tol:
                return True
        return False

    def build_roads(self):
        self.roads = {}
        for i, j in self._legs:
            a, b = self.leg_line(i, j)
            m = material(f"road_{i}_{j}", PALETTE["road"])
            self.roads[(i, j)] = (obj(f"road_{i}_{j}", strip_vf(a, b, 0.32, 0.005), m), m)
        ring = merge([strip_vf(a, b, 0.36, 0.006) for a, b in zip(RING, RING[1:] + RING[:1])]
                     + [strip_vf(GATE, GATE_IN, 0.36, 0.006)])
        obj("ring_road", ring, self.mats["ring"])
        obj("yard", box_vf(10.2, 2.4, 0.01, cx=0.0, cy=-4.95, cz=-0.005), self.mats["ground2"])

    def build_plans(self):
        """Each plan's daily routes as glowing overlays, one colour per truck (all waves), for the
        optimization beat. Hidden unless show_plan() asks for them."""
        self.plan_objs = {}
        for name in ("naive", "optimized"):
            self.plan_objs[name] = []
            for k in range(self.rep["meta"]["fleet"]["trucks"]):
                segs = []
                for r in self.rep["plans"][name]:
                    if r["truck"] == k:
                        s = r["suppliers"]
                        pts = [self.w.entry[s[0]]] + [self.w.node[x] for x in s] + [self.w.entry[s[-1]]]
                        segs += [strip_vf(a, b, 0.24, 0.12 + 0.02 * k) for a, b in zip(pts, pts[1:])]
                if segs:
                    m = material(f"plan_{name}_{k}", ROUTE_COLORS[k % len(ROUTE_COLORS)], emit=3.0)
                    self.plan_objs[name].append(obj(f"plan_{name}_{k}", merge(segs), m))
        self.show_plan(None)

    def show_plan(self, name: str | None) -> None:
        for n, objs in self.plan_objs.items():
            for ob in objs:
                ob.hide_render = ob.hide_viewport = n != name

    def build_sites(self):
        self.site_flags = {}
        for s in self.rep["network"]["suppliers"]:
            p = self.w.node[s["id"]]
            body = merge([box_vf(1.7, 1.2, 0.55, p.x, p.y), box_vf(0.9, 0.7, 0.85, p.x - 0.35, p.y + 0.15)])
            obj(f"site_{s['id']}", body, self.mats["site"])
            m = material(f"site_flag_{s['id']}", PALETTE["site_roof"])
            self.site_flags[s["id"]] = (obj(f"site_roof_{s['id']}", box_vf(1.72, 1.22, 0.05, p.x, p.y, 0.55), m), m)
            obj(f"site_pad_{s['id']}", box_vf(2.6, 2.1, 0.01, p.x + 0.35, p.y - 0.35, -0.004), self.mats["ground2"])

    def build_plant(self):
        mats, n = self.mats, self.w.n_parts
        obj("plant_floor", box_vf(8.0, 5.0, 0.05), mats["floor"])
        walls = [box_vf(8.0, 0.08, 0.45, 0, 2.5), box_vf(0.08, 5.0, 0.45, -4.0, 0), box_vf(0.08, 5.0, 0.45, 4.0, 0)]
        edges = [-4.0] + [x for d in DOOR_X for x in (d - 0.45, d + 0.45)] + [4.0]
        walls += [box_vf(b - a, 0.08, 0.45, (a + b) / 2, -2.5) for a, b in zip(edges[::2], edges[1::2])]
        obj("plant_walls", merge(walls), mats["wall"])
        obj("door_frames", merge([box_vf(0.95, 0.12, 0.06, d, -2.5, 0.45) for d in DOOR_X]), mats["accent"])
        # supermarket: a rack base per lane, and a stock block whose height is the lane's cover
        obj("sm_racks", merge([box_vf(0.22, 0.46, 0.03, sm_pos(p).x, sm_pos(p).y, 0.05) for p in range(n)]),
            mats["rack"])
        self.sm_stock = [obj(f"sm_{p}", box_vf(0.18, 0.4, 1.0, cz=0.0), mats["stock"], (*sm_pos(p), 0.08))
                         for p in range(n)]
        # tugger loop, line conveyor, line-side racks
        obj("tugger_loop", merge([strip_vf(a, b, 0.1, 0.055) for a, b in zip(LOOP, LOOP[1:])]), mats["conveyor"])
        obj("conveyor", box_vf(LINE_X1 - LINE_X0 + 0.3, 0.42, 0.1, (LINE_X0 + LINE_X1) / 2, LINE_Y, 0.05),
            mats["conveyor"])
        self.rack_mats = []
        self.racks = []
        for p in range(n):
            m = material(f"rack_{p}", PALETTE["stock"])
            q = rack_pos(p, n)
            self.racks.append(obj(f"rack_{p}", box_vf(0.11, 0.16, 1.0), m, (q.x, q.y, 0.05)))
            self.rack_mats.append(m)
        self.cars = [obj(f"car_{k}", merge([box_vf(0.26, 0.2, 0.09), box_vf(0.14, 0.17, 0.07, -0.02, 0, 0.09)]),
                         mats["car"], (0, LINE_Y, 0.15)) for k in range(13)]
        obj("andon_pole", box_vf(0.05, 0.05, 1.2, LINE_X1 + 0.1, 2.25, 0.05), mats["wall"])
        self.andon_mat = material("andon", PALETTE["green"], emit=2.0)
        obj("andon_lamp", box_vf(0.24, 0.24, 0.22, LINE_X1 + 0.1, 2.25, 1.25), self.andon_mat)
        # pins over the focus part's supermarket lane and line rack
        fp = self.rep["story"]["focus_part"]
        self.pin_mat = material("pin", PALETTE["accent"], emit=4.0)
        self.pins = [obj(f"pin_{k}", pin_vf(0.12, 0.3), self.pin_mat, (q.x, q.y, z))
                     for k, (q, z) in enumerate(((sm_pos(fp), 0.8), (rack_pos(fp, n), 0.7)))]
        q = rack_pos(fp, n)   # a glowing plate that shows when the focus part is out at the line
        self.alarm = obj("focus_alarm", box_vf(0.34, 0.3, 0.05), material("alarm", PALETTE["red"], emit=5.0),
                         (q.x, q.y, 0.05))
        # staging pallets behind the doors (shown by count), and a focus marker
        self.pallets = [obj(f"pallet_{k}", box_vf(0.17, 0.17, 0.15), mats["trailer"],
                            (DOOR_X[k % 4] - 0.3 + 0.2 * ((k // 4) % 4), -2.1 + 0.2 * (k // 16), 0.05))
                        for k in range(48)]

    def build_vehicles(self):
        m = self.mats
        self.trucks = []
        for k in range(self.rep["meta"]["fleet"]["trucks"]):
            root = empty(f"truck_{k}")
            obj(f"truck_{k}_trailer", box_vf(0.95, 0.36, 0.36, -0.2, 0, 0.06), m["trailer"], parent=root)
            obj(f"truck_{k}_cab", box_vf(0.3, 0.34, 0.32, 0.45, 0, 0.06), m["truck"], parent=root)
            flag_m = material(f"truck_{k}_flag", PALETTE["truck"])
            obj(f"truck_{k}_flag", box_vf(0.95, 0.36, 0.03, -0.2, 0, 0.42), flag_m, parent=root)
            self.trucks.append((root, flag_m))
        self.bg = []
        for k in range(8):
            root = empty(f"bg_{k}")
            obj(f"bg_{k}_body", merge([box_vf(0.95, 0.36, 0.36, -0.2, 0, 0.06), box_vf(0.3, 0.34, 0.3, 0.45, 0, 0.06)]),
                m["bg_truck"], parent=root)
            self.bg.append(root)
        self.vans = []
        for k in range(4):
            root = empty(f"van_{k}")
            obj(f"van_{k}_body", box_vf(0.5, 0.26, 0.26, 0, 0, 0.05), m["accent_glow"], parent=root)
            self.vans.append(root)
        self.tugger = empty("tugger")
        obj("tugger_head", box_vf(0.24, 0.18, 0.16, 0.0, 0, 0.06), m["truck"], parent=self.tugger)
        self.tugger_load = obj("tugger_load", merge([box_vf(0.2, 0.17, 0.13, -0.26 * (i + 1), 0, 0.08)
                                                     for i in range(3)]), m["stock"], parent=self.tugger)
        obj("tugger_carts", merge([box_vf(0.22, 0.18, 0.03, -0.26 * (i + 1), 0, 0.05) for i in range(3)]),
            m["rack"], parent=self.tugger)
        self.forks = []
        for k in range(3):
            root = empty(f"fork_{k}")
            obj(f"fork_{k}_body", merge([box_vf(0.16, 0.11, 0.1, 0, 0, 0.06), box_vf(0.1, 0.1, 0.09, 0.12, 0, 0.06)]),
                m["accent_glow"], parent=root)
            self.forks.append(root)

    def build_lights_camera(self):
        sc = bpy.context.scene
        sun = bpy.data.lights.new("sun", "SUN")
        sun.energy = 3.2
        sun.angle = math.radians(8)
        ob = bpy.data.objects.new("sun", sun)
        ob.rotation_euler = (math.radians(48), 0, math.radians(32))
        COLL.objects.link(ob)
        world = bpy.data.worlds.new("world")
        if world.node_tree is None:
            world.use_nodes = True
        bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
        bg.inputs[0].default_value = lin("#223047")
        bg.inputs[1].default_value = 1.4
        sc.world = world
        cam = bpy.data.cameras.new("cam")
        cam.clip_start, cam.clip_end = 0.05, 2000
        self.cam = bpy.data.objects.new("cam", cam)
        COLL.objects.link(self.cam)
        sc.camera = self.cam

    def look(self, loc, target, lens: float) -> None:
        self.cam.location = Vector(loc)
        self.cam.data.lens = lens
        self.cam.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()

    def anchors(self) -> dict:
        """Screen positions (0-1, y down) of the things the overlay labels point at, as
        [x, y, visible]. Call after apply() and look()."""
        from bpy_extras.object_utils import world_to_camera_view
        s, n = self.rep["story"], self.w.n_parts
        van = next((r for r in self.vans if not r.hide_render), None)
        world = {
            "supplier": (*self.w.node[11], 0.95),
            "truck_in": (*self.trucks[1][0].location[:2], 0.5),
            "dock": (DOOR_X[1], -2.5, 0.55),
            "supermarket": (-2.3, -0.6, 0.7),
            "tugger": (*self.tugger.location[:2], 0.3),
            "line": ((LINE_X0 + LINE_X1) / 2, LINE_Y, 0.35),
            "focus_truck": (*self.trucks[s["focus_truck"]][0].location[:2], 0.5),
            "focus_rack": (*rack_pos(s["focus_part"], n), 1.0),
            "focus_lane": (*sm_pos(s["focus_part"]), 1.1),
            "focus_supplier": (*self.w.node[s["focus_supplier"]], 1.0),
            "courier": (*van.location[:2], 0.35) if van else None,
        }
        sc = bpy.context.scene
        bpy.context.view_layer.update()
        out = {}
        for k, p in world.items():
            if p is None:
                out[k] = [0.0, 0.0, False]
                continue
            v = world_to_camera_view(sc, self.cam, Vector(p))
            vis = v.z > 0 and -0.02 <= v.x <= 1.02 and -0.02 <= v.y <= 1.02
            out[k] = [round(v.x, 4), round(1.0 - v.y, 4), bool(vis)]
        return out

    # -- state at time t
    def apply(self, run_name: str, t: float, focus: dict | None = None) -> None:
        run = self.rep["runs"][run_name]
        focus = focus or self.rep["story"]
        self.apply_traffic(t)
        self.apply_trucks(run, t, focus)
        self.apply_dock_extras(run, t)
        self.apply_plant(run, t, focus)

    def apply_traffic(self, t: float, neutral: bool = False) -> None:
        mult = self.rep["traffic"]["multiplier"]
        k = max(0, min(95, int(t // 15)))
        base, amber, red = lin(PALETTE["road"]), lin(PALETTE["amber"]), lin(PALETTE["red"])
        for (i, j), (_, m) in self.roads.items():
            x = 1.0 if neutral else max(mult[str(n)][k] for n in (i, j) if n)
            if x < 1.15:
                set_mat(m, base, 0.0)
            elif x < 1.45:
                set_mat(m, mix(base, amber, (x - 1.15) / 0.3), 1.2 * (x - 1.15) / 0.3)
            else:
                set_mat(m, mix(amber, red, (x - 1.45) / 0.25), 1.2 + 1.3 * min(1.0, (x - 1.45) / 0.25))

    def truck_pose(self, run: dict, k: int, t: float) -> tuple[Vector, float]:
        track = run["trucks"][k]["track"]
        doors = [d for d in run["dock"] if d["truck"] == k]

        def plant_spot(e) -> tuple[Vector, float]:
            if e["state"] == "gate":
                return GATE_IN.copy(), math.pi / 2
            if e["state"] == "yard":
                return YARD[k % len(YARD)].copy(), SOUTH
            d = next((d for d in doors if d["t_arrive"] is not None and d["t_arrive"] <= e["t0"] + 1e-6
                      and (d["t_leave"] is None or e["t0"] <= d["t_leave"] + 1e-6)), None)
            x = DOOR_X[d["door"]] if d and d["door"] is not None else DOOR_X[k % 4]
            return Vector((x, DOOR_Y)), SOUTH

        def pose(e, tt) -> tuple[Vector, float]:
            if e["state"] == "travel":
                return along(w.leg_path(e["from"], e["to"]), (tt - e["t0"]) / max(e["t1"] - e["t0"], 1e-6))
            if e["state"] in ("load", "late"):
                return w.site_spot(e["at"]), 0.0
            return plant_spot(e)

        w = self.w
        starts = [e["t0"] for e in track]
        i = bisect.bisect_right(starts, t) - 1
        if i < 0:
            return PARK[k].copy(), SOUTH
        e = track[i]
        if t >= e["t1"]:                    # between trips: parked at the plant
            return (PARK[k].copy(), SOUTH) if e.get("at") == 0 else pose(e, e["t1"])
        p, h = pose(e, t)
        if e["state"] not in ("travel", "load", "late") and i > 0:   # glide between plant spots
            prev_p, _ = pose(track[i - 1], track[i - 1]["t1"])
            f = (t - e["t0"]) / 2.0
            if f < 1.0:
                p = prev_p.lerp(p, f)
        return p, h

    def apply_trucks(self, run: dict, t: float, focus: dict) -> None:
        for k, (root, flag) in enumerate(self.trucks):
            p, h = self.truck_pose(run, k, t)
            root.location = (p.x, p.y, 0.0)
            root.rotation_euler = (0, 0, h)
            if k == focus.get("focus_truck"):
                set_mat(flag, lin(PALETTE["amber"]), 2.5)
            else:
                set_mat(flag, lin(PALETTE["truck"]), 0.0)
        # supplier roofs glow amber while a truck waits there for a load that is not ready
        late = {e["at"] for tr in run["trucks"] for e in tr["track"] if e["state"] == "late" and e["t0"] <= t < e["t1"]}
        for s, (_, m) in self.site_flags.items():
            set_mat(m, lin(PALETTE["amber"] if s in late else PALETTE["site_roof"]), 2.0 if s in late else 0.0)
        # couriers
        vans = [c for c in run["couriers"] if c["t_dispatch"] <= t < c["t_received"]]
        for k, root in enumerate(self.vans):
            if k >= len(vans):
                root.hide_render = root.hide_viewport = True
                for ch in root.children:
                    ch.hide_render = ch.hide_viewport = True
                continue
            c = vans[k]
            if t < c["t_at_supplier"]:
                p, h = along(self.w.leg_path(0, c["supplier"]), (t - c["t_dispatch"]) / (c["t_at_supplier"] - c["t_dispatch"]))
            elif t < c["t_pickup"]:
                p, h = self.w.site_spot(c["supplier"]) + Vector((0, -0.45)), 0.0
            elif t < c["t_at_plant"]:
                p, h = along(self.w.leg_path(c["supplier"], 0), (t - c["t_pickup"]) / (c["t_at_plant"] - c["t_pickup"]))
            else:
                p, h = Vector((-3.6, DOOR_Y)), SOUTH
            root.location, root.rotation_euler = (p.x, p.y, 0.0), (0, 0, h)
            for ob in (root, *root.children):
                ob.hide_render = ob.hide_viewport = False

    def apply_dock_extras(self, run: dict, t: float) -> None:
        """Other carriers' trucks: they appear at the gate and leave from the door."""
        live = [d for d in run["dock"] if d["kind"] == "background" and d["t_arrive"] is not None
                and d["t_arrive"] <= t < (d["t_leave"] if d["t_leave"] is not None else 1e9)]
        for k, root in enumerate(self.bg):
            show = k < len(live)
            for ob in (root, *root.children):
                ob.hide_render = ob.hide_viewport = not show
            if not show:
                continue
            d = live[k]
            if d["t_yard"] is None or t < d["t_yard"]:
                p, h = GATE_IN.copy(), math.pi / 2
            elif d["t_door"] is None or t < d["t_door"]:
                p, h = YARD[(k + 3) % len(YARD)].copy(), SOUTH
            else:
                p, h = Vector((DOOR_X[d["door"]], DOOR_Y)), SOUTH
            root.location, root.rotation_euler = (p.x, p.y, 0.0), (0, 0, h)

    def apply_plant(self, run: dict, t: float, focus: dict) -> None:
        w, n = self.w, self.w.n_parts
        cov = run["cover_h"]
        ts = cov["t"]
        fp = focus.get("focus_part")
        for p in range(n):
            sm = interp(ts, cov["supermarket"][p], t)
            self.sm_stock[p].scale = (1, 1, 0.02 + 0.075 * min(sm, 6.0))
            ln = interp(ts, cov["line"][p], t)
            self.racks[p].scale = (1, 1, 0.02 + 0.11 * min(ln, 3.0))
            if ln < 0.05:
                set_mat(self.rack_mats[p], lin(PALETTE["red"]), 3.0)
            elif ln < 0.5:
                set_mat(self.rack_mats[p], lin(PALETTE["amber"]), 1.5)
            elif p == fp:
                set_mat(self.rack_mats[p], lin(PALETTE["accent"]), 1.0)
            else:
                set_mat(self.rack_mats[p], lin(PALETTE["stock"]), 0.0)
            if p == fp:   # the pin and the alarm plate follow the focus part's stock at the line
                col = PALETTE["red"] if ln < 0.05 else PALETTE["amber"] if ln < 0.5 else PALETTE["accent"]
                set_mat(self.pin_mat, lin(col), 4.0)
                self.alarm.hide_render = self.alarm.hide_viewport = ln >= 0.05
        # line: cars advance one spacing per takt while the line runs
        status = w.line_status(run, t)
        set_mat(self.andon_mat, lin({"running": PALETTE["green"], "stopped": PALETTE["red"], "off": PALETTE["off"]}[status]),
                2.0 if status != "off" else 0.3)
        span = LINE_X1 - LINE_X0
        spacing = span / len(self.cars)
        off = (w.running_minutes(run, t) / w.takt) * spacing * 0.35
        for k, car in enumerate(self.cars):
            car.location = (LINE_X0 + ((k * spacing + off) % span), LINE_Y, 0.15)
        # tugger: on its loop since the last departure; loaded on the way out
        deps = [x for x in run["tugger"] if x["t_depart"] <= t]
        if deps:
            d = deps[-1]
            loop = self.rep["meta"]["plant"]["tugger_loop_min"]
            p, h = along(LOOP, (t - d["t_depart"]) / loop)
            self.tugger.location, self.tugger.rotation_euler = (p.x, p.y, 0.0), (0, 0, h)
            loaded = d["containers"] > 0 and t < d["t_drop"]
            self.tugger_load.hide_render = self.tugger_load.hide_viewport = not loaded
        # hot runs: a forklift from the part's supermarket lane straight to its line rack
        runs = [h for h in run["hot_runs"] if h["t_start"] <= t < h["t_drop"] + 1.0]
        for k, root in enumerate(self.forks):
            show = k < len(runs)
            for ob in (root, *root.children):
                ob.hide_render = ob.hide_viewport = not show
            if show:
                h = runs[k]
                a, b = sm_pos(h["part"]), rack_pos(h["part"], n) - Vector((0, 0.22))
                f = (t - h["t_start"]) / max(h["t_drop"] - h["t_start"], 1e-6)
                p, hd = along([a, Vector((a.x, b.y)), b], f)
                root.location, root.rotation_euler = (p.x, p.y, 0.0), (0, 0, hd)
        # staging: containers unloaded and not yet received, one pallet per container (capped)
        unl = {}
        for d in run["dock"]:
            if d["kind"] == "milk" and d["t_unloaded"] is not None:
                unl[(d["truck"], d["t_arrive"])] = d["t_unloaded"]
        n_stage = sum(r["containers"] for r in run["receipts"]
                      if unl.get((r["truck"], r["t_arrive"]), 1e9) <= t < r["t_received"])
        for k, pal in enumerate(self.pallets):
            pal.hide_render = pal.hide_viewport = k >= n_stage


# ------------------------------------------------------------------ render
def setup_render(res: str, samples: int) -> None:
    sc = bpy.context.scene
    items = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items}
    sc.render.engine = "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in items else "BLENDER_EEVEE"
    w, h = (int(x) for x in res.split("x"))
    sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = w, h, 100
    sc.eevee.taa_render_samples = samples
    sc.view_settings.view_transform = "Standard"
    sc.render.image_settings.file_format = "PNG"
    setup_bloom(sc)


def setup_bloom(sc) -> None:
    """A soft glow on emissive things (roads in traffic, the twin's actions, the andon light).
    The compositor API moved between Blender versions, so this tries the 5.x way, then 4.x."""
    try:
        if hasattr(sc, "compositing_node_group"):
            ng = bpy.data.node_groups.new("bloom", "CompositorNodeTree")
            ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
            sc.compositing_node_group = ng
            out = ng.nodes.new("NodeGroupOutput")
        else:
            sc.use_nodes = True
            ng = sc.node_tree
            ng.nodes.clear()
            out = ng.nodes.new("CompositorNodeComposite")
        rl = ng.nodes.new("CompositorNodeRLayers")
        gl = ng.nodes.new("CompositorNodeGlare")
        if "Type" in gl.inputs:
            gl.inputs["Type"].default_value = "Bloom"
        else:
            gl.glare_type = "BLOOM"
        for key, val in (("Threshold", 0.8), ("Highlights Threshold", 0.8), ("Strength", 0.6), ("Size", 0.6)):
            if key in gl.inputs and hasattr(gl.inputs[key], "default_value"):
                gl.inputs[key].default_value = val
        ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
        ng.links.new(gl.outputs["Image"], out.inputs[0])
        sc.render.use_compositing = True
        print("bloom: on")
    except Exception as e:   # a plain render is still correct
        print(f"bloom: off ({e})")


# name, run, minute of day, camera location, target, lens, plan overlay
STILLS = [
    ("01_region_static_1100", "static", 11 * 60, (6, -78, 58), (-4, 6, 0), 30, None),
    ("02_plant_static_1530", "static", 15 * 60 + 30, (-0.5, -13.0, 8.0), (0.2, 0.0, 0), 32, None),
    ("03_plant_twin_1244", "twin", 12 * 60 + 44, (-0.5, -13.0, 8.0), (0.2, 0.0, 0), 32, None),
    ("04_courier_twin_1305", "twin", 13 * 60 + 5, (-12.5, -14.0, 7.0), (-14.0, -3.5, 0), 30, None),
    ("05_plan_naive", "static", 11 * 60, (6, -78, 58), (-4, 6, 0), 30, "naive"),
    ("06_plan_optimized", "static", 11 * 60, (6, -78, 58), (-4, 6, 0), 30, "optimized"),
]


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--stills", default=None, help="render the preview stills into this folder")
    ap.add_argument("--animate", default=None, help="render the video's frames into this folder")
    ap.add_argument("--aspect", default="16x9", choices=("16x9", "4x5"))
    ap.add_argument("--range", default=None, help="frames a:b (default: all)")
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--res", default=None, help="WxH; default 1280x720 for stills, full HD for frames")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--save-blend", default=None)
    a = ap.parse_args(argv)
    rep = json.loads(Path(a.replay).read_text())

    for ob in list(bpy.data.objects):
        bpy.data.objects.remove(ob, do_unlink=True)
    global COLL
    COLL = bpy.context.scene.collection
    sc = Scene(rep)

    if a.stills:
        setup_render(a.res or "1280x720", a.samples)
        out = Path(a.stills).resolve()   # Blender reads a relative path against its own root, not the cwd
        out.mkdir(parents=True, exist_ok=True)
        for name, run, t, loc, target, lens, plan in STILLS:
            sc.apply(run, t)
            sc.show_plan(plan)
            sc.look(loc, target, lens)
            bpy.context.scene.render.filepath = str(out / f"{name}.png")
            bpy.ops.render.render(write_still=True)
            print(f"rendered {name}")
    if a.animate:
        animate(sc, a)
    if a.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(a.save_blend).resolve()))


def animate(sc: Scene, a) -> None:
    """Render the frames of the timeline and a manifest of label anchors for compose.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import timeline as tl
    setup_render(a.res or ("1920x1080" if a.aspect == "16x9" else "1080x1350"), a.samples)
    # Portrait: Blender fits the sensor to the taller side, so widen the lens to keep the width.
    lens_k = 1.0 if a.aspect == "16x9" else 0.8
    out = Path(a.animate).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in a.range.split(":")) if a.range else (0, tl.n_frames())
    man_path = out / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {}
    for f in range(lo, hi, a.step):
        st = tl.frame_state(f)
        sc.apply(st["run"], st["t"])
        sc.show_plan(st["plan"])
        if st["plan"]:   # let the route colours carry the picture
            sc.apply_traffic(st["t"], neutral=True)
        cam = st["cam"]
        if isinstance(cam, dict):   # follow a truck
            p, _ = sc.truck_pose(sc.rep["runs"][st["run"]], cam["id"], st["t"])
            dx, dy, dz = cam["offset"]
            sc.look((p.x + dx, p.y + dy, dz), (p.x, p.y, 0.0), cam["lens"] * lens_k)
        else:
            loc, target, lens = cam
            sc.look(loc, target, lens * lens_k)
        bpy.context.scene.render.filepath = str(out / f"{f:04d}.png")
        bpy.ops.render.render(write_still=True)
        manifest[str(f)] = {"beat": st["beat"], "run": st["run"], "t": round(st["t"], 2), "anchors": sc.anchors()}
        if f % 30 == 0:
            print(f"frame {f} ({st['beat']}, t={st['t']:.0f})", flush=True)
    man_path.write_text(json.dumps(manifest))


if __name__ == "__main__":
    main()
