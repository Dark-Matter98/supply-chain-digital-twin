"""The showcase video's edit: beats, simulated time, camera, and every caption, label and card.

Plain Python with no imports beyond the standard library, so Blender (which renders the 3D
frames) and compose.py (which draws the overlays) read the same file. Times are seconds of
video; `sim` keys are minutes from 00:00 of the replayed day. See docs/showcase/STORYBOARD.md.
"""
from __future__ import annotations

FPS = 30
DURATION = 42.0


def hm(h: int, m: float = 0.0) -> float:
    return h * 60.0 + m


# Camera keys: (second, location, target, lens). A beat's camera can instead follow an object:
# {"follow": "truck", "id": 4, "offset": (dx, dy, dz), "lens": mm}.
PLANT_WIDE = ((-0.5, -9.5, 6.5), (0.0, -0.3, 0.0), 30)
LINE_A = ((1.5, -5.8, 4.1), (1.8, 1.15, 0.0), 30)    # framed on part 19's line rack (x ~2.2)
LINE_B = ((2.0, -3.7, 2.9), (2.1, 1.3, 0.0), 30)
AERIAL_A = ((4.0, -64.0, 47.0), (-3.0, 4.0, 0.0), 30)
AERIAL_B = ((0.0, -57.0, 41.0), (-3.0, 4.0, 0.0), 30)

BEATS = [
    {"id": "hook", "t0": 0.0, "t1": 4.0, "run": "static",
     "sim": [(0.0, hm(8, 50)), (4.0, hm(9, 6))],
     "cam": [(0.0, (12.0, -80.0, 60.0), (-6.0, 6.0, 0.0), 28), (4.0, (6.0, -62.0, 44.0), (-3.0, 3.0, 0.0), 30)]},
    {"id": "system", "t0": 4.0, "t1": 11.0, "run": "static",
     "sim": [(4.0, hm(10, 25)), (11.0, hm(11, 35))],
     "cam": [(4.0, (6.0, -62.0, 44.0), (-3.0, 3.0, 0.0), 30), (6.2, (-2.0, -21.0, 13.5), (-1.0, -1.5, 0.0), 30),
             (8.6, (-3.4, -9.6, 5.6), (-1.6, -0.9, 0.0), 32), (11.0, (1.4, -7.4, 4.8), (1.1, 1.0, 0.0), 32)]},
    {"id": "problem_road", "t0": 11.0, "t1": 15.0, "run": "static",
     "sim": [(11.0, hm(9, 0)), (15.0, hm(13, 50))],
     "cam": {"follow": "truck", "id": 4, "offset": (2.5, -10.5, 7.5), "lens": 30}},
    {"id": "problem_line", "t0": 15.0, "t1": 19.0, "run": "static",
     "sim": [(15.0, hm(13, 50)), (18.0, hm(15, 11)), (19.0, hm(15, 14))],
     "cam": [(15.0, *LINE_A), (19.0, *LINE_B)]},
    {"id": "insight", "t0": 19.0, "t1": 22.0, "run": "static",
     "sim": [(19.0, hm(15, 14)), (22.0, hm(16, 20))],
     "cam": [(19.0, *PLANT_WIDE), (22.0, (-0.3, -8.8, 6.0), (0.0, -0.2, 0.0), 30)]},
    {"id": "rewind", "t0": 22.0, "t1": 23.0, "run": "static",
     "sim": [(22.0, hm(16, 20)), (23.0, hm(12, 0))],
     "cam": [(22.0, (-0.3, -8.8, 6.0), (0.0, -0.2, 0.0), 30), (23.0, *PLANT_WIDE)]},
    {"id": "twin_sees", "t0": 23.0, "t1": 25.5, "run": "twin",
     "sim": [(23.0, hm(12, 0)), (25.5, hm(12, 32))],
     "cam": [(23.0, *PLANT_WIDE), (25.5, (-0.8, -8.2, 5.6), (-0.4, 0.3, 0.0), 30)]},
    {"id": "twin_acts", "t0": 25.5, "t1": 28.5, "run": "twin",
     "sim": [(25.5, hm(12, 32)), (28.5, hm(13, 45))],
     "cam": [(25.5, (-6.0, -21.0, 13.0), (-6.5, -2.0, 0.0), 30), (28.5, (-9.5, -23.0, 14.5), (-8.5, -1.5, 0.0), 30)]},
    {"id": "twin_holds", "t0": 28.5, "t1": 31.0, "run": "twin",
     "sim": [(28.5, hm(13, 45)), (30.0, hm(15, 8)), (31.0, hm(15, 20))],
     "cam": [(28.5, *LINE_A), (31.0, *LINE_B)]},
    {"id": "plans", "t0": 31.0, "t1": 37.0, "run": "twin",
     "sim": [(31.0, hm(11, 0)), (37.0, hm(11, 0))],
     "cam": [(31.0, *AERIAL_A), (37.0, *AERIAL_B)],
     "plan": [(31.0, "naive"), (33.8, "optimized")]},
    {"id": "end", "t0": 37.0, "t1": 42.0, "run": "twin",
     "sim": [(37.0, hm(15, 10)), (42.0, hm(15, 25))],
     "cam": [(37.0, (0.0, -28.0, 22.0), (0.0, -0.5, 0.0), 30), (42.0, (0.0, -34.0, 29.0), (0.0, 0.0, 0.0), 30)]},
]

# ------------------------------------------------------------------ overlays (compose.py)
# Captions: (from s, to s, headline, sub-line or "").
CAPTIONS = [
    (0.3, 4.0, "What if your plant could see a line stop 3 hours before it happens?", ""),
    (4.4, 11.0, "A digital twin follows every container", "from supplier, by truck, through the dock and supermarket, to the line"),
    (11.3, 15.0, "Traffic slows one truck", "its load for part 19 is now hours late"),
    (15.3, 19.0, "Part 19 runs out. The line stops.", ""),
    (19.2, 22.0, "A truck at the gate is not a part at the line", "the load reached the plant at 14:19; the line waited until 16:15"),
    (23.2, 25.5, "Same day, same traffic, with the twin", "at 12:15 it projects part 19 will run 22 min short, 3 hours ahead"),
    (25.7, 28.5, "It acts before the stop", "rush part 19 to the line, and send a courier to supplier 10"),
    (28.7, 31.0, "15:10 comes and goes. The line keeps running.", ""),
    (31.2, 37.0, "Optimization sets the plan. The twin protects it all day.", ""),
]

# Labels pinned to things in the scene: (from s, to s, anchor, text). Anchors are projected by Blender.
LABELS = [
    (4.6, 6.6, "supplier", "Supplier"),
    (5.2, 7.6, "truck_in", "Milk-run truck"),
    (6.6, 9.0, "dock", "Dock doors"),
    (8.2, 10.6, "supermarket", "Supermarket"),
    (9.0, 11.0, "tugger", "Tugger train"),
    (9.4, 11.0, "line", "Assembly line"),
    (11.3, 15.0, "focus_truck", "Truck 4: part 19's load"),
    (15.3, 19.0, "focus_rack", "Part 19 at the line"),
    (19.2, 20.4, "focus_truck", "Truck 4 at the dock"),
    (20.2, 21.4, "focus_lane", "Load in the supermarket 15:43"),
    (21.2, 22.0, "line", "Line restarts 16:15"),
    (23.4, 25.5, "focus_rack", "Part 19: projected 22 min short"),
    (25.8, 28.5, "courier", "Courier"),
    (25.8, 28.5, "focus_supplier", "Supplier 10"),
    (25.8, 28.5, "focus_truck", "Truck 4, still in traffic"),
    (28.7, 31.0, "focus_rack", "Part 19 at the line"),
]

# Which HUD pieces are on, by beat.
HUD = {
    "hook": {"clock"}, "system": {"clock"},
    "problem_road": {"clock", "run", "line"}, "problem_line": {"clock", "run", "line", "cover", "cost"},
    "insight": {"clock", "run", "line", "cover", "cost"}, "rewind": {"clock", "rewind"},
    "twin_sees": {"clock", "run", "line", "cover", "cost", "actions"},
    "twin_acts": {"clock", "run", "line", "cost", "actions"},
    "twin_holds": {"clock", "run", "line", "cover", "cost", "actions"},
    "plans": {"stats"}, "end": {"endcard"},
}


# ------------------------------------------------------------------ lookup
def _smooth(f: float) -> float:
    f = max(0.0, min(1.0, f))
    return f * f * (3 - 2 * f)


def _lerp(a, b, f):
    return tuple(x + (y - x) * f for x, y in zip(a, b))


def beat_at(sec: float) -> dict:
    for b in BEATS:
        if b["t0"] <= sec < b["t1"]:
            return b
    return BEATS[-1]


def sim_time(b: dict, sec: float) -> float:
    keys = b["sim"]
    if sec <= keys[0][0]:
        return keys[0][1]
    for (s0, t0), (s1, t1) in zip(keys, keys[1:]):
        if sec <= s1:
            return t0 + (t1 - t0) * (sec - s0) / (s1 - s0)
    return keys[-1][1]


def camera(b: dict, sec: float):
    """(location, target, lens), eased between keys; or the follow spec unchanged."""
    cam = b["cam"]
    if isinstance(cam, dict):
        return cam
    if sec <= cam[0][0]:
        return cam[0][1:]
    for k0, k1 in zip(cam, cam[1:]):
        if sec <= k1[0]:
            f = _smooth((sec - k0[0]) / (k1[0] - k0[0]))
            return _lerp(k0[1], k1[1], f), _lerp(k0[2], k1[2], f), k0[3] + (k1[3] - k0[3]) * f
    return cam[-1][1:]


def plan_at(b: dict, sec: float):
    shown = None
    for s, name in b.get("plan", []):
        if sec >= s:
            shown = name
    return shown


def frame_state(frame: int) -> dict:
    sec = frame / FPS
    b = beat_at(sec)
    return {"frame": frame, "sec": sec, "beat": b["id"], "run": b["run"], "t": sim_time(b, sec),
            "cam": camera(b, sec), "plan": plan_at(b, sec)}


def n_frames() -> int:
    return int(round(DURATION * FPS))
