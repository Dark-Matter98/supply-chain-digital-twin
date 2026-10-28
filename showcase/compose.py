"""Draw the captions, HUD, labels and cards over Blender's frames, and encode the MP4.

    .venv/Scripts/python showcase/compose.py --frames outputs/showcase/frames_16x9 --aspect 16x9
    .venv/Scripts/python showcase/compose.py --frames outputs/showcase/frames_4x5 --aspect 4x5

Everything shown comes from the replay (clock, line status, line-down so far, part cover, the
twin's actions) or from timeline.py (captions, labels). Numbers in the cards come from the
replay's `headline`, which export_replay.py reads from the result files.
"""
from __future__ import annotations

import argparse
import bisect
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import timeline as tl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAME = "Prashant Dhakal"
FONT_DIR = Path("C:/Windows/Fonts")
INK, MUTED, PANEL = (242, 245, 249), (166, 178, 194), (11, 15, 23)
ACCENT, AMBER, RED, GREEN = (53, 194, 255), (242, 163, 58), (255, 77, 77), (61, 220, 132)

SIZES = {   # px at the output resolution
    "16x9": dict(m=56, cap=46, sub=28, clock=60, small=23, label=25, card=54, card_t=24, end=66, end_sub=29, gauge=300),
    "4x5": dict(m=44, cap=50, sub=30, clock=60, small=25, label=27, card=54, card_t=25, end=60, end_sub=28, gauge=280),
}


@lru_cache(maxsize=None)
def font(kind: str, px: int):
    name = {"regular": "segoeui.ttf", "semi": "seguisb.ttf", "bold": "segoeuib.ttf", "black": "seguibl.ttf"}[kind]
    return ImageFont.truetype(str(FONT_DIR / name), px)


def fade(sec: float, t0: float, t1: float, d: float = 0.25) -> float:
    return max(0.0, min(1.0, (sec - t0) / d, (t1 - sec) / d))


def rgba(c, a: float):
    return (*c, int(255 * max(0.0, min(1.0, a))))


def hhmm(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60) % 24:02d}:{int(t % 60):02d}"


def wrap(draw, text: str, f, max_w: float, balance: bool = True) -> list[str]:
    """Greedy word wrap; with balance, the narrowest width that keeps the same number of lines,
    so a caption never ends on one stray word."""
    def greedy(width):
        lines, cur = [], ""
        for w in text.split():
            trial = f"{cur} {w}".strip()
            if draw.textlength(trial, font=f) <= width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        return lines + [cur] if cur else lines

    lines = greedy(max_w)
    if balance and len(lines) > 1:
        lo, hi = max_w / len(lines), max_w
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if len(greedy(mid)) == len(lines) else (mid, hi)
        lines = greedy(hi)
    return lines


class Overlay:
    def __init__(self, rep: dict, aspect: str, W: int, H: int):
        self.rep, self.aspect, self.W, self.H = rep, aspect, W, H
        self.s = SIZES[aspect]
        self.portrait = aspect == "4x5"
        self.fp = rep["story"]["focus_part"]
        self.rate = rep["meta"]["line_down_usd_per_min"]
        self.events = self.twin_events()

    # -------------------------------------------------------------- data
    def twin_events(self) -> list[tuple[float, str, tuple]]:
        """The twin's first forecast and its actions, in plain words, for the action log."""
        run = self.rep["runs"]["twin"]
        out = []
        first = next((r for r in run["risk"] if r["least_slack_min"] < 0), None)
        if first:
            out.append((first["t"], f"Forecast: part {first['part']} runs {-first['least_slack_min']:.0f} min short",
                        AMBER))
        words = {"priority": lambda a: f"Priority dock door, truck {a['truck']}",
                 "hot_run": lambda a: f"Rush part {a['part']} to the line",
                 "courier": lambda a: f"Courier to supplier {a['supplier']}, part {a['part']}: ${a['premium_usd']:,.0f}",
                 "reroute": lambda a: f"Re-route truck {a['truck']}"}
        for a in run["actions"]:
            out.append((a["t"], words.get(a["kind"], lambda a: a["kind"])(a), ACCENT))
        return sorted(out, key=lambda e: e[0])[:5]

    def line_status(self, run: dict, t: float) -> str:
        line = self.rep["meta"]["line"]
        if not line["prod_start_h"] * 60 <= t < line["prod_end_h"] * 60:
            return "off"
        return "stopped" if any(s["t0"] <= t < s["t1"] for s in run["line_stops"]) else "running"

    def down_so_far(self, run: dict, t: float) -> float:
        return sum(max(0.0, min(t, s["t1"]) - s["t0"]) for s in run["line_stops"] if s["t0"] < t)

    def cover(self, run: dict, t: float) -> float:
        ts, ys = run["cover_h"]["t"], run["cover_h"]["line"][self.fp]
        i = max(0, min(len(ts) - 2, bisect.bisect_right(ts, t) - 1))
        f = max(0.0, min(1.0, (t - ts[i]) / (ts[i + 1] - ts[i])))
        return ys[i] + (ys[i + 1] - ys[i]) * f

    # -------------------------------------------------------------- pieces
    def panel(self, d, box, a: float, r: int = 14, outline=None):
        d.rounded_rectangle(box, r, fill=rgba(PANEL, 0.72 * a), outline=rgba(outline, a) if outline else None,
                            width=2 if outline else 0)

    def pill(self, d, x, y, text, color, a, filled=False) -> int:
        f = font("bold", self.s["small"])
        w = d.textlength(text, font=f)
        h = self.s["small"] + 16
        d.rounded_rectangle((x, y, x + w + 28, y + h), h // 2, fill=rgba(color if filled else PANEL, (0.9 if filled else 0.7) * a),
                            outline=rgba(color, a), width=2)
        d.text((x + 14, y + h / 2), text, font=f, fill=rgba(PANEL if filled else color, a), anchor="lm")
        return h

    def left_column(self, d, st, hud, a: float = 1.0):
        """Clock, run tag, stock gauge and the twin's action log. Returns the area it used."""
        s, m = self.s, self.s["m"]
        run = self.rep["runs"][st["run"]]
        t = st["t"]
        y = m
        day = f"Day {self.rep['meta']['day']} of a simulated plant"
        widths = [d.textlength(day, font=font("regular", s["small"]))]
        d.text((m, y), hhmm(t), font=font("semi", s["clock"]), fill=rgba(INK, a))
        y += int(s["clock"] * 1.15)
        d.text((m + 2, y), day, font=font("regular", s["small"]), fill=rgba(MUTED, a))
        y += s["small"] + 18
        if "run" in hud:
            twin = st["run"] == "twin"
            tag = "WITH THE TWIN" if twin else "WITHOUT THE TWIN: STATIC PLAN"
            widths.append(d.textlength(tag, font=font("bold", s["small"])) + 28)
            y += self.pill(d, m, y, tag, ACCENT if twin else MUTED, a, filled=twin) + 16
        if "cover" in hud:
            widths.append(s["gauge"] + 100)
            c = self.cover(run, t)
            col = RED if c < 0.05 else AMBER if c < 0.5 else GREEN
            d.text((m, y), f"Part {self.fp} stock at the line", font=font("regular", s["small"]), fill=rgba(MUTED, a))
            y += s["small"] + 8
            gw, gh = s["gauge"], 14
            d.rounded_rectangle((m, y, m + gw, y + gh), 7, fill=rgba((40, 48, 60), a))
            if c > 0.01:
                d.rounded_rectangle((m, y, m + max(gh, gw * min(c, 3.0) / 3.0), y + gh), 7, fill=rgba(col, a))
            d.text((m + gw + 14, y + gh / 2), f"{c:.1f} h", font=font("semi", s["small"] + 2), fill=rgba(col, a), anchor="lm")
            y += gh + 22
        if "actions" in hud and st["run"] == "twin":
            shown = [e for e in self.events if e[0] <= t]
            if shown:
                d.text((m, y), "THE TWIN", font=font("bold", s["small"]), fill=rgba(ACCENT, a))
                y += s["small"] + 8
                for te, text, col in shown:
                    ea = a * min(1.0, (t - te) / 4.0 + 0.35)
                    d.text((m, y), hhmm(te), font=font("semi", s["small"]), fill=rgba(col, ea))
                    d.text((m + 74, y), text, font=font("regular", s["small"]), fill=rgba(INK, ea))
                    widths.append(74 + d.textlength(text, font=font("regular", s["small"])))
                    y += s["small"] + 9
        return (m - 8, m - 8, m + max(widths) + 8, y + 8)

    def right_column(self, d, st, hud, a: float = 1.0):
        """Line status and line-down so far. Returns the area it used, or None."""
        s, m, W = self.s, self.s["m"], self.W
        run = self.rep["runs"][st["run"]]
        w_max, y = 0.0, m
        if "line" in hud:
            status = self.line_status(run, st["t"])
            col, text = {"running": (GREEN, "LINE RUNNING"), "stopped": (RED, "LINE STOPPED"),
                         "off": (MUTED, "LINE OFF")}[status]
            f = font("bold", s["small"])
            w = d.textlength(text, font=f) + 28
            self.pill(d, W - m - w, m + 6, text, col, a, filled=status == "stopped")
            w_max, y = w, m + s["small"] + 22
        if "cost" in hud:
            mins = self.down_so_far(run, st["t"])
            col = RED if mins > 0 else INK
            y = m + s["small"] + 40
            d.text((W - m, y), "Line down today", font=font("regular", s["small"]), fill=rgba(MUTED, a), anchor="ra")
            y += s["small"] + 6
            d.text((W - m, y), f"{mins:.1f} min", font=font("semi", s["clock"] - 14), fill=rgba(col, a), anchor="ra")
            y += s["clock"] - 4
            money = f"${mins * self.rate:,.0f}"
            d.text((W - m, y), money, font=font("semi", s["small"] + 6), fill=rgba(col, a), anchor="ra")
            y += s["small"] + 14
            w_max = max(w_max, d.textlength("Line down today", font=font("regular", s["small"])),
                        d.textlength(money, font=font("semi", s["small"] + 6)))
        return (W - m - w_max - 8, m - 8, W - m + 8, y + 8) if w_max else None

    def caption(self, d, sec: float) -> list:
        s, W, H = self.s, self.W, self.H
        boxes = []
        for t0, t1, head, sub in tl.CAPTIONS:
            a = fade(sec, t0, t1)
            if a <= 0:
                continue
            max_w = W * (0.86 if self.portrait else 0.8)
            fh, fs = font("bold", s["cap"]), font("regular", s["sub"])
            lines = [(x, fh, INK) for x in wrap(d, head, fh, max_w)]
            lines += [(x, fs, MUTED) for x in wrap(d, sub, fs, max_w)] if sub else []
            heights = [int(f.size * 1.28) for _, f, _ in lines]
            width = max(d.textlength(x, font=f) for x, f, _ in lines)
            pad = 26
            bottom = H - s["m"] - 30
            top = bottom - sum(heights) - 2 * pad + 6
            box = (W / 2 - width / 2 - pad, top, W / 2 + width / 2 + pad, bottom)
            self.panel(d, box, a)
            boxes.append(box)
            y = top + pad
            for (x, f, c), h in zip(lines, heights):
                d.text((W / 2, y), x, font=f, fill=rgba(c, a), anchor="ma")
                y += h
        return boxes

    def labels(self, d, sec: float, anchors: dict, reserved: list):
        """Each label tries up-right, up-left, down-right, down-left of its anchor, and takes the
        first spot that stays on screen and clear of the HUD, the caption and other labels."""
        s, m = self.s, self.s["m"]

        def clear(b):
            return (b[0] >= m * 0.5 and b[1] >= m * 0.5 and b[2] <= self.W - m * 0.5 and b[3] <= self.H - m
                    and not any(b[0] < r[2] and r[0] < b[2] and b[1] < r[3] and r[1] < b[3] for r in reserved))

        for t0, t1, key, text in tl.LABELS:
            a = fade(sec, t0, t1, 0.3)
            x, y, vis = anchors.get(key, (0, 0, False))
            if a <= 0 or not vis:
                continue
            px, py = x * self.W, y * self.H
            f = font("semi", s["label"])
            w, h = d.textlength(text, font=f) + 28, s["label"] + 20
            spots = [(px + 46, py - 64 - h), (px - 46 - w, py - 64 - h), (px + 46, py + 52), (px - 46 - w, py + 52)]
            spot = next(((bx, by) for bx, by in spots if clear((bx, by, bx + w, by + h))), None)
            if spot is None:   # nowhere clear: better no label than one over the HUD
                continue
            bx, by = spot
            box = (bx, by, bx + w, by + h)
            reserved.append(box)
            cx = bx if bx > px else bx + w
            cy = by + h if by < py else by
            d.line((px, py, cx, cy), fill=rgba(INK, 0.85 * a), width=2)
            d.ellipse((px - 7, py - 7, px + 7, py + 7), fill=rgba(ACCENT, a), outline=rgba(INK, a), width=2)
            self.panel(d, box, a, r=10, outline=ACCENT)
            d.text((bx + 14, by + h / 2), text, font=f, fill=rgba(INK, a), anchor="lm")

    def rewind(self, im: Image.Image, d, sec: float):
        b = next(b for b in tl.BEATS if b["id"] == "rewind")
        a = fade(sec, b["t0"], b["t1"], 0.15)
        tint = Image.new("RGBA", im.size, rgba((20, 60, 90), 0.35 * a))
        im.alpha_composite(tint)
        for y in range(0, self.H, 6):
            d.line((0, y, self.W, y), fill=rgba((0, 0, 0), 0.18 * a))
        cx, cy, r = self.W / 2, self.H * 0.42, 34
        for k in (0, 1):   # two left-pointing triangles
            ox = cx - 30 + k * 46
            d.polygon([(ox - r, cy), (ox + r * 0.6, cy - r), (ox + r * 0.6, cy + r)], fill=rgba(INK, a))
        d.text((cx, cy + 64), "REWIND", font=font("black", self.s["cap"] + 8), fill=rgba(INK, a), anchor="ma")
        d.text((cx, cy + 64 + self.s["cap"] + 26), "Same day. Same traffic. Now with the twin.",
               font=font("regular", self.s["sub"]), fill=rgba(INK, a), anchor="ma")

    def stats(self, d, sec: float):
        s, m = self.s, self.s["m"]
        h = self.rep["headline"]
        opt, rp = h["optimization"], h["re_planning"]
        cards = []
        if sec < 33.8:
            cards.append((31.0, 33.8, "Naive plan: every supplier, every wave", f"{opt['pickups_per_day']['naive']} pickups a day", ""))
        else:
            cards.append((33.8, 37.0, "Optimized with OR-Tools: routes, pickup frequency, fleet",
                          f"{opt['pickups_per_day']['optimized']} pickups a day",
                          f"{opt['logistics_pct']:.0f}% logistics cost (${-opt['logistics_usd_per_day']:,.0f}/day), 20 test runs"))
        if sec >= 35.0:
            cards.append((35.0, 37.0, "Plus in-day re-planning by the twin", "About half the total cost",
                          f"{rp['R1']['change_pct']:.0f}% and {rp['R2']['change_pct']:.0f}% in two traffic worlds, 20 test runs each"))
        y = m + (110 if self.portrait else 40)
        for t0, t1, title, big, sub in cards:
            a = fade(sec, t0, t1, 0.3)
            ft, fb, fs = font("semi", s["card_t"]), font("bold", s["card"]), font("regular", s["card_t"])
            w = max(d.textlength(title, font=ft), d.textlength(big, font=fb), d.textlength(sub, font=fs) if sub else 0) + 52
            hgt = s["card_t"] + s["card"] + (s["card_t"] + 14 if sub else 0) + 58
            self.panel(d, (m, y, m + w, y + hgt), a, outline=ACCENT)
            d.text((m + 26, y + 20), title, font=ft, fill=rgba(MUTED, a))
            d.text((m + 26, y + 26 + s["card_t"]), big, font=fb, fill=rgba(INK, a))
            if sub:
                d.text((m + 26, y + 36 + s["card_t"] + s["card"]), sub, font=fs, fill=rgba(ACCENT, a))
            y += hgt + 18
        d.text((m + 4, y), "Each colour is one truck's routes across the day's four dispatch waves",
               font=font("regular", s["small"]), fill=rgba(MUTED, fade(sec, 31.0, 37.0, 0.3)))

    def endcard(self, im: Image.Image, d, sec: float):
        b = next(b for b in tl.BEATS if b["id"] == "end")
        a = max(0.0, min(1.0, (sec - b["t0"]) / 0.8))
        im.alpha_composite(Image.new("RGBA", im.size, rgba((8, 11, 17), 0.80 * a)))
        s, W, H = self.s, self.W, self.H
        cx = W / 2
        lines = [
            (0.0, "Inbound Supply Chain Digital Twin", font("bold", s["end"]), INK, 1.25),
            (0.3, "Supply chain  ·  plant process  ·  optimization  ·  system decisions", font("semi", s["end_sub"]), ACCENT, 2.0),
            (0.6, "Discrete-event simulation (SimPy)  ·  Optimization (OR-Tools)", font("regular", s["end_sub"]), MUTED, 1.4),
            (0.6, "Probabilistic forecasting (Kalman filter, Monte Carlo)  ·  Python", font("regular", s["end_sub"]), MUTED, 2.4),
            (1.0, NAME, font("semi", s["end_sub"] + 10), INK, 2.2),
            (1.3, "Synthetic plant and data. Every movement replayed from the simulation.", font("regular", s["small"]), MUTED, 1.0),
        ]
        max_w = W - 2 * s["m"]
        rows = []
        for delay, text, f, col, gap in lines:
            for k, piece in enumerate(wrap(d, text, f, max_w)):
                rows.append((delay, piece, f, col, f.size * (1.25 if k < len(wrap(d, text, f, max_w)) - 1 else gap)))
        total = sum(r[4] for r in rows)
        y = H / 2 - total / 2
        for delay, text, f, col, step in rows:
            la = max(0.0, min(1.0, (sec - b["t0"] - 0.4 - delay) / 0.5))
            d.text((cx, y), text, font=f, fill=rgba(col, la), anchor="ma")
            y += step

    def watermark(self, d, a: float):
        d.text((self.W / 2, self.H - 18), "Simulated plant  ·  every movement replayed from the digital twin",
               font=font("regular", self.s["small"] - 3), fill=rgba(MUTED, 0.8 * a), anchor="ms")

    # -------------------------------------------------------------- frame
    def draw(self, base: Image.Image, frame: int, man: dict) -> Image.Image:
        st = tl.frame_state(frame)
        st["t"] = man.get("t", st["t"])
        sec, beat = st["sec"], st["beat"]
        hud = tl.HUD.get(beat, set())
        im = base.convert("RGBA")
        if beat == "rewind":
            self.rewind(im, ImageDraw.Draw(im), sec)
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(layer)
        if beat == "end":
            self.endcard(im, ImageDraw.Draw(im), sec)
            return im.convert("RGB")
        # The HUD columns go on their own layer so a backing panel can be drawn under them once
        # their size is known; the plant floor is light and text over it needs the contrast.
        hud_layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(hud_layer)
        reserved = []
        if "clock" in hud:
            reserved.append(self.left_column(hd, st, hud))
        right = self.right_column(hd, st, hud)
        if right:
            reserved.append(right)
        for box in reserved:
            self.panel(d, (box[0] - 10, box[1] - 6, box[2] + 10, box[3] + 2), 0.85)
        layer.alpha_composite(hud_layer)
        if "stats" in hud:
            self.stats(d, sec)
        reserved += self.caption(d, sec)
        self.labels(d, sec, man.get("anchors", {}), reserved)
        self.watermark(d, 1.0)
        im.alpha_composite(layer)
        return im.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--aspect", default="16x9", choices=("16x9", "4x5"))
    ap.add_argument("--replay", default=str(ROOT / "showcase" / "web" / "data" / "replay_R2_s115_d10.json"))
    ap.add_argument("--out", default=None, help="MP4 path (default outputs/showcase/twin_showcase_<aspect>.mp4)")
    ap.add_argument("--png", default=None, help="write composed PNGs here instead of an MP4 (for checks)")
    ap.add_argument("--poster", type=float, default=18.4, help="also save the composed frame at this second")
    args = ap.parse_args()
    frames = Path(args.frames)
    man = json.loads((frames / "manifest.json").read_text())
    rep = json.loads(Path(args.replay).read_text())
    keys = sorted(int(k) for k in man)
    W, H = Image.open(frames / f"{keys[0]:04d}.png").size
    ov = Overlay(rep, args.aspect, W, H)
    poster_frame = int(round(args.poster * tl.FPS))

    if args.png:
        out = Path(args.png)
        out.mkdir(parents=True, exist_ok=True)
        for f in keys:
            ov.draw(Image.open(frames / f"{f:04d}.png"), f, man[str(f)]).save(out / f"{f:04d}.png")
        print(f"{len(keys)} composed frames in {out}")
        return

    mp4 = Path(args.out or ROOT / "outputs" / "showcase" / f"twin_showcase_{args.aspect}.mp4")
    mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(tl.FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in keys:
        im = ov.draw(Image.open(frames / f"{f:04d}.png"), f, man[str(f)])
        if f == poster_frame:
            im.save(mp4.with_suffix(".poster.png"))
        proc.stdin.write(im.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"Wrote {mp4} ({len(keys)} frames, {len(keys) / tl.FPS:.1f} s)")


if __name__ == "__main__":
    main()
