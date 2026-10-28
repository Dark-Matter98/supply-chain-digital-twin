"""Record the control-tower app's cut (showcase/web, ?video) to MP4, frame by frame.

Serves the project on a local port, drives the installed Chrome with Playwright, and steps
window.__frame(f) through each segment of the cut in story.js (EDIT): the hero day, a stress
test, then the hero page's results and end card. Each story gets its own page; segments are
joined by a crossfade. Screenshots are piped to ffmpeg. Each frame is rendered on demand, so the
video does not depend on machine speed.

    .venv/Scripts/python showcase/record.py --aspect 16x9
    .venv/Scripts/python showcase/record.py --aspect 4x5
    .venv/Scripts/python showcase/record.py --aspect 16x9 --range 300:330 --png outputs/showcase/web_check
"""
from __future__ import annotations

import argparse
import functools
import io
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SIZES = {"16x9": (1920, 1080), "4x5": (1080, 1350)}


class Quiet(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def serve() -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(ROOT)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def cut(edit: dict) -> list[list[tuple[str, int, float]]]:
    """The cut as output frames, each a list of (scenario, story frame, weight). Within a
    crossfade the last frames of one segment blend into the first frames of the next."""
    fps, x = edit["fps"], round(edit["xfade"] * edit["fps"])
    out = []
    for scenario, a, b in edit["segments"]:
        src = [(scenario, f) for f in range(round(a * fps), round(b * fps))]
        if out and x:
            for j in range(x):
                w = (j + 1) / (x + 1)
                (sa, fa, _), = out[len(out) - x + j]
                out[len(out) - x + j] = [(sa, fa, 1 - w), (*src[j], w)]
            src = src[x:]
        out += [[(s, f, 1.0)] for s, f in src]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aspect", default="16x9", choices=SIZES)
    ap.add_argument("--range", default=None, help="output frames a:b of the cut (default: all of it)")
    ap.add_argument("--png", default=None, help="write PNG frames here instead of an MP4 (for checks)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--poster", type=float, default=28.5, help="also save the frame at this second")
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    W, H = SIZES[a.aspect]
    httpd = serve()
    url = f"http://127.0.0.1:{httpd.server_address[1]}/showcase/web/index.html?video"
    out = Path(a.out or ROOT / "outputs" / "showcase" / f"twin_control_tower_{a.aspect}.mp4")
    out.parent.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=not a.headed,
                                     args=["--enable-gpu", "--ignore-gpu-blocklist", "--use-angle=d3d11"])
        pages = {}

        def page_for(scenario: str):
            if scenario not in pages:
                page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
                page.on("console", lambda m: m.type == "error" and print("console:", m.text))
                page.goto(url + (f"&scenario={scenario}" if scenario else ""))
                page.wait_for_function("window.__ready === true", timeout=60000)
                page.evaluate("document.fonts.ready")
                pages[scenario] = page
            return pages[scenario]

        def shot(scenario: str, f: int) -> bytes:
            page = page_for(scenario)
            page.evaluate(f"window.__frame({f})")
            return page.screenshot(type="png")

        edit = page_for("").evaluate("window.__story.edit")
        frames = cut(edit)
        fps = edit["fps"]
        lo, hi = (int(x) for x in a.range.split(":")) if a.range else (0, len(frames))
        poster = int(round(a.poster * fps))

        proc = None
        if a.png:
            Path(a.png).mkdir(parents=True, exist_ok=True)
        else:
            proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "png",
                                     "-framerate", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow",
                                     "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                                    stdin=subprocess.PIPE)
        t0 = time.time()
        for k in range(lo, hi):
            parts = frames[k]
            if len(parts) == 1:
                png = shot(*parts[0][:2])
            else:
                (sa, fa, _), (sb, fb, w) = parts
                im = Image.blend(Image.open(io.BytesIO(shot(sa, fa))).convert("RGB"),
                                 Image.open(io.BytesIO(shot(sb, fb))).convert("RGB"), w)
                buf = io.BytesIO()
                im.save(buf, "PNG", compress_level=1)
                png = buf.getvalue()
            if a.png:
                (Path(a.png) / f"{k:04d}.png").write_bytes(png)
            else:
                proc.stdin.write(png)
                if k == poster:
                    out.with_suffix(".poster.png").write_bytes(png)
            if k % 150 == 0:
                print(f"frame {k}/{hi} ({time.time() - t0:.0f} s)", flush=True)
        browser.close()
    httpd.shutdown()
    if proc:
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg failed")
        print(f"Wrote {out} ({hi - lo} frames, {(hi - lo) / fps:.1f} s) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
