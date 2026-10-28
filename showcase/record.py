"""Record the control-tower app's story (showcase/web, ?video) to MP4, frame by frame.

Serves the project on a local port, drives the installed Chrome with Playwright, steps
window.__frame(f) for every frame of story.js, and pipes the screenshots to ffmpeg. Each
frame is rendered on demand, so the video does not depend on machine speed.

    .venv/Scripts/python showcase/record.py --aspect 16x9
    .venv/Scripts/python showcase/record.py --aspect 4x5
    .venv/Scripts/python showcase/record.py --aspect 16x9 --range 300:330 --png outputs/showcase/web_check
"""
from __future__ import annotations

import argparse
import functools
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aspect", default="16x9", choices=SIZES)
    ap.add_argument("--range", default=None, help="frames a:b (default: the whole story)")
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
        page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        page.on("console", lambda m: m.type == "error" and print("console:", m.text))
        page.goto(url)
        page.wait_for_function("window.__ready === true", timeout=60000)
        page.evaluate("document.fonts.ready")
        fps, total = page.evaluate("[window.__story.fps, window.__story.frames]")
        lo, hi = (int(x) for x in a.range.split(":")) if a.range else (0, total)
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
        for f in range(lo, hi):
            page.evaluate(f"window.__frame({f})")
            png = page.screenshot(type="png")
            if a.png:
                (Path(a.png) / f"{f:04d}.png").write_bytes(png)
            else:
                proc.stdin.write(png)
                if f == poster:
                    out.with_suffix(".poster.png").write_bytes(png)
            if f % 150 == 0:
                print(f"frame {f}/{hi} ({time.time() - t0:.0f} s)", flush=True)
        browser.close()
    httpd.shutdown()
    if proc:
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg failed")
        print(f"Wrote {out} ({hi - lo} frames, {(hi - lo) / fps:.1f} s) in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
