"""Render assets/hero.png and assets/flow.png from a REAL demo run.

    python scripts/make_assets.py

1. Runs the offline demo (synthetic ffmpeg video + hand-written sample transcript,
   heuristic picker) through the real pipeline: face tracking, cropping, caption burn.
2. Grabs frames from the rendered 9:16 shorts and the 16:9 source with ffmpeg.
3. Lays them out with the highlight timeline in HTML and screenshots it with Playwright.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shorts_forge.demo import demo_segments, make_demo_video  # noqa: E402
from shorts_forge.highlights import hook_features, split_sentences  # noqa: E402
from shorts_forge.pipeline import Options, run  # noqa: E402

WORK = ROOT / "build" / "assets-work"
ASSETS = ROOT / "assets"
FONT = Path("/usr/share/fonts/opentype/inter")


class SampleTranscriber:
    name = "sample-transcript"

    def transcribe(self, media, language=None):
        return demo_segments()


def b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def frame(video: Path, t: float, out: Path, width: int) -> Path:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t:.2f}", "-i", str(video),
                    "-frames:v", "1", "-vf", f"scale={width}:-2:flags=lanczos", str(out)], check=True)
    return out


def shoot(html: Path, png: Path, w: int, h: int) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
        page.goto(html.as_uri())
        page.wait_for_timeout(400)
        page.screenshot(path=str(png))
        b.close()


def main() -> None:
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    ASSETS.mkdir(exist_ok=True)
    src = make_demo_video(WORK / "demo-source.mp4")
    res = run(src, WORK / "out", SampleTranscriber(), None, Options(count=2), log=print)
    m = json.loads(res.manifest.read_text())

    # phones: title card + two caption moments of clip 1, one of clip 2
    c1, c2 = res.clips[0], res.clips[1]
    shots = [(c1, 1.0), (c1, 8.25), (c2, 5.2)]
    phones = []
    for i, (c, t) in enumerate(shots):
        f = frame(c.video, t, WORK / f"phone{i}.png", 540)
        phones.append({"img": b64(f), "score": c.highlight.hook_score, "idx": c.index,
                       "cut": f"{c.start:.1f}s - {c.end:.1f}s", "t": t})
    thumbs = [b64(frame(src, t, WORK / f"src{i}.png", 260))
              for i, t in enumerate([4, 14, 24, 34, 44, 54, 64, 74])]

    sentences = split_sentences(demo_segments())
    bars = []
    for s in sentences:
        sc, _ = hook_features(s.text, s.text, s.end - s.start)
        bars.append({"start": s.start, "end": s.end, "score": sc, "text": s.text})
    data = {"duration": m["duration"], "bars": bars, "phones": phones, "thumbs": thumbs,
            "clips": [{"start": c["cut"][0], "end": c["cut"][1], "score": c["hook_score"],
                       "title": c["title"], "reason": c["reason"], "idx": c["index"],
                       "reframe": c["reframe"]} for c in m["clips"]]}
    html = (ROOT / "scripts" / "hero.html").read_text().replace("__DATA__", json.dumps(data))
    html = html.replace("__FONTDIR__", FONT.as_uri())
    (WORK / "hero.html").write_text(html)
    shoot(WORK / "hero.html", ASSETS / "hero.png", 1600, 900)
    flow = (ROOT / "scripts" / "flow.html").read_text().replace("__FONTDIR__", FONT.as_uri())
    (WORK / "flow.html").write_text(flow)
    shoot(WORK / "flow.html", ASSETS / "flow.png", 1600, 720)
    print("wrote", ASSETS / "hero.png", ASSETS / "flow.png")


if __name__ == "__main__":
    main()
