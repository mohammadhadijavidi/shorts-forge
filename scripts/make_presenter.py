"""Export the synthetic demo artwork from SVG to the PNGs shipped in the package.

    python scripts/make_presenter.py

* ``scripts/presenter.svg`` -> ``shorts_forge/demo/presenter_rest.png`` and
  ``presenter_talk.png`` (transparent, 600x800; the two mouth poses).
* ``scripts/studio.svg`` -> ``shorts_forge/demo/studio.jpg`` (1280x720 backdrop).

The presenter is an original illustration drawn for this project; it is not a photo
and does not depict any real person. Only needed when you change the SVGs: the PNGs
are committed so ``shorts-forge demo`` needs nothing but ffmpeg. Requires Playwright.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "shorts_forge" / "demo"


def render(svg: str, png: Path, w: int, h: int, transparent: bool) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=1)
        page.set_content(f"<html><body style='margin:0;background:transparent'>{svg}</body></html>")
        page.screenshot(path=str(png), omit_background=transparent,
                        clip={"x": 0, "y": 0, "width": w, "height": h})
        b.close()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src = (ROOT / "scripts" / "presenter.svg").read_text()
    for pose, hide in (("rest", "talk"), ("talk", "rest")):
        svg = src.replace(f'<g class="{hide}">', f'<g class="{hide}" display="none">')
        render(svg, OUT / f"presenter_{pose}.png", 600, 800, True)
    tmp = OUT / "_studio.png"
    render((ROOT / "scripts" / "studio.svg").read_text(), tmp, 1280, 720, False)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp), "-q:v", "3",
                    str(OUT / "studio.jpg")], check=True)
    tmp.unlink()
    print("wrote", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
