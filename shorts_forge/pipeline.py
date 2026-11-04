"""Long video in, ranked vertical shorts out.

transcribe -> sentences -> highlight picker (heuristic or LLM) -> per clip:
face-track crop plan -> ASS captions -> one ffmpeg encode -> manifest.json
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import __version__
from .captions import STYLES, build_ass, clip_words
from .highlights import Highlight, find_highlights
from .reframe import CropPlan, describe, make_detector, plan_crop
from .render import extract_frame, probe, render_clip
from .segments import Segment
from .transcribe import segments_to_json


@dataclass
class Options:
    count: int = 3
    min_dur: float = 20.0
    max_dur: float = 60.0
    layout: str = "auto"  # auto (face, else center) | face | center | fit
    detector: str = "auto"  # auto | yunet | haar | none
    style: str = "bold"
    captions: bool = True
    title_card: bool = True
    width: int = 1080
    height: int = 1920
    crf: int = 20
    preset: str = "veryfast"
    words_per_caption: int = 3
    pad_start: float = 0.15
    pad_end: float = 0.35
    thumbnails: bool = True
    dry_run: bool = False


@dataclass
class ClipResult:
    index: int
    highlight: Highlight
    start: float
    end: float
    plan: CropPlan | None
    video: Path | None
    captions: Path | None
    thumbnail: Path | None
    seconds: float = 0.0

    def to_dict(self, root: Path) -> dict:
        def rel(p: Path | None) -> str | None:
            if p is None:
                return None
            try:
                return str(p.relative_to(root))
            except ValueError:
                return str(p)
        return {"index": self.index, **self.highlight.to_dict(),
                "cut": [round(self.start, 3), round(self.end, 3)],
                "reframe": self.plan.to_dict() if self.plan else None,
                "video": rel(self.video), "captions": rel(self.captions),
                "thumbnail": rel(self.thumbnail), "render_seconds": round(self.seconds, 2)}


@dataclass
class RunResult:
    source: Path
    out_dir: Path
    picker: str
    clips: list[ClipResult] = field(default_factory=list)
    manifest: Path | None = None


def slugify(text: str, limit: int = 40) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = re.sub(r"[^\w\s-]", "", t, flags=re.U).strip().lower()
    t = re.sub(r"[\s_-]+", "-", t)
    return t[:limit].strip("-") or "clip"


def run(video: Path, out_dir: Path, transcriber, llm=None, opts: Options | None = None,
        language: str | None = None, log: Callable[[str], None] = lambda _m: None) -> RunResult:
    opts = opts or Options()
    if opts.style not in STYLES:
        raise ValueError(f"unknown caption style {opts.style!r}; choose {', '.join(STYLES)}")
    video = Path(video)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(video)
    log(f"source: {video.name} {info.width}x{info.height} {info.duration:.1f}s")

    t0 = time.time()
    segments: list[Segment] = transcriber.transcribe(video, language)
    (out_dir / "transcript.json").write_text(segments_to_json(segments), encoding="utf-8")
    words = [w for s in segments for w in s.words]
    log(f"transcript: {len(segments)} segments, {len(words)} words via "
        f"{getattr(transcriber, 'name', '?')} ({time.time() - t0:.1f}s)")

    picker = "heuristic" if llm is None else f"{llm.name}:{llm.model}"
    highlights, sentences = find_highlights(segments, opts.count, opts.min_dur, opts.max_dur, llm)
    log(f"highlights: {len(highlights)} picked from {len(sentences)} sentences ({picker})")
    result = RunResult(video, out_dir, picker)

    detector = None
    if opts.layout in ("auto", "face"):
        detector = make_detector(opts.detector)
        if detector is None and opts.layout == "face":
            raise RuntimeError("--layout face needs OpenCV (pip install 'shorts-forge[face]')")

    ranked = sorted(highlights, key=lambda h: -h.hook_score)
    for n, h in enumerate(ranked, 1):
        start = max(0.0, h.start - opts.pad_start)
        end = min(info.duration or h.end + opts.pad_end, h.end + opts.pad_end)
        stem = f"{n:02d}-{slugify(h.title)}"
        clip = ClipResult(n, h, start, end, None, None, None, None)
        c0 = time.time()
        if opts.layout == "fit":
            plan = plan_crop(video, start, end, (info.width, info.height), "center", None)
        else:
            plan = plan_crop(video, start, end, (info.width, info.height),
                             "center" if opts.layout == "center" else "face", detector)
        clip.plan = plan
        if opts.captions:
            ass = build_ass(clip_words(words, start, end), opts.style, opts.width, opts.height,
                            opts.words_per_caption,
                            title=h.title if opts.title_card else None)
            clip.captions = out_dir / f"{stem}.ass"
            clip.captions.write_text(ass, encoding="utf-8")
        if not opts.dry_run:
            clip.video = render_clip(video, start, end, plan, out_dir / f"{stem}.mp4",
                                     clip.captions, (opts.width, opts.height),
                                     "fit" if opts.layout == "fit" else "crop",
                                     info.has_audio, opts.crf, opts.preset)
            if opts.thumbnails:
                clip.thumbnail = extract_frame(clip.video, min(1.2, (end - start) / 2),
                                               out_dir / f"{stem}.jpg")
        clip.seconds = time.time() - c0
        log(f"  #{n} [{h.hook_score:3d}] {start:7.2f}-{end:7.2f}s  {h.title}  | "
            f"{'fit (blurred fill)' if opts.layout == 'fit' else describe(plan)}")
        result.clips.append(clip)

    manifest = {
        "tool": f"shorts-forge {__version__}",
        "source": str(video), "duration": round(info.duration, 3),
        "source_size": [info.width, info.height], "output_size": [opts.width, opts.height],
        "transcriber": getattr(transcriber, "name", "?"), "picker": picker,
        "options": {k: v for k, v in vars(opts).items()},
        "clips": [c.to_dict(out_dir) for c in result.clips],
    }
    result.manifest = out_dir / "manifest.json"
    result.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
