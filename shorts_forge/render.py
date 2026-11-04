"""ffmpeg plumbing: probe, cut + reframe + caption in one encode, thumbnails."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .captions import FONTS_DIR
from .reframe import CropPlan


class FFmpegError(RuntimeError):
    pass


def require(tool: str = "ffmpeg") -> str:
    path = shutil.which(tool)
    if path is None:
        raise FFmpegError(f"{tool} not found. Install ffmpeg: https://ffmpeg.org/download.html")
    return path


def run(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise FFmpegError(f"{cmd[0]} failed ({res.returncode}): {res.stderr.strip()[-800:]}")


@dataclass
class MediaInfo:
    width: int
    height: int
    duration: float
    fps: float
    has_audio: bool


def probe(video: Path) -> MediaInfo:
    require("ffprobe")
    res = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,width,height,avg_frame_rate:format=duration",
                          "-of", "json", str(video)], capture_output=True, text=True)
    if res.returncode != 0:
        raise FFmpegError(f"ffprobe cannot read {video}: {res.stderr.strip()[:300]}")
    data = json.loads(res.stdout or "{}")
    streams = data.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise FFmpegError(f"{video} has no video stream")
    num, _, den = str(v.get("avg_frame_rate") or "25/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 25.0
    return MediaInfo(int(v["width"]), int(v["height"]),
                     float((data.get("format") or {}).get("duration") or 0.0), fps or 25.0,
                     any(s.get("codec_type") == "audio" for s in streams))


def escape_filter_path(path: Path | str) -> str:
    """Escape a path for use as a filter option value inside a filtergraph."""
    s = str(path).replace("\\", "/")
    # level 1: the filter's option parser
    for ch in ("\\", "'", ":"):
        s = s.replace(ch, "\\" + ch)
    # level 2: the filtergraph parser
    for ch in ("\\", "'", "[", "]", ",", ";"):
        s = s.replace(ch, "\\" + ch)
    return s


def video_filter(plan: CropPlan, out_w: int, out_h: int, layout: str, ass_file: Path | None,
                 fonts_dir: Path = FONTS_DIR) -> str:
    """Filtergraph for one short. ``layout``: crop (face/center plan) or fit (blurred fill)."""
    if layout == "fit":
        chain = (f"[0:v]split=2[bg][fg];"
                 f"[bg]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
                 f"crop={out_w}:{out_h},boxblur=20:2,eq=brightness=-0.08[bgb];"
                 f"[fg]scale={out_w}:-2[fgs];"
                 f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1")
    else:
        chain = f"[0:v]{plan.ffmpeg_filter()},scale={out_w}:{out_h}:flags=lanczos,setsar=1"
    if ass_file is not None:
        chain += f",ass={escape_filter_path(ass_file)}:fontsdir={escape_filter_path(fonts_dir)}"
    return chain + ",format=yuv420p[v]"


def render_command(video: Path, start: float, end: float, filter_script: Path, output: Path,
                   has_audio: bool = True, crf: int = 20, preset: str = "veryfast",
                   fps: float | None = None) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
           "-filter_complex_script", str(filter_script), "-map", "[v]"]
    if has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "160k", "-af", "aresample=async=1"]
    if fps:
        cmd += ["-r", f"{fps:g}"]
    cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-profile:v", "high",
            "-movflags", "+faststart", str(output)]
    return cmd


def render_clip(video: Path, start: float, end: float, plan: CropPlan, output: Path,
                ass_file: Path | None, out_size: tuple[int, int] = (1080, 1920),
                layout: str = "crop", has_audio: bool = True, crf: int = 20,
                preset: str = "veryfast") -> Path:
    require("ffmpeg")
    output.parent.mkdir(parents=True, exist_ok=True)
    script = output.with_suffix(".filter.txt")
    script.write_text(video_filter(plan, out_size[0], out_size[1], layout, ass_file),
                      encoding="utf-8")
    try:
        run(render_command(video, start, end, script, output, has_audio, crf, preset))
    finally:
        script.unlink(missing_ok=True)
    return output


def extract_frame(video: Path, t: float, output: Path, width: int | None = None) -> Path:
    require("ffmpeg")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.3f}",
           "-i", str(video), "-frames:v", "1"]
    if width:
        cmd += ["-vf", f"scale={width}:-2"]
    run(cmd + ["-update", "1", str(output)])
    return output
