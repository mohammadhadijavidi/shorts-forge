"""Command line interface.

    shorts-forge VIDEO [VIDEO ...] [-o OUT] [--count 3] [--min 20] [--max 60] ...
    shorts-forge demo [-o OUT]        # synthetic video + sample transcript, fully offline
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .captions import STYLES
from .llm import PROVIDERS, LLMError, get_llm
from .pipeline import Options, run


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader (KEY=VALUE lines); real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def make_transcriber(args, video: Path):
    from .transcribe import FasterWhisperTranscriber, JsonTranscriber, OpenAITranscriber

    if args.transcript:
        p = Path(args.transcript)
        if p.is_dir():
            p = p / f"{video.stem}.json"
        return JsonTranscriber(p)
    if args.stt == "openai":
        return OpenAITranscriber(model=args.stt_model)
    return FasterWhisperTranscriber(model=args.whisper_model or "small", device=args.device)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="shorts-forge",
        description="Turn long videos into viral vertical shorts: transcribe, pick hooks, "
                    "face-track to 9:16, burn animated captions.")
    p.add_argument("videos", nargs="+", metavar="VIDEO",
                   help="input video file(s), or the word 'demo' for the offline demo")
    p.add_argument("-o", "--out", default="shorts", help="output folder (default: ./shorts)")
    p.add_argument("-n", "--count", type=int, default=3, help="shorts per video (default 3)")
    p.add_argument("--min", dest="min_dur", type=float, default=20, help="min clip seconds")
    p.add_argument("--max", dest="max_dur", type=float, default=60, help="max clip seconds")
    g = p.add_argument_group("transcription")
    g.add_argument("--stt", choices=["faster-whisper", "openai"], default="faster-whisper",
                   help="speech-to-text backend (default: local faster-whisper)")
    g.add_argument("--whisper-model", help="faster-whisper model (default: small)")
    g.add_argument("--device", default="auto", help="faster-whisper device: auto|cpu|cuda")
    g.add_argument("--stt-model", help="model for --stt openai (default whisper-1)")
    g.add_argument("--language", help="spoken language code (default: auto-detect)")
    g.add_argument("--transcript", help="re-use a transcript JSON (file, or a folder with "
                                        "<video-stem>.json files for batch runs)")
    g = p.add_argument_group("highlight picker")
    g.add_argument("--llm", choices=PROVIDERS, default=None,
                   help="LLM provider (default: $SHORTS_FORGE_LLM_PROVIDER or 'heuristic')")
    g.add_argument("--model", help="LLM model name")
    g.add_argument("--base-url", help="base URL for an OpenAI-compatible server")
    g = p.add_argument_group("video")
    g.add_argument("--layout", choices=["auto", "face", "center", "fit"], default="auto",
                   help="auto: follow faces, else center | fit: whole frame on blurred fill")
    g.add_argument("--detector", choices=["auto", "yunet", "haar", "none"], default="auto")
    g.add_argument("--style", choices=list(STYLES), default="bold", help="caption style")
    g.add_argument("--no-captions", action="store_true", help="do not burn captions")
    g.add_argument("--no-title", action="store_true", help="no title card in the first seconds")
    g.add_argument("--words", type=int, default=3, help="max words on screen (default 3)")
    g.add_argument("--size", default="1080x1920", help="output size WxH (default 1080x1920)")
    g.add_argument("--crf", type=int, default=20, help="x264 quality (lower = better)")
    g.add_argument("--preset", default="veryfast", help="x264 preset (default veryfast)")
    p.add_argument("--dry-run", action="store_true",
                   help="pick clips and write captions/manifest, skip rendering")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--version", action="version", version=f"shorts-forge {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    log = (lambda _m: None) if args.quiet else (lambda m: print(m, file=sys.stderr))
    try:
        w, h = (int(x) for x in args.size.lower().split("x"))
    except ValueError:
        print(f"error: --size must look like 1080x1920, got {args.size!r}", file=sys.stderr)
        return 2
    opts = Options(count=args.count, min_dur=args.min_dur, max_dur=args.max_dur,
                   layout=args.layout, detector=args.detector, style=args.style,
                   captions=not args.no_captions, title_card=not args.no_title,
                   width=w - w % 2, height=h - h % 2, crf=args.crf, preset=args.preset,
                   words_per_caption=args.words, dry_run=args.dry_run)
    if opts.min_dur > opts.max_dur:
        print("error: --min must be <= --max", file=sys.stderr)
        return 2
    try:
        llm = get_llm(args.llm, args.model, args.base_url)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out_root = Path(args.out)

    jobs: list[tuple[Path, object]] = []
    if args.videos == ["demo"]:
        from .demo import demo_segments, make_demo_video

        class SampleTranscriber:
            name = "sample-transcript (hand-written demo script, not Whisper)"

            def transcribe(self, media, language=None):
                return demo_segments()

        src = out_root / "demo-source.mp4"
        log("rendering synthetic demo video with ffmpeg ...")
        make_demo_video(src, preset=opts.preset)
        jobs.append((src, SampleTranscriber()))
    else:
        for v in args.videos:
            path = Path(v)
            if not path.is_file():
                print(f"error: {v} not found", file=sys.stderr)
                return 2
            jobs.append((path, None))

    batch = len(jobs) > 1
    failures = 0
    for path, transcriber in jobs:
        out_dir = out_root / path.stem if batch else out_root
        try:
            res = run(path, out_dir, transcriber or make_transcriber(args, path), llm, opts,
                      args.language, log)
        except Exception as exc:  # keep a batch going; report at the end
            failures += 1
            print(f"error: {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        for c in res.clips:
            target = c.video or c.captions
            print(f"{c.highlight.hook_score:3d}  {c.start:8.2f}-{c.end:8.2f}  {target}")
        log(f"manifest: {res.manifest}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
