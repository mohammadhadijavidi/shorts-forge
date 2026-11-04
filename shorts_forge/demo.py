"""Self-contained demo material: a synthetic "talking head" video plus a SAMPLE transcript.

Nothing here is real-world data:

* the video is generated with ffmpeg from original artwork drawn for this project
  (``scripts/presenter.svg`` and ``scripts/studio.svg``, exported by
  ``scripts/make_presenter.py``): an illustrated presenter, not a photo of anyone, who
  walks across a podcast-studio backdrop and "talks" (the mouth pose flips while the
  sample script is spoken), so the face tracker has a moving subject to follow. The
  bundled YuNet detector finds the illustrated face; the audio is a quiet tone.
* the transcript is a hand-written script with evenly spaced, synthetic word timings.
  It stands in for Whisper output so the demo runs without model weights.
"""

from __future__ import annotations

from pathlib import Path

from .render import require, run
from .segments import Segment, Word

DEMO_DIR = Path(__file__).parent / "demo"
PRESENTER_REST = DEMO_DIR / "presenter_rest.png"  # 600x800 RGBA, closed mouth
PRESENTER_TALK = DEMO_DIR / "presenter_talk.png"  # same drawing, open mouth
STUDIO = DEMO_DIR / "studio.jpg"  # 1280x720 backdrop

# Hand-written sample script (NOT a real recording). Some sentences have strong hooks,
# some are deliberately flat, so the highlight picker has something to choose between.
SCRIPT = [
    "Okay so welcome back to the channel, today is a slightly different episode.",
    "I want to talk about the way most people edit their videos.",
    "We will get into tools a little later in the episode.",
    "Why do 90 percent of your shorts die in the first three seconds?",
    "Because you start with the context instead of the payoff.",
    "Nobody scrolling on their phone cares about your intro.",
    "Put the most surprising sentence first and explain it after.",
    "That one change doubled the average watch time on my channel.",
    "Um, anyway, let me check the chat for a second.",
    "Right, a few people are asking about the microphone, it is a cheap one.",
    "Here is the biggest mistake I see from new creators.",
    "They film one long video and never cut it into clips.",
    "One podcast episode can become ten shorts if you find the right moments.",
    "You do not need an editor, you need a system that finds the hooks for you.",
    "The secret is simple: transcribe, score every moment, and crop to vertical.",
    "Okay, that is it for the main part, thanks for staying until the end.",
    "Let me know in the comments what you want to see next week.",
]


def demo_segments(start: float = 1.0, wps: float = 2.7, pause: float = 0.45) -> list[Segment]:
    """Turn :data:`SCRIPT` into segments with synthetic word timestamps."""
    t = start
    segs = []
    for sentence in SCRIPT:
        toks = sentence.split()
        words = []
        for tok in toks:
            dur = max(0.22, min(0.75, len(tok) * 0.055 + 0.12)) * 2.7 / wps
            words.append(Word(round(t, 3), round(t + dur * 0.92, 3), tok))
            t += dur
        segs.append(Segment(words[0].start, words[-1].end, sentence, words))
        t += pause
    return segs


def demo_duration(segments: list[Segment] | None = None) -> float:
    segs = segments or demo_segments()
    return round(segs[-1].end + 1.5, 2)


def make_demo_video(output: Path, duration: float | None = None, size: tuple[int, int] = (1280, 720),
                    fps: int = 25, preset: str = "veryfast") -> Path:
    """Render the synthetic 16:9 test video with ffmpeg (a few seconds of CPU time)."""
    require("ffmpeg")
    duration = duration or demo_duration()
    w, h = size
    ph = int(h * 0.96)  # presenter height on screen (head and shoulders)
    pw = int(ph * 0.75)
    pw -= pw % 2
    # The subject holds left, walks right, holds, then drifts back to center.
    lo, hi, mid = int(w * 0.04), int(w * 0.96 - pw), int((w - pw) / 2)
    d = duration
    x = (f"if(lt(t,{d * 0.25:.2f}),{lo},"
         f"if(lt(t,{d * 0.40:.2f}),{lo}+({hi}-{lo})*(t-{d * 0.25:.2f})/{d * 0.15:.2f},"
         f"if(lt(t,{d * 0.65:.2f}),{hi},"
         f"if(lt(t,{d * 0.75:.2f}),{hi}+({mid}-{hi})*(t-{d * 0.65:.2f})/{d * 0.10:.2f},{mid}))))")
    y = f"{h - ph + int(h * 0.03)}+{max(2, h // 120)}*sin(t*2.1)"
    # lip flap: the open-mouth pose shows for part of every ~0.4 s while the script runs
    talk = f"between(t,1,{d - 1.5:.2f})*gt(sin(t*15.7)+0.35*sin(t*6.1),0.2)"
    graph = (
        f"[0:v]scale={w}:{h},setsar=1,format=rgb24[bg];"
        f"[1:v]scale={pw}:{ph},format=rgba[rest];"
        f"[2:v]scale={pw}:{ph},format=rgba[talk];"
        f"[bg][rest]overlay=x='{x}':y='{y}':eval=frame[a];"
        f"[a][talk]overlay=x='{x}':y='{y}':eval=frame:enable='{talk}',"
        f"format=yuv420p[v]"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-loop", "1", "-framerate", str(fps), "-t", f"{d:.2f}", "-i", str(STUDIO),
         "-loop", "1", "-framerate", str(fps), "-t", f"{d:.2f}", "-i", str(PRESENTER_REST),
         "-loop", "1", "-framerate", str(fps), "-t", f"{d:.2f}", "-i", str(PRESENTER_TALK),
         "-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=44100:d={d:.2f}",
         "-filter_complex", graph, "-map", "[v]", "-map", "3:a",
         "-af", "volume=0.08", "-c:v", "libx264", "-preset", preset, "-crf", "23",
         "-c:a", "aac", "-b:a", "96k", "-shortest", "-movflags", "+faststart", str(output)])
    return output
