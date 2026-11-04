"""Use shorts-forge as a library.

    python examples/python_api.py path/to/video.mp4

Without an argument it renders the synthetic demo video first. Transcription uses
faster-whisper when a real video is given (pip install "shorts-forge[whisper]").
"""

import sys
from pathlib import Path

from shorts_forge.demo import demo_segments, make_demo_video
from shorts_forge.llm import get_llm
from shorts_forge.pipeline import Options, run
from shorts_forge.transcribe import FasterWhisperTranscriber


class SampleTranscriber:
    """Hand-written SAMPLE transcript for the synthetic demo video (not Whisper output)."""

    name = "sample-transcript"

    def transcribe(self, media, language=None):
        return demo_segments()


def main() -> None:
    out = Path("api-shorts")
    if len(sys.argv) > 1:
        video = Path(sys.argv[1])
        transcriber = FasterWhisperTranscriber(model="small")
    else:
        video = make_demo_video(out / "demo-source.mp4")
        transcriber = SampleTranscriber()

    # None -> offline heuristic picker. Try get_llm("openai") / get_llm("ollama", model="qwen2.5:7b").
    llm = get_llm()
    opts = Options(count=2, min_dur=15, max_dur=45, style="karaoke", layout="auto")
    result = run(video, out, transcriber, llm, opts, log=print)
    for clip in result.clips:
        h = clip.highlight
        print(f"#{clip.index} hook {h.hook_score:3d}  {clip.start:6.1f}-{clip.end:6.1f}s  "
              f"{h.title}\n    why: {h.reason}\n    -> {clip.video}")


if __name__ == "__main__":
    main()
