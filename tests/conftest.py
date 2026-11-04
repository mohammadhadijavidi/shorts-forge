import shutil
from pathlib import Path

import pytest

from shorts_forge.demo import make_demo_video
from shorts_forge.segments import Segment, Word

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                  reason="ffmpeg/ffprobe not installed")


def make_segments(sentences, start=0.5, wps=3.0, pause=0.4):
    """Synthetic transcript: evenly timed words (stands in for Whisper output)."""
    t = start
    out = []
    for text in sentences:
        words = []
        for tok in text.split():
            words.append(Word(round(t, 3), round(t + 0.9 / wps, 3), tok))
            t += 1 / wps
        out.append(Segment(words[0].start, words[-1].end, text, words))
        t += pause
    return out


class FakeTranscriber:
    """Mocked speech-to-text: Whisper weights are not needed to test the pipeline."""

    name = "fake"

    def __init__(self, segments):
        self.segments = segments
        self.calls = 0

    def transcribe(self, media, language=None):
        self.calls += 1
        return self.segments


@pytest.fixture(scope="session")
def demo_video(tmp_path_factory) -> Path:
    """A real 640x360, 24 s H.264+AAC test video rendered by ffmpeg.

    The subject is the bundled illustrated presenter (original vector artwork, not a
    photo of a person) walking across a synthetic studio backdrop.
    """
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("media") / "talk.mp4"
    return make_demo_video(out, duration=24.0, size=(640, 360), fps=15)
