"""End-to-end tests: real ffmpeg cutting, face-tracked cropping and caption burning.

Transcription is mocked (FakeTranscriber) because Whisper weights are not needed to
test everything downstream of speech-to-text.
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from shorts_forge import cli
from shorts_forge.pipeline import Options, run
from shorts_forge.reframe import make_detector, plan_crop
from shorts_forge.render import probe
from shorts_forge.transcribe import segments_to_json

from conftest import FakeTranscriber, make_segments, needs_ffmpeg

pytestmark = needs_ffmpeg
cv2 = pytest.importorskip("cv2")

SCRIPT = [
    "Okay so welcome back, this is the intro part.",                 # flat
    "We will talk about some general things later on.",              # flat
    "Why do 90 percent of your shorts die in three seconds?",        # hook
    "Because you start with context instead of the payoff.",
    "Nobody scrolling cares about your intro at all.",
    "Um anyway let me check the chat for a second here.",            # flat
    "Right, a few people asked about the microphone again.",
    "Here is the biggest mistake new creators make every day.",      # hook
    "They never cut their long videos into clips.",
]


def frame_at(video: Path, t: float):
    res = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t}", "-i", str(video),
                          "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
                         capture_output=True, check=True)
    return cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)


def yellow_pixels(img) -> int:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return int(cv2.inRange(hsv, (22, 170, 200), (34, 255, 255)).sum() // 255)


@pytest.fixture(scope="module")
def rendered(demo_video, tmp_path_factory):
    out = tmp_path_factory.mktemp("out with space's")  # exercises filter-path escaping
    fake = FakeTranscriber(make_segments(SCRIPT, start=0.5, wps=3.2))
    opts = Options(count=2, min_dur=5, max_dur=9, width=360, height=640, crf=28)
    res = run(demo_video, out, fake, None, opts)
    return res, fake, out


def test_pipeline_renders_ranked_vertical_shorts(rendered):
    res, fake, out = rendered
    assert fake.calls == 1
    assert len(res.clips) == 2
    assert res.clips[0].highlight.hook_score >= res.clips[1].highlight.hook_score
    titles = [c.highlight.title for c in res.clips]
    assert any(t.startswith("Why do 90 percent") for t in titles)
    for c in res.clips:
        info = probe(c.video)
        assert (info.width, info.height) == (360, 640)
        assert info.has_audio
        assert info.duration == pytest.approx(c.end - c.start, abs=0.3)
        assert c.thumbnail.is_file() and c.captions.is_file()


def test_manifest_and_transcript_written(rendered):
    res, _, out = rendered
    m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert m["picker"] == "heuristic" and m["transcriber"] == "fake"
    assert len(m["clips"]) == 2 and m["clips"][0]["video"].endswith(".mp4")
    assert m["clips"][0]["reframe"]["detector"] in ("yunet", "haar")
    assert json.loads((out / "transcript.json").read_text())["segments"]


def test_captions_are_actually_burned(rendered, demo_video, tmp_path):
    res, _, _ = rendered
    clip = res.clips[0]
    captioned = frame_at(clip.video, 1.0)
    plain = run(demo_video, tmp_path, FakeTranscriber(make_segments(SCRIPT, 0.5, 3.2)), None,
                Options(count=1, min_dur=5, max_dur=9, width=360, height=640, crf=28,
                        captions=False, thumbnails=False))
    raw = frame_at(plain.clips[0].video, 1.0)
    band = slice(int(640 * 0.55), int(640 * 0.70))  # caption band: baseline at 68% of height
    assert yellow_pixels(captioned[band]) > 80  # the highlighted word, in the caption band
    assert yellow_pixels(raw) < 10
    diff = cv2.absdiff(captioned[band], raw[band]).mean()
    assert diff > 3


def test_demo_subject_is_synthetic_artwork_and_yunet_detects_it():
    """The demo uses a drawn presenter (no real-person photo); YuNet must still fire on it,
    otherwise the face-tracking tests below would only be exercising the center fallback."""
    from shorts_forge.demo import DEMO_DIR, PRESENTER_REST, PRESENTER_TALK

    assert sorted(p.name for p in DEMO_DIR.iterdir()) == [
        "presenter_rest.png", "presenter_talk.png", "studio.jpg"]
    det = make_detector("auto")
    if det is None or det.name != "yunet":
        pytest.skip("YuNet not available in this OpenCV build")
    for png in (PRESENTER_REST, PRESENTER_TALK):
        rgba = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        assert rgba.shape == (800, 600, 4)  # transparent cut-out from scripts/presenter.svg
        alpha = rgba[..., 3:4] / 255.0
        img = (rgba[..., :3] * alpha + np.array([40, 30, 30]) * (1 - alpha)).astype(np.uint8)
        faces = det.detect(img)
        assert faces and max(f.score for f in faces) > 0.8
        best = max(faces, key=lambda f: f.w * f.h)
        assert abs(best.cx - 300) < 40 and 280 < best.cy < 460  # the drawn head


def test_face_tracking_follows_the_moving_speaker(demo_video):
    det = make_detector("auto")
    if det is None:
        pytest.skip("no face detector in this OpenCV build")
    # in the 24 s demo video the presenter walks from the left (t<6) to the right (t>9.6)
    plan = plan_crop(demo_video, 2.0, 14.0, (640, 360), "face", det)
    assert plan.mode == "face" and plan.face_coverage > 0.8
    early, late = plan.offset_at(1.0), plan.offset_at(11.0)
    assert late - early > 150  # the crop window moved right with the speaker
    # and in the actual render the face sits near the middle of the vertical frame
    info = probe(demo_video)
    assert info.width == 640


def test_rendered_frame_keeps_face_centered(rendered):
    res, _, _ = rendered
    det = make_detector("auto")
    if det is None:
        pytest.skip("no face detector")
    for c in res.clips:
        img = frame_at(c.video, (c.end - c.start) / 2)
        faces = det.detect(img)
        assert faces, f"no face in the middle of clip {c.index}"
        best = max(faces, key=lambda f: f.w * f.h)
        assert abs(best.cx - img.shape[1] / 2) < img.shape[1] * 0.3


def test_center_and_fit_layouts(demo_video, tmp_path):
    segs = make_segments(SCRIPT, 0.5, 3.2)
    for layout in ("center", "fit"):
        res = run(demo_video, tmp_path / layout, FakeTranscriber(segs), None,
                  Options(count=1, min_dur=5, max_dur=9, width=270, height=480, crf=30,
                          layout=layout, style="clean", thumbnails=False))
        info = probe(res.clips[0].video)
        assert (info.width, info.height) == (270, 480)
        assert res.clips[0].plan.mode == "center"


def test_cli_batch_with_saved_transcripts_and_dry_run(demo_video, tmp_path, capsys):
    v2 = tmp_path / "second.mp4"
    v2.write_bytes(demo_video.read_bytes())
    tdir = tmp_path / "transcripts"
    tdir.mkdir()
    seg_json = segments_to_json(make_segments(SCRIPT, 0.5, 3.2))
    for stem in ("talk", "second"):
        (tdir / f"{stem}.json").write_text(seg_json, encoding="utf-8")
    code = cli.main([str(demo_video), str(v2), "-o", str(tmp_path / "batch"), "--transcript",
                     str(tdir), "--min", "5", "--max", "9", "-n", "1", "--dry-run", "-q",
                     "--llm", "mock"])
    assert code == 0
    for stem in ("talk", "second"):
        m = json.loads((tmp_path / "batch" / stem / "manifest.json").read_text())
        assert m["picker"] == "mock:mock-heuristic"
        assert m["clips"][0]["video"] is None and m["clips"][0]["captions"].endswith(".ass")
    assert ".ass" in capsys.readouterr().out


def test_cli_rejects_bad_input(tmp_path):
    assert cli.main([str(tmp_path / "missing.mp4")]) == 2
    assert cli.main(["x.mp4", "--size", "big"]) == 2
    assert cli.main(["x.mp4", "--min", "50", "--max", "10"]) == 2
