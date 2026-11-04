import re

import pytest

from shorts_forge.captions import (ass_color, ass_time, build_ass, clip_words, group_words,
                                   is_rtl)
from shorts_forge.reframe import (CropPlan, Face, choose_face, crop_size, fill_gaps,
                                  piecewise_expr, plan_from_track, simplify, smooth_path)
from shorts_forge.render import escape_filter_path
from shorts_forge.segments import Word


# ------------------------------------------------------------------ reframe geometry

def test_crop_size_for_common_sources():
    assert crop_size(1920, 1080) == (608, 1080, "x")
    assert crop_size(1280, 720) == (404, 720, "x")
    w, h, axis = crop_size(1080, 1920)
    assert (w, h, axis) == (1080, 1920, "y")
    w, h, axis = crop_size(1080, 1080)
    assert axis == "x" and h == 1080 and w % 2 == 0 and abs(w / h - 9 / 16) < 0.01


def eval_expr(expr: str, t: float) -> float:
    """Evaluate our ffmpeg if/lt expression in Python (same semantics)."""
    py = expr.replace("if(", "_if(").replace("lt(", "_lt(")
    return eval(py, {"_if": lambda c, a, b: a if c else b, "_lt": lambda a, b: a < b, "t": t})


def test_piecewise_expr_matches_plan_interpolation():
    plan = CropPlan(1280, 720, 404, 720, [(0.0, 100), (2.0, 100), (4.0, 500), (6.0, 300)])
    expr = piecewise_expr(plan.keyframes)
    for t in [0, 1, 2, 2.5, 3, 4, 5, 6, 9]:
        assert eval_expr(expr, t) == pytest.approx(plan.offset_at(t), abs=0.01)
    assert piecewise_expr([(0, 42)]) == "42"
    assert plan.ffmpeg_filter().startswith("crop=w=404:h=720:x='")


def test_fill_gaps_bridges_and_holds():
    assert fill_gaps([None, 10.0, None, None, 40.0, None]) == [10, 10, 20, 30, 40, 40]
    assert fill_gaps([None, None]) == [None, None]


def test_smooth_path_removes_jitter_and_holds_in_deadzone():
    noisy = [500 + (15 if i % 2 else -15) for i in range(40)]
    out = smooth_path(noisy, window=5, alpha=0.35, deadzone=60)
    assert max(out) - min(out) == 0  # camera stays perfectly still
    step = [100.0] * 20 + [600.0] * 20
    out = smooth_path(step, deadzone=60)
    assert out[0] == pytest.approx(100) and out[-1] > 550  # it still follows real moves


def test_simplify_keeps_corners():
    pts = [(float(t), 0.0) for t in range(10)] + [(float(t), (t - 9) * 50.0) for t in range(10, 20)]
    keys = simplify(pts, tolerance=2)
    assert keys[0] == pts[0] and keys[-1] == pts[-1] and len(keys) <= 4


def test_choose_face_prefers_big_and_continuity():
    big = Face(100, 100, 200, 200, 0.9)
    small_near = Face(900, 100, 150, 150, 0.9)
    assert choose_face([big, small_near], None, 1280) is big
    assert choose_face([big, small_near], prev_cx=900, frame_w=1280) is small_near
    assert choose_face([], None, 1280) is None


def test_plan_from_track_follows_subject_and_clamps():
    track = [(i / 4, 200.0 if i < 20 else 1100.0) for i in range(40)]
    plan = plan_from_track(track, 1280, 720, "test")
    assert plan.mode == "face" and plan.crop_w == 404
    assert plan.offset_at(0) == 0  # face at x=200 -> clamped to left edge
    assert plan.offset_at(9.75) == 1280 - 404  # face at x=1100 -> clamped to right edge


def test_plan_without_faces_is_center():
    plan = plan_from_track([(0.0, None), (0.25, None)], 1920, 1080)
    assert plan.mode == "center" and plan.keyframes == [(0.0, (1920 - 608) // 2)]


# ------------------------------------------------------------------ captions

def test_ass_helpers():
    assert ass_color("#FFD400") == "&H0000D4FF"
    assert ass_color("#102030", 0x80) == "&H80302010"
    assert ass_time(3725.456) == "1:02:05.46"
    with pytest.raises(ValueError):
        ass_color("yellow")


def test_clip_words_retimes_and_drops_outside():
    words = [Word(1, 2, "a"), Word(10, 11, "b"), Word(11, 12, "c"), Word(30, 31, "d")]
    out = clip_words(words, 9.5, 20)
    assert [(w.start, w.text) for w in out] == [(0.5, "b"), (1.5, "c")]


def test_group_words_breaks_on_count_punctuation_and_pause():
    ws = [Word(0, .2, "one"), Word(.3, .5, "two"), Word(.6, .8, "three"), Word(.9, 1.1, "four."),
          Word(1.2, 1.4, "five"), Word(3.0, 3.2, "six")]
    groups = [[w.text for w in p.words] for p in group_words(ws, max_words=3)]
    assert groups == [["one", "two", "three"], ["four."], ["five"], ["six"]]


def test_build_ass_highlights_each_word_once():
    ws = [Word(0.0, 0.4, "Stop"), Word(0.5, 0.9, "editing"), Word(1.0, 1.4, "like"),
          Word(1.5, 1.9, "this.")]
    ass = build_ass(ws, "bold", 1080, 1920, max_words=3, title="Hook")
    events = [l for l in ass.splitlines() if l.startswith("Dialogue: 0")]
    assert len(events) == 4
    assert "{\\1c&H0000D4FF" in events[0] and "STOP" in events[0]
    assert "Inter Black" in ass and "PlayResY: 1920" in ass
    assert re.search(r"Dialogue: 1,0:00:00.00,0:00:03.00,Title,,0,0,0,,\{\\fad", ass)
    assert "STOP" not in events[3]  # new phrase after 3 words


def test_build_ass_rtl_uses_arabic_font_and_auto_direction():
    ws = [Word(0, .5, "چرا"), Word(.6, 1.0, "ویدیوهای"), Word(1.1, 1.5, "شما")]
    ass = build_ass(ws, "karaoke")
    assert is_rtl("چرا ویدیو") and not is_rtl("Why video")
    style = next(l for l in ass.splitlines() if l.startswith("Style: Caption"))
    assert "Vazirmatn" in style and style.endswith(",-1")


def test_escape_filter_path_handles_special_chars():
    s = escape_filter_path("/tmp/my clips/it's: [1], ok;.ass")
    assert "\\\\\\'" in s and "\\\\:" in s and "\\[" in s and "\\," in s and "\\;" in s
