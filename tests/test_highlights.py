import json

import pytest

from shorts_forge.highlights import (Sentence, build_prompt, candidates, find_highlights,
                                     hook_features, parse_llm_json, pick_heuristic, pick_llm,
                                     select, split_sentences, validate)
from shorts_forge.llm import MockLLM, get_llm, LLMError
from shorts_forge.segments import Segment, Word

from conftest import make_segments


def sents(spec):
    """[(start, end, text)] -> sentences."""
    return [Sentence(i, a, b, t) for i, (a, b, t) in enumerate(spec)]


def test_split_sentences_uses_punctuation_and_pauses():
    words = [Word(0, .3, "Hello"), Word(.4, .7, "there."), Word(.8, 1.0, "Second"),
             Word(1.1, 1.3, "one"), Word(4.0, 4.3, "after"), Word(4.4, 4.6, "pause?")]
    out = split_sentences([Segment(0, 4.6, "x", words)])
    assert [s.text for s in out] == ["Hello there.", "Second one", "after pause?"]
    assert out[1].start == 0.8 and out[1].end == 1.3
    assert [s.id for s in out] == [0, 1, 2]


def test_split_sentences_without_word_timings_spreads_words():
    out = split_sentences([Segment(10, 14, "one two three four.")])
    assert len(out) == 1 and out[0].start == 10 and out[0].end == pytest.approx(14)


def test_split_sentences_persian_question_mark():
    segs = make_segments(["چرا ویدیوهای کوتاه شما دیده نمی‌شوند؟ جواب ساده است."])
    out = split_sentences(segs)
    assert [s.text.endswith(("؟", ".")) for s in out] == [True, True]


def test_hook_features_rewards_questions_numbers_and_penalises_filler():
    strong, why = hook_features("Why do 90 percent of your shorts fail?", "x " * 50, 20)
    weak, why_weak = hook_features("Um so anyway let me check the chat for a second and",
                                   "x " * 20, 20)
    assert strong > weak + 30
    assert "opens with a question" in why and "concrete number in the hook" in why
    assert "starts with filler" in why_weak
    assert 0 <= weak <= 100 and 0 <= strong <= 100


def test_candidates_respect_duration_limits_and_whole_sentences():
    s = sents([(0, 8, "a."), (8, 16, "b."), (16, 30, "c."), (30, 70, "d.")])
    cands = candidates(s, 10, 30)
    assert cands and all(10 <= c.duration <= 30 for c in cands)
    assert {(c.start_id, c.end_id) for c in cands} == {(0, 1), (0, 2), (1, 2), (2, 2)}


def test_select_never_overlaps():
    s = sents([(i * 5, i * 5 + 5, f"Why {i}?") for i in range(20)])
    picks = select(candidates(s, 10, 20), 5)
    picks.sort(key=lambda h: h.start)
    assert all(a.end <= b.start for a, b in zip(picks, picks[1:]))


def test_heuristic_prefers_the_hooky_part():
    segs = make_segments([
        "So welcome back everyone and thanks for coming.", "Today is a normal day here.",
        "We will talk a bit about some general things later.", "Okay let us start now slowly.",
        "Why do 90 percent of your videos fail in three seconds?",
        "Nobody cares about your intro, put the payoff first.",
        "That single change doubled my watch time.",
        "Um anyway let me check the chat for a second.", "Right, the microphone is cheap.",
    ])
    picks, _ = find_highlights(segs, count=1, min_dur=6, max_dur=15)
    assert picks[0].text.startswith("Why do 90 percent")


def test_short_transcript_falls_back_to_whole_video():
    segs = make_segments(["Short and sweet.", "Only two lines."])
    picks, _ = find_highlights(segs, count=3, min_dur=20, max_dur=60)
    assert len(picks) == 1 and picks[0].reason.startswith("whole video")


def test_parse_llm_json_tolerates_fences_and_chatter():
    raw = 'Sure! Here you go:\n```json\n{"highlights": [{"start_id": 1, "end_id": 2}]}\n```'
    assert parse_llm_json(raw) == [{"start_id": 1, "end_id": 2}]
    assert parse_llm_json('[{"start_id": 0, "end_id": 0}]') == [{"start_id": 0, "end_id": 0}]
    with pytest.raises(ValueError):
        parse_llm_json("I cannot help with that.")


def test_validate_repairs_bad_model_output():
    s = sents([(i * 6, i * 6 + 6, f"Sentence {i}.") for i in range(12)])
    raw = [
        {"start_id": 0, "end_id": 0, "title": "too short, grown", "hook_score": "88"},
        {"start_id": 5, "end_id": 11, "title": "too long, trimmed", "hook_score": 70},
        {"start_id": 1, "end_id": 2, "hook_score": 99},  # overlaps the first one
        {"start_id": 99, "end_id": 100},  # out of range
        {"title": "no ids"},
        {"start_id": 9, "end_id": 3, "hook_score": 400},  # reversed + score clamp
    ]
    out = validate(raw, s, count=5, min_dur=18, max_dur=30)
    spans = sorted((h.start_id, h.end_id) for h in out)
    # (9,3) -> reversed ids fixed and grown to (9,11), score clamped to 100;
    # (1,2) grown to (1,3); (0,0)->(0,2) and (5,11)->(5,9) lose on overlap.
    assert spans == [(1, 3), (9, 11)]
    assert max(h.hook_score for h in out) == 100
    for h in out:
        assert 18 <= h.duration <= 30 and 0 <= h.hook_score <= 100
    ordered = sorted(out, key=lambda h: h.start)
    assert all(a.end <= b.start for a, b in zip(ordered, ordered[1:]))


class RecordingLLM:
    name, model = "fake", "fake-1"

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def complete(self, system, user):
        self.prompts.append((system, user))
        return self.answer


def test_pick_llm_sends_numbered_sentences_and_uses_answer():
    s = sents([(i * 5, i * 5 + 5, f"Line {i}.") for i in range(10)])
    llm = RecordingLLM(json.dumps({"highlights": [
        {"start_id": 6, "end_id": 8, "title": "The twist", "hook_score": 91, "reason": "payoff"}]}))
    out = pick_llm(llm, s, count=2, min_dur=10, max_dur=20)
    assert [(h.start_id, h.end_id, h.title, h.hook_score) for h in out] == [(6, 8, "The twist", 91)]
    system, user = llm.prompts[0]
    assert "JSON" in system and '"id": 6' in user and "between 10 and 20 seconds" in user


def test_mock_llm_round_trips_through_the_llm_path():
    segs = make_segments([f"Why does idea number {i} matter to you?" for i in range(30)])
    picks_mock, sentences = find_highlights(segs, 2, 8, 16, llm=MockLLM())
    picks_heur = pick_heuristic(sentences, 2, 8, 16)
    assert [(h.start_id, h.end_id) for h in picks_mock] == [(h.start_id, h.end_id) for h in picks_heur]
    assert "<sentences>" in build_prompt(sentences, 2, 8, 16)


def test_get_llm_providers(monkeypatch):
    monkeypatch.delenv("SHORTS_FORGE_LLM_PROVIDER", raising=False)
    assert get_llm() is None  # heuristic by default
    assert get_llm("mock").name == "mock"
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LLMError):
        get_llm("openai")
    local = get_llm("ollama", model="llama3.2")
    assert local.base_url.startswith("http://localhost:11434") and local.model == "llama3.2"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert get_llm("anthropic").name == "anthropic"
    with pytest.raises(LLMError):
        get_llm("nope")
