"""Plain data types shared by every stage of the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Word:
    start: float
    end: float
    text: str
    prob: float | None = None


@dataclass
class Segment:
    """One chunk of speech as returned by a speech-to-text backend."""

    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
