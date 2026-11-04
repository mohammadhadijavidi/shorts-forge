"""Find the moments worth turning into shorts.

The transcript is split into sentences (using word timestamps). A highlight is always a
run of whole sentences, so a clip never starts or ends mid-word.

Two pickers share the same candidate model and validation:

* **heuristic** - a transparent, offline hook scorer (questions, numbers, "you",
  curiosity / contrarian words, a short punchy opener, filler penalty, pace).
* **llm** - any provider from :mod:`shorts_forge.llm` reads the numbered sentences and
  returns JSON (``start_id``, ``end_id``, ``title``, ``hook_score``, ``reason``). The
  answer is validated, snapped to sentence boundaries, clamped to the duration limits
  and de-overlapped before use, so a sloppy model cannot produce a broken clip.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .segments import Segment, Word

SENTENCE_END = re.compile(r"[.!?؟۔。]+[\"')\]]*$")  # . ! ? ؟ ۔ 。

HOOK_WORDS = {
    # curiosity / contrarian / stakes - English
    "secret", "secrets", "mistake", "mistakes", "never", "nobody", "stop", "why", "how",
    "truth", "biggest", "actually", "wrong", "most", "everyone", "nothing", "only",
    "worst", "best", "crazy", "surprising", "lie", "lied", "fail", "failed", "failure",
    "million", "free", "money", "hack", "simple", "real", "first", "changed", "insane",
    # Persian
    "چرا", "چطور", "راز", "اشتباه", "هیچ‌کس", "هرگز", "بزرگ‌ترین", "واقعیت", "میلیون",
    # Arabic
    "لماذا", "كيف", "سر", "خطأ", "أبدا", "الحقيقة",
}
FILLER_START = {"so", "um", "uh", "and", "like", "yeah", "okay", "ok", "well", "but",
                "anyway", "right", "خب", "یعنی", "اوم"}
SECOND_PERSON = {"you", "your", "you're", "youre", "you've", "yourself", "تو", "شما", "أنت"}


@dataclass
class Sentence:
    id: int
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)


@dataclass
class Highlight:
    start: float
    end: float
    start_id: int
    end_id: int
    title: str
    hook_score: int  # 0-100
    reason: str
    text: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"], d["end"] = round(self.start, 3), round(self.end, 3)
        d["duration"] = round(self.duration, 3)
        return d


# --------------------------------------------------------------------------- sentences

def split_sentences(segments: list[Segment], max_words: int = 40) -> list[Sentence]:
    """Regroup transcript words into sentences (punctuation, long pauses or a word cap)."""
    words: list[Word] = []
    for s in segments:
        if s.words:
            words.extend(s.words)
        elif s.text.strip():  # no word timings: spread the words evenly over the segment
            toks = s.text.split()
            step = (s.end - s.start) / max(1, len(toks))
            words.extend(Word(s.start + i * step, s.start + (i + 1) * step, t)
                         for i, t in enumerate(toks))
    out: list[Sentence] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            out.append(Sentence(len(out), cur[0].start, cur[-1].end,
                                " ".join(w.text for w in cur), list(cur)))
            cur.clear()

    for i, w in enumerate(words):
        if cur and w.start - cur[-1].end > 1.2:  # long silence = sentence break
            flush()
        cur.append(w)
        if SENTENCE_END.search(w.text) or len(cur) >= max_words:
            flush()
    flush()
    return out


# --------------------------------------------------------------------------- heuristic

def _tokens(text: str) -> list[str]:
    return [t.strip(".,!?;:\"'()[]«»،؛؟").lower() for t in text.split() if t.strip()]


def hook_features(first: str, body: str, duration: float) -> tuple[int, list[str]]:
    """Score how well a clip's opening grabs attention. Returns (0-100, reasons)."""
    toks = _tokens(first)
    all_toks = _tokens(body)
    score = 30.0
    why: list[str] = []
    if "?" in first or "؟" in first:
        score += 18
        why.append("opens with a question")
    if any(re.search(r"\d", t) for t in toks):
        score += 12
        why.append("concrete number in the hook")
    if SECOND_PERSON & set(toks):
        score += 10
        why.append("speaks to the viewer")
    hooks = sorted(HOOK_WORDS & set(toks))
    if hooks:
        score += min(20, 8 * len(hooks))
        why.append("curiosity words: " + ", ".join(hooks[:3]))
    if 3 <= len(toks) <= 14:
        score += 8
        why.append("short punchy opener")
    elif len(toks) > 24:
        score -= 8
    if toks and toks[0] in FILLER_START:
        score -= 12
        why.append("starts with filler")
    body_hooks = len(HOOK_WORDS & set(all_toks)) - len(hooks)
    score += min(8, 2 * max(0, body_hooks))
    if duration > 0:
        wps = len(all_toks) / duration
        if wps >= 2.6:
            score += 6
            why.append(f"fast pace ({wps:.1f} words/s)")
        elif wps < 1.4:
            score -= 8
            why.append("slow / lots of silence")
    return int(max(0, min(100, round(score)))), why


def _title_from(text: str, limit: int = 60) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= limit:
        return t
    cut = t[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "..."


def candidates(sentences: list[Sentence], min_dur: float, max_dur: float) -> list[Highlight]:
    """Every run of whole sentences whose length fits [min_dur, max_dur], heuristically scored."""
    out: list[Highlight] = []
    for i, first in enumerate(sentences):
        for j in range(i, len(sentences)):
            dur = sentences[j].end - first.start
            if dur > max_dur:
                break
            if dur < min_dur:
                continue
            body = " ".join(s.text for s in sentences[i:j + 1])
            score, why = hook_features(first.text, body, dur)
            # mild preference for clips that land near the middle of the allowed range
            target = (min_dur + max_dur) / 2
            score = int(max(0, min(100, score - abs(dur - target) / max(1.0, target) * 6)))
            out.append(Highlight(first.start, sentences[j].end, i, j, _title_from(first.text),
                                 score, "; ".join(why) or "steady, complete thought", body))
    return out


def select(cands: list[Highlight], count: int, min_gap: float = 1.0) -> list[Highlight]:
    """Greedy best-first pick of non-overlapping highlights."""
    chosen: list[Highlight] = []
    for c in sorted(cands, key=lambda h: (-h.hook_score, h.start)):
        if all(c.end + min_gap <= h.start or c.start >= h.end + min_gap for h in chosen):
            chosen.append(c)
        if len(chosen) >= count:
            break
    return chosen


def pick_heuristic(sentences: list[Sentence], count: int = 3, min_dur: float = 20,
                   max_dur: float = 60) -> list[Highlight]:
    return select(candidates(sentences, min_dur, max_dur), count)


# --------------------------------------------------------------------------- LLM

SYSTEM_PROMPT = """You are a senior short-form video editor (TikTok, YouTube Shorts, Instagram Reels).
You receive a transcript split into numbered sentences with timestamps in seconds.
Pick the segments most likely to go viral as standalone vertical shorts.

Rules:
- A segment is a contiguous run of sentences: start_id..end_id (inclusive).
- Each segment must be understandable without the rest of the video and end on a complete thought.
- The first sentence is the hook: it must stop the scroll (a question, a bold claim, a number, stakes).
- Respect the requested duration range. Segments must not overlap.
- hook_score is 0-100: how strongly the first 3 seconds grab attention.
- title is a short, catchy on-screen title in the transcript's language (max 60 characters).
- reason is one sentence explaining why this moment works.

Answer with JSON only, no prose:
{"highlights": [{"start_id": 0, "end_id": 3, "title": "...", "hook_score": 87, "reason": "..."}]}"""


def build_prompt(sentences: list[Sentence], count: int, min_dur: float, max_dur: float) -> str:
    rows = [{"id": s.id, "start": round(s.start, 2), "end": round(s.end, 2), "text": s.text}
            for s in sentences]
    return (f"Pick the best {count} highlights. Each must last between {min_dur:g} and "
            f"{max_dur:g} seconds.\n<constraints>{json.dumps({'count': count, 'min': min_dur, 'max': max_dur})}"
            f"</constraints>\n<sentences>\n{json.dumps(rows, ensure_ascii=False)}\n</sentences>")


def heuristic_answer(user_prompt: str) -> str:
    """What :class:`~shorts_forge.llm.MockLLM` returns: the heuristic picks, as LLM-shaped JSON."""
    m = re.search(r"<sentences>\s*(.*?)\s*</sentences>", user_prompt, re.S)
    c = re.search(r"<constraints>(.*?)</constraints>", user_prompt, re.S)
    rows = json.loads(m.group(1)) if m else []
    cons = json.loads(c.group(1)) if c else {"count": 3, "min": 20, "max": 60}
    sents = [Sentence(r["id"], float(r["start"]), float(r["end"]), r["text"]) for r in rows]
    picks = pick_heuristic(sents, int(cons["count"]), float(cons["min"]), float(cons["max"]))
    return json.dumps({"highlights": [
        {"start_id": h.start_id, "end_id": h.end_id, "title": h.title,
         "hook_score": h.hook_score, "reason": h.reason} for h in picks]}, ensure_ascii=False)


def parse_llm_json(text: str) -> list[dict]:
    """Extract the highlight list from a model answer (tolerates code fences and chatter)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    pairs = [("{", "}"), ("[", "]")]
    pairs.sort(key=lambda oc: text.find(oc[0]) if oc[0] in text else len(text))
    for opener, closer in pairs:  # whichever JSON container appears first wins
        a, b = text.find(opener), text.rfind(closer)
        if a != -1 and b > a:
            try:
                data = json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                data = data.get("highlights") or data.get("clips") or []
            if isinstance(data, list):
                return [d for d in data if isinstance(d, dict)]
    raise ValueError(f"no JSON highlight list in model answer: {text[:200]!r}")


def validate(raw: list[dict], sentences: list[Sentence], count: int, min_dur: float,
             max_dur: float) -> list[Highlight]:
    """Turn model output into safe highlights: valid ids, duration limits, no overlaps."""
    n = len(sentences)
    out: list[Highlight] = []
    for item in raw:
        try:
            i, j = int(item["start_id"]), int(item["end_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (0 <= i < n):
            continue
        j = max(i, min(j, n - 1))
        # grow to reach min_dur, then shrink to respect max_dur
        while j + 1 < n and sentences[j].end - sentences[i].start < min_dur \
                and sentences[j + 1].end - sentences[i].start <= max_dur:
            j += 1
        while j > i and sentences[j].end - sentences[i].start > max_dur:
            j -= 1
        dur = sentences[j].end - sentences[i].start
        if dur > max_dur * 1.25 or dur < min_dur * 0.5:
            continue
        try:
            score = int(round(float(item.get("hook_score", item.get("score", 50)))))
        except (TypeError, ValueError):
            score = 50
        body = " ".join(s.text for s in sentences[i:j + 1])
        title = str(item.get("title") or "").strip() or _title_from(sentences[i].text)
        out.append(Highlight(sentences[i].start, sentences[j].end, i, j, title[:80],
                             max(0, min(100, score)), str(item.get("reason") or "").strip(), body))
    return select(out, count, min_gap=0.0)


def pick_llm(llm, sentences: list[Sentence], count: int = 3, min_dur: float = 20,
             max_dur: float = 60) -> list[Highlight]:
    answer = llm.complete(SYSTEM_PROMPT, build_prompt(sentences, count, min_dur, max_dur))
    return validate(parse_llm_json(answer), sentences, count, min_dur, max_dur)


def find_highlights(segments: list[Segment], count: int = 3, min_dur: float = 20,
                    max_dur: float = 60, llm=None) -> tuple[list[Highlight], list[Sentence]]:
    sentences = split_sentences(segments)
    if not sentences:
        return [], sentences
    if llm is None:
        picks = pick_heuristic(sentences, count, min_dur, max_dur)
    else:
        picks = pick_llm(llm, sentences, count, min_dur, max_dur)
    if not picks:  # transcript shorter than min_dur: offer the whole thing once
        total = sentences[-1].end - sentences[0].start
        if total <= max_dur:
            body = " ".join(s.text for s in sentences)
            score, why = hook_features(sentences[0].text, body, total)
            picks = [Highlight(sentences[0].start, sentences[-1].end, 0, len(sentences) - 1,
                               _title_from(sentences[0].text), score,
                               "whole video (shorter than --min)", body)]
    return picks, sentences
