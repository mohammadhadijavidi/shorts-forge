"""Animated, word-by-word captions as an ASS subtitle file (burned in by ffmpeg/libass).

Words are grouped into short on-screen phrases (2-4 words is the short-form norm).
Each phrase is emitted once per word, so the word being spoken is highlighted with a
colour change and a small "pop" scale animation while the rest of the phrase stays
visible. Right-to-left scripts (Persian, Arabic, Hebrew, Urdu) switch to a font with
Arabic-script coverage and libass' automatic base direction so word order stays correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .segments import Word

FONTS_DIR = Path(__file__).parent / "fonts"
RTL_RE = re.compile(r"[֐-ࣿיִ-﷿ﹰ-﻿]")


@dataclass
class CaptionStyle:
    name: str
    font: str
    rtl_font: str
    size: int  # at PlayResY 1920
    color: str  # #RRGGBB
    highlight: str
    outline: str
    outline_px: float
    shadow_px: float
    uppercase: bool
    pop: int  # % scale of the active word (100 = no pop)
    box: bool = False  # opaque box behind the text instead of an outline
    box_color: str = "#000000"
    box_alpha: int = 0x60  # 0 = opaque, 255 = invisible


STYLES = {
    # Big, bold, one active word in yellow that pops - the "Hormozi" look
    "bold": CaptionStyle("bold", "Inter Black", "Vazirmatn", 96, "#FFFFFF", "#FFD400",
                         "#000000", 7, 3, True, 118),
    # Karaoke: the whole phrase in white, spoken word in green, no scaling
    "karaoke": CaptionStyle("karaoke", "Inter Black", "Vazirmatn", 84, "#FFFFFF", "#3DF58A",
                            "#000000", 6, 2, False, 100),
    # Clean: sentence-case text on a soft box, subtle highlight
    "clean": CaptionStyle("clean", "Inter Black", "Vazirmatn", 72, "#FFFFFF", "#7CC4FF",
                          "#000000", 0, 0, False, 106, box=True),
}


@dataclass
class Phrase:
    start: float
    end: float
    words: list[Word]


def is_rtl(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(bool(RTL_RE.match(c)) for c in letters) / len(letters) > 0.3


def ass_color(hex_rgb: str, alpha: int = 0) -> str:
    """#RRGGBB -> ASS &HAABBGGRR."""
    h = hex_rgb.lstrip("#")
    if not re.fullmatch(r"[0-9A-Fa-f]{6}", h):
        raise ValueError(f"colour must be #RRGGBB, got {hex_rgb!r}")
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, cs = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def ass_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def clip_words(words: list[Word], start: float, end: float) -> list[Word]:
    """Words inside [start, end], re-timed so the clip starts at 0."""
    out = []
    for w in words:
        if w.end <= start or w.start >= end:
            continue
        out.append(Word(max(0.0, w.start - start), min(end, w.end) - start, w.text.strip(), w.prob))
    return [w for w in out if w.text]


def group_words(words: list[Word], max_words: int = 3, max_chars: int = 18,
                max_gap: float = 0.6) -> list[Phrase]:
    """Split words into short phrases; break on punctuation, pauses and length."""
    phrases: list[Phrase] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            phrases.append(Phrase(cur[0].start, cur[-1].end, list(cur)))
            cur.clear()

    for w in words:
        if cur:
            chars = sum(len(x.text) + 1 for x in cur) + len(w.text)
            if (len(cur) >= max_words or chars > max_chars or w.start - cur[-1].end > max_gap):
                flush()
        cur.append(w)
        if re.search(r"[.!?,;:؟،؛]$", w.text):
            flush()
    flush()
    # keep each phrase on screen until the next one starts (no flicker), capped
    for a, b in zip(phrases, phrases[1:]):
        a.end = max(a.end, min(b.start, a.end + 0.5))
    return phrases


def _clean(text: str, style: CaptionStyle, rtl: bool) -> str:
    t = text.strip()
    if style.uppercase and not rtl:
        t = t.upper()
    if style.name == "bold":
        t = t.rstrip(",.;:")  # trailing punctuation looks noisy in big captions
    return ass_escape(t)


def phrase_events(phrase: Phrase, style: CaptionStyle, rtl: bool) -> list[tuple[float, float, str]]:
    """One event per spoken word: the full phrase with that word highlighted."""
    hi = ass_color(style.highlight)
    base = ass_color(style.color)
    events = []
    n = len(phrase.words)
    for i, w in enumerate(phrase.words):
        start = phrase.start if i == 0 else w.start
        end = phrase.words[i + 1].start if i + 1 < n else phrase.end
        if end - start < 0.02:
            continue
        parts = []
        for j, other in enumerate(phrase.words):
            txt = _clean(other.text, style, rtl)
            if j == i:
                pop = ""
                if style.pop != 100:
                    pop = (f"\\fscx{style.pop}\\fscy{style.pop}"
                           f"\\t(0,120,\\fscx{(style.pop + 100) // 2 + 2}\\fscy{(style.pop + 100) // 2 + 2})")
                parts.append(f"{{\\1c{hi}{pop}}}{txt}{{\\1c{base}\\fscx100\\fscy100}}")
            else:
                parts.append(txt)
        events.append((start, end, " ".join(parts)))
    return events


def build_ass(words: list[Word], style: str | CaptionStyle = "bold", width: int = 1080,
              height: int = 1920, max_words: int = 3, position: float = 0.68,
              title: str | None = None, title_seconds: float = 3.0) -> str:
    """Return a complete ASS document for a vertical clip (times relative to the clip)."""
    st = STYLES[style] if isinstance(style, str) else style
    text_all = " ".join(w.text for w in words) + " " + (title or "")
    rtl = is_rtl(text_all)
    font = st.rtl_font if rtl else st.font
    size = int(st.size * height / 1920 * (0.95 if rtl else 1.0))
    max_chars = 22 if rtl else (16 if st.uppercase else 20)
    margin_v = int(height * (1 - position))
    encoding = -1 if rtl else 1  # -1: let libass/fribidi pick the base direction
    border_style = 3 if st.box else 1
    back = ass_color(st.box_color, st.box_alpha) if st.box else ass_color("#000000", 0x80)
    outline = (14 if st.box else st.outline_px) * height / 1920
    outline_col = back if st.box else ass_color(st.outline)
    title_size = int(78 * height / 1920 * (0.9 if rtl else 1.0))
    lines = [
        "[Script Info]",
        "; generated by shorts-forge",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,{font},{size},{ass_color(st.color)},{ass_color(st.highlight)},"
        f"{outline_col},{back},-1,0,0,0,100,100,0,0,{border_style},{outline:.1f},"
        f"{st.shadow_px * height / 1920:.1f},2,{int(width * 0.07)},{int(width * 0.07)},{margin_v},{encoding}",
        f"Style: Title,{font},{title_size},{ass_color('#FFFFFF')},{ass_color('#FFFFFF')},"
        f"{ass_color('#101010', 0x30)},{ass_color('#101010', 0x30)},-1,0,0,0,100,100,0,0,3,"
        f"{18 * height / 1920:.1f},0,8,{int(width * 0.08)},{int(width * 0.08)},{int(height * 0.045)},{encoding}",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if title:
        fade = "{\\fad(150,250)}"
        lines.append(f"Dialogue: 1,{ass_time(0)},{ass_time(title_seconds)},Title,,0,0,0,,"
                     f"{fade}{ass_escape(title)}")
    for phrase in group_words(words, max_words=max_words, max_chars=max_chars):
        for start, end, text in phrase_events(phrase, st, rtl):
            lines.append(f"Dialogue: 0,{ass_time(start)},{ass_time(end)},Caption,,0,0,0,,{text}")
    return "\n".join(lines) + "\n"
