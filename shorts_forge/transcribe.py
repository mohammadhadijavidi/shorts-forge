"""Speech-to-text backends.

* ``faster-whisper``  local Whisper via CTranslate2 (CPU or CUDA), word timestamps.
* ``openai``          any OpenAI-compatible ``/audio/transcriptions`` endpoint
                      (OpenAI ``whisper-1``, Groq ``whisper-large-v3``, a local server...).
* ``json``            re-use a transcript saved by a previous run (no model needed).

Every backend returns segments with word-level timestamps, which the caption
renderer needs for word-by-word highlighting.
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import subprocess
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Protocol

from .segments import Segment, Word



class Transcriber(Protocol):
    name: str

    def transcribe(self, media: Path, language: str | None = None) -> list[Segment]: ...


class FasterWhisperTranscriber:
    name = "faster-whisper"

    def __init__(self, model: str = "large-v3-turbo", device: str = "auto",
                 compute_type: str = "default", beam_size: int = 5, vad: bool = True,
                 prompt: str | None = None, download_root: str | None = None):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.vad = vad
        self.prompt = prompt
        self.download_root = download_root
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise RuntimeError(
                    "faster-whisper is not installed. Install it with:\n"
                    "  pip install 'shorts-forge[whisper]'\n"
                    "or use --stt openai (API) / --transcript (saved transcript JSON)."
                ) from exc
            try:
                self._model = WhisperModel(self.model_name, device=self.device,
                                           compute_type=self.compute_type,
                                           download_root=self.download_root)
            except Exception as exc:  # network, disk, CUDA... surface one clear message
                raise RuntimeError(
                    f"could not load Whisper model {self.model_name!r}: "
                    f"{type(exc).__name__}: {str(exc)[:200]}\n"
                    "The first run downloads the weights from Hugging Face. If that host is "
                    "blocked, download a CTranslate2 model elsewhere and pass its folder with "
                    "--model /path/to/model, or use --stt openai."
                ) from exc
        return self._model

    def transcribe(self, media: Path, language: str | None = None) -> list[Segment]:
        model = self._load()
        segments, _info = model.transcribe(
            str(media), language=language, beam_size=self.beam_size, vad_filter=self.vad,
            word_timestamps=True, initial_prompt=self.prompt or None,
            condition_on_previous_text=False,  # limits repetition loops on long files
        )
        out = []
        for s in segments:
            words = [Word(w.start, w.end, w.word.strip(), getattr(w, "probability", None))
                     for w in (s.words or []) if w.word.strip()]
            out.append(Segment(s.start, s.end, s.text.strip(), words,
                               getattr(s, "no_speech_prob", None), getattr(s, "avg_logprob", None)))
        return out


def extract_audio(media: Path, dest: Path, bitrate: str = "48k") -> Path:
    """Extract mono 16 kHz audio with ffmpeg (small enough for 25 MB API limits)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Install it: https://ffmpeg.org/download.html")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(media), "-vn",
           "-ac", "1", "-ar", "16000", "-b:a", bitrate, str(dest)]
    subprocess.run(cmd, check=True)
    return dest


def _multipart(fields: dict[str, str | list[str]], file_field: str, path: Path) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for key, value in fields.items():
        for v in value if isinstance(value, list) else [value]:
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{v}\r\n'
                         .encode())
    ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
                 f'filename="{path.name}"\r\nContent-Type: {ctype}\r\n\r\n'.encode())
    parts.append(path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def parse_verbose_json(data: dict) -> list[Segment]:
    """Convert an OpenAI-style ``verbose_json`` transcription into segments."""
    words = [Word(float(w["start"]), float(w["end"]), str(w["word"]).strip())
             for w in data.get("words") or [] if str(w.get("word", "")).strip()]
    segs = []
    for s in data.get("segments") or []:
        start, end = float(s["start"]), float(s["end"])
        inside = [w for w in words if w.start >= start - 0.01 and w.end <= end + 0.01]
        segs.append(Segment(start, end, str(s["text"]).strip(), inside,
                            s.get("no_speech_prob"), s.get("avg_logprob")))
    if not segs and data.get("text"):
        end = words[-1].end if words else float(data.get("duration") or 0.0)
        segs.append(Segment(0.0, end, str(data["text"]).strip(), words))
    return segs


class OpenAITranscriber:
    """OpenAI-compatible speech-to-text API (OpenAI, Groq, or a self-hosted server)."""

    name = "openai"

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, prompt: str | None = None, timeout: float = 600):
        self.model = model or os.getenv("SHORTS_FORGE_STT_MODEL", "whisper-1")
        self.base_url = (base_url or os.getenv("SHORTS_FORGE_STT_BASE_URL")
                         or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = (api_key or os.getenv("SHORTS_FORGE_STT_API_KEY")
                        or os.getenv("OPENAI_API_KEY", ""))
        self.prompt = prompt
        self.timeout = timeout

    def transcribe(self, media: Path, language: str | None = None) -> list[Segment]:
        with tempfile.TemporaryDirectory(prefix="shorts-forge-") as tmp:
            audio = extract_audio(media, Path(tmp) / "audio.mp3")
            fields: dict[str, str | list[str]] = {
                "model": self.model, "response_format": "verbose_json",
                "timestamp_granularities[]": ["segment", "word"],
            }
            if language:
                fields["language"] = language
            if self.prompt:
                fields["prompt"] = self.prompt
            body, ctype = _multipart(fields, "file", audio)
            req = urllib.request.Request(f"{self.base_url}/audio/transcriptions", data=body,
                                         headers={"Content-Type": ctype,
                                                  "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        return parse_verbose_json(data)


def segments_to_json(segments: list[Segment]) -> str:
    def seg(s: Segment) -> dict:
        return {"start": round(s.start, 3), "end": round(s.end, 3), "text": s.text,
                "words": [{"start": round(w.start, 3), "end": round(w.end, 3), "word": w.text}
                          for w in s.words]}
    return json.dumps({"segments": [seg(s) for s in segments]}, ensure_ascii=False, indent=1)


def segments_from_json(text: str) -> list[Segment]:
    """Read our own transcript JSON (segments with nested words) or OpenAI verbose_json."""
    data = json.loads(text)
    if isinstance(data, list):
        data = {"segments": data}
    segs = data.get("segments") or []
    if segs and any("words" in s for s in segs):
        out = []
        for s in segs:
            words = [Word(float(w["start"]), float(w["end"]), str(w["word"]).strip(),
                          w.get("probability"))
                     for w in s.get("words") or [] if str(w.get("word", "")).strip()]
            out.append(Segment(float(s["start"]), float(s["end"]), str(s["text"]).strip(), words,
                               s.get("no_speech_prob"), s.get("avg_logprob")))
        return out
    return parse_verbose_json(data)


class JsonTranscriber:
    name = "json"

    def __init__(self, path: Path):
        self.path = Path(path)

    def transcribe(self, media: Path | None = None, language: str | None = None) -> list[Segment]:
        return segments_from_json(self.path.read_text(encoding="utf-8"))
