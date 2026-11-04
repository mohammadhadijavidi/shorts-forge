# Contributing

Thanks for helping make shorts-forge better.

1. Fork and clone, then `pip install -e ".[dev]"` (add `[whisper]` to run real transcription).
2. Run the tests: `pytest -q`. They need `ffmpeg` (with libass) on PATH and no model weights or API keys:
   transcription is mocked, but cropping and caption burning run through real ffmpeg.
3. Keep pull requests small and focused, with a test for every bug fix. Hook-scoring changes should
   come with a transcript snippet in `tests/test_highlights.py` showing the before/after pick.
4. Regenerate the README images with `python scripts/make_assets.py` (needs Playwright + Chromium)
   if you change the caption look or the pipeline.
5. Never commit API keys, or videos/transcripts of people who did not agree to it.

Good first issues: more caption styles, speaker-diarization-aware cropping (two-shot layouts),
silence trimming inside clips, an SRT/VTT export next to the ASS file.
