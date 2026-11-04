#!/usr/bin/env bash
# Fully offline demo: no Whisper weights, no API key.
# Renders a synthetic test video with ffmpeg, uses a hand-written SAMPLE transcript,
# picks highlights with the heuristic scorer, face-tracks to 9:16 and burns captions.
set -euo pipefail
shorts-forge demo -o demo-shorts
ls -lh demo-shorts
# Same video, mock LLM path (exercises prompt -> JSON -> validation without a key),
# karaoke captions, re-using the sample transcript:
shorts-forge demo-shorts/demo-source.mp4 -o demo-shorts/mock \
  --transcript examples/sample.transcript.json --llm mock --style karaoke --count 2
