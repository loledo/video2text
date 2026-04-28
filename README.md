# video2text

Local video transcription CLI powered by [Hugging Face](https://huggingface.co/) Whisper models. No cloud APIs — all inference runs on your machine.

## Features

- Transcribes any video format supported by FFmpeg
- Auto-selects the best model for your hardware (CUDA / Apple MPS / CPU)
- Handles videos of any length via 30-second chunked processing
- Outputs plain text (`.txt`) and/or JSON with timestamps (`.json`)

## Requirements

- Python 3.11+
- [Poetry](https://python-poetry.org/)
- [FFmpeg](https://ffmpeg.org/) installed system-wide

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
apt install ffmpeg

# Windows
winget install ffmpeg
```

## Installation

```bash
git clone https://github.com/loledo/video2text.git
cd video2text
poetry install
```

## Usage

```bash
# Auto-detect device, save both .txt and .json alongside the input file
poetry run python transcribe.py video.mp4

# Output only plain text
poetry run python transcribe.py video.mp4 --output txt

# Force CPU with a lighter model
poetry run python transcribe.py video.mp4 --device cpu --model openai/whisper-base

# Save outputs to a specific directory
poetry run python transcribe.py video.mp4 --out-dir ./results
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--output` | `both` | `txt`, `json`, or `both` |
| `--model` | auto | Any HuggingFace Whisper model ID |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, or `mps` |
| `--out-dir` | input file dir | Directory to write output files |

## Model Selection

The default model is chosen automatically based on available hardware:

| Device | Default Model | Notes |
|---|---|---|
| CUDA / MPS | `openai/whisper-large-v3-turbo` | ~3GB VRAM with float16 |
| CPU | `openai/whisper-small` | Practical speed on CPU |

Models are downloaded from the HuggingFace Hub on first run and cached locally.

## Output

Given `lecture.mp4`, the tool produces:

**`lecture.txt`** — plain transcription text

**`lecture.json`** — structured output with per-segment timestamps:
```json
{
  "text": "Full transcription here...",
  "segments": [
    { "start": 0.0, "end": 4.5, "text": "Welcome to the lecture." },
    { "start": 4.5, "end": 9.2, "text": "Today we will cover..." }
  ]
}
```
