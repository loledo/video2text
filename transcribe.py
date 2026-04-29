#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from pathlib import Path

from video2text.audio import extract_audio
from video2text.formatters import save_json, save_txt
from video2text.transcriber import transcribe

log = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe a video file to text using a local Hugging Face Whisper model.",
    )
    parser.add_argument("input", help="Path to input video file")
    parser.add_argument(
        "--output",
        choices=["txt", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Hugging Face model ID (default: auto-selected by device)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory to save output files (default: same directory as input)",
    )

    args = parser.parse_args()

    video_path = args.input
    stem = Path(video_path).stem
    out_dir = Path(args.out_dir) if args.out_dir else Path(video_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Input: %s", video_path)
    wav_path = extract_audio(video_path)

    try:
        result = transcribe(
            wav_path,
            model_id=args.model,
            device=args.device,
            return_timestamps=True,
        )
    finally:
        os.unlink(wav_path)

    if args.output in ("txt", "both"):
        out_file = out_dir / f"{stem}.txt"
        save_txt(result, str(out_file))
        log.info("Saved: %s", out_file)

    if args.output in ("json", "both"):
        out_file = out_dir / f"{stem}.json"
        save_json(result, str(out_file))
        log.info("Saved: %s", out_file)

    print(result["text"])


if __name__ == "__main__":
    main()
