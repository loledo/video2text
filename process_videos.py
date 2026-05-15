#!/usr/bin/env python3
"""
process_videos.py — CLI entry point for the TikTok Travel Agent extraction pipeline.

Usage:
    python process_videos.py <input_dir> [--chunks-dir CHUNKS_DIR] [--device DEVICE] [--skip-existing]
"""
import argparse
import logging
import sys
from pathlib import Path

from tqdm import tqdm

from agent.extractor import IMAGE_EXTS, VIDEO_EXTS, extract_file

log = logging.getLogger(__name__)

SUPPORTED_EXTS = IMAGE_EXTS | VIDEO_EXTS


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="process_videos",
        description=(
            "Walk an input directory and extract travel knowledge chunks from "
            "TikTok videos and images using Whisper + Florence-2 + Claude."
        ),
    )
    parser.add_argument("input_dir", help="Directory containing video/image files to process.")
    parser.add_argument(
        "--chunks-dir",
        default="chunks",
        help="Directory to write chunk JSON files (default: ./chunks).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for local model inference (default: auto).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose chunk JSON already exists in chunks-dir.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    chunks_dir = Path(args.chunks_dir)

    if not input_dir.is_dir():
        log.error("Input directory does not exist: %s", input_dir)
        sys.exit(1)

    chunks_dir.mkdir(parents=True, exist_ok=True)

    # Collect supported files (case-insensitive extension check)
    all_files = [
        f for f in sorted(input_dir.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ]

    if not all_files:
        log.warning("No supported files found in: %s", input_dir)
        sys.exit(0)

    log.info("Found %d supported file(s) in %s", len(all_files), input_dir)

    n_processed = 0
    n_skipped = 0
    n_failed = 0

    with tqdm(total=len(all_files), desc="Processing files", unit="file", dynamic_ncols=True) as pbar:
        for file_path in all_files:
            stem = file_path.stem
            out_path = chunks_dir / f"{stem}.json"

            if args.skip_existing and out_path.exists():
                log.info("Skipping (already exists): %s", file_path.name)
                n_skipped += 1
                pbar.update(1)
                continue

            pbar.set_description(f"Processing {file_path.name}")
            try:
                result = extract_file(str(file_path), str(chunks_dir))
                if result.get("extractable", False):
                    n_processed += 1
                    log.info("Done: %s  (%d chunk(s))", file_path.name, len(result.get("chunks", [])))
                else:
                    log.info("Not extractable: %s", file_path.name)
                    n_processed += 1  # still counts as processed (not a failure)
            except Exception as exc:
                log.error("Failed: %s — %s", file_path.name, exc)
                n_failed += 1

            pbar.update(1)

    # Summary
    print(
        f"\nSummary: {n_processed} processed | {n_skipped} skipped | {n_failed} failed",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
