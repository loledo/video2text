#!/usr/bin/env python3
"""Waypoint interactive travel chat CLI."""
import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from agent import chunk_store
from agent.travel_agent import TravelAgent

log = logging.getLogger(__name__)


def _collect_destinations(chunks: list[dict]) -> set[str]:
    """Return the set of unique destination names across all chunks."""
    destinations: set[str] = set()
    for chunk in chunks:
        for dest in chunk.get("destinations", []):
            if dest:
                destinations.add(dest)
    return destinations


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(
        prog="waypoint",
        description="Chat with Waypoint, your TikTok-powered travel agent.",
    )
    parser.add_argument(
        "--chunks-dir",
        default="chunks",
        help="Directory containing chunk JSON files (default: chunks)",
    )
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)

    # Validate chunks directory
    if not chunks_dir.exists() or not any(chunks_dir.glob("*.json")):
        print(
            f"No chunks found in '{chunks_dir}'.\n"
            "Run: python process_videos.py <your-videos-dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate required environment variables
    if not os.environ.get("TAVILY_API_KEY"):
        print(
            "Error: TAVILY_API_KEY environment variable is not set.\n"
            "Get a free key at https://tavily.com and run:\n"
            "  export TAVILY_API_KEY=tvly-...",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load chunks
    log.info("Loading chunks from %s ...", chunks_dir)
    chunks = chunk_store.load_all_chunks(chunks_dir)

    if not chunks:
        print(
            f"No extractable chunks found in '{chunks_dir}'.\n"
            "Run: python process_videos.py <your-videos-dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    destinations = _collect_destinations(chunks)
    dest_count = len(destinations)
    dest_preview = ", ".join(sorted(destinations)[:5])
    if dest_count > 5:
        dest_preview += f", ... (+{dest_count - 5} more)"

    # Startup banner
    print(
        f"\n{'=' * 60}\n"
        f"  Waypoint — Your TikTok-Powered Travel Agent\n"
        f"{'=' * 60}\n"
        f"  Loaded {len(chunks)} chunk(s) covering {dest_count} destination(s)\n"
        f"  Destinations: {dest_preview}\n"
        f"{'=' * 60}\n"
        f"  Type /quit, /exit, or /q to exit. Ctrl+C also works.\n"
        f"{'=' * 60}\n"
    )

    agent = TravelAgent(chunks=chunks)

    # REPL loop
    while True:
        try:
            user_input = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            log.info("Session ended by user.")
            break

        user_input = user_input.strip()

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye! Safe travels.")
            break

        try:
            response = agent.chat(user_input)
        except Exception as exc:
            log.error("Unexpected error: %s", exc, exc_info=True)
            print(f"\n[Error: {exc}]")
            continue

        print(f"\nWaypoint: {response}\n")


if __name__ == "__main__":
    main()
