import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


def save_chunks(chunks_data: dict, out_path: str | Path) -> None:
    """Write chunks_data as JSON (indent=2) to out_path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks_data, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Saved chunks: %s", out_path)


def load_all_chunks(chunks_dir: str | Path) -> list[dict]:
    """
    Read all *.json files in chunks_dir and return a flat list of all chunk dicts
    across all files where extractable == True, sorted by source_file.
    """
    chunks_dir = Path(chunks_dir)
    all_chunks: list[dict] = []

    json_files = sorted(chunks_dir.glob("*.json"))
    for json_file in json_files:
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Could not read %s: %s", json_file, exc)
            continue

        if not data.get("extractable", False):
            continue

        source_file = data.get("source_file", json_file.stem)
        for chunk in data.get("chunks", []):
            chunk_copy = dict(chunk)
            chunk_copy.setdefault("source_file", source_file)
            all_chunks.append(chunk_copy)

    all_chunks.sort(key=lambda c: c.get("source_file", ""))
    return all_chunks
