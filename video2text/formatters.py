import json
from pathlib import Path


def save_txt(result: dict, output_path: str) -> None:
    Path(output_path).write_text(result["text"], encoding="utf-8")


def save_json(result: dict, output_path: str) -> None:
    segments = [
        {
            "start": chunk["timestamp"][0],
            "end": chunk["timestamp"][1],
            "text": chunk["text"].strip(),
        }
        for chunk in result.get("chunks", [])
    ]
    payload = {"text": result["text"], "segments": segments}
    Path(output_path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
