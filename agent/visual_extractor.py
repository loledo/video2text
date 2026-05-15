import base64
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)

MAX_VIDEO_FRAMES = 12

_SYSTEM_PROMPT = """\
You are a travel content analyst. You receive one or more images from a TikTok travel post or video.
Read ALL text visible in the images: overlay text bubbles, pill-shaped captions, subtitles, on-screen graphics, \
visible signs, menu items, rating scores, place name labels, emoji+text combinations.
Synthesize what you read into clean structured travel knowledge chunks for a RAG system.

Rules:
- Extract every readable text element, including white/coloured overlay bubbles and numbered or bulleted lists.
- Deduplicate identical text that repeats across frames.
- Reconstruct numbered or bulleted sequences into coherent prose.
- If no travel-relevant text exists (only watermarks, music credits, app UI chrome), set extractable: false with empty chunks.
- Identify destination names from the text. If unclear, set destinations to [].
- Classify each chunk: "recommendation", "tip", "warning", "itinerary", "review", "list", or "general".
- Return ONLY valid JSON, no surrounding explanation text.\
"""

_USER_TEMPLATE = """\
Source file: {filename}{frame_note}

Extract all travel knowledge from the attached image(s) and return JSON:
{{
  "source_file": "{filename}",
  "extractable": true,
  "destinations": ["city or country"],
  "chunks": [
    {{
      "chunk_id": "{stem}_visual_chunk_0",
      "type": "tip",
      "destinations": [],
      "text": "self-contained travel knowledge paragraph",
      "keywords": ["keyword1", "keyword2"],
      "source_frames": [0]
    }}
  ]
}}\
"""


def _encode_image(path: Path) -> tuple[str, str]:
    media_types = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    media_type = media_types.get(path.suffix.lower(), "image/jpeg")
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _find_ffmpeg() -> str:
    from video2text.audio import _find_ffmpeg as _base
    return _base()


def extract_visual_text(file_path: str, filename: str, device: str = "auto") -> dict:
    """
    Send image or video frames directly to Claude Vision to extract travel knowledge.
    `device` is accepted for API compatibility but unused (no local model).
    """
    file_path = Path(file_path)
    stem = Path(filename).stem
    is_image = file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}

    fallback = {
        "source_file": filename,
        "extractable": False,
        "destinations": [],
        "chunks": [],
    }

    tmpdir_obj = None
    try:
        if is_image:
            log.info("Claude Vision — image: %s", filename)
            image_paths = [file_path]
            frame_note = ""
        else:
            log.info("Claude Vision — sampling frames from: %s", filename)
            ffmpeg = _find_ffmpeg()
            tmpdir_obj = tempfile.TemporaryDirectory()
            tmpdir = Path(tmpdir_obj.name)
            frame_pattern = str(tmpdir / "frame_%04d.png")
            subprocess.run(
                [ffmpeg, "-y", "-i", str(file_path), "-vf", "fps=0.5",
                 "-frame_pts", "1", "-loglevel", "error", frame_pattern],
                check=True, capture_output=True,
            )
            all_frames = sorted(tmpdir.glob("frame_*.png"))
            if len(all_frames) > MAX_VIDEO_FRAMES:
                step = len(all_frames) / MAX_VIDEO_FRAMES
                all_frames = [all_frames[round(i * step)] for i in range(MAX_VIDEO_FRAMES)]
            image_paths = all_frames
            frame_note = f"\n({len(image_paths)} frames sampled at 0.5 fps)"
            log.info("Sending %d frame(s) to Claude Vision.", len(image_paths))

        content: list[dict] = []
        for path in image_paths:
            data, media_type = _encode_image(path)
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data},
            })
        content.append({
            "type": "text",
            "text": _USER_TEMPLATE.format(filename=filename, stem=stem, frame_note=frame_note),
        })

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
        )
        result = json.loads(_strip_code_fence(response.content[0].text))
        log.info(
            "Visual extraction done — extractable=%s  chunks=%d",
            result.get("extractable"), len(result.get("chunks", [])),
        )
        return result

    except json.JSONDecodeError as exc:
        log.error("JSON parse error (visual) for %s: %s", filename, exc)
    except Exception as exc:
        log.error("Claude Vision error for %s: %s", filename, exc)
    finally:
        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()

    return fallback
