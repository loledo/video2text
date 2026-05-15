import json
import logging
import subprocess
import tempfile
from pathlib import Path

import anthropic
import torch

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Florence-2 lazy singleton
# ---------------------------------------------------------------------------

_florence_model = None
_florence_processor = None


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_florence():
    global _florence_model, _florence_processor
    if _florence_model is None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        device = _detect_device()
        log.info("Loading Florence-2-large on device: %s", device)
        _florence_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/Florence-2-large",
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True,
        ).to(device)
        _florence_processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-large",
            trust_remote_code=True,
        )
        log.info("Florence-2-large loaded.")
    return _florence_model, _florence_processor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VISUAL_SYSTEM_PROMPT = """\
You are a travel content analyst. You will receive text extracted by OCR from frames of a TikTok travel video or image. Your job is to synthesize this raw OCR output into clean, structured travel knowledge chunks optimized for RAG.

Rules:
- Deduplicate text that appears identically across multiple frames.
- Reconstruct list-style content (numbered tips, bullet points) into coherent sequential form.
- Never describe visuals — only process the TEXT content.
- If no readable travel-relevant text is present (only music credits, generic captions, or pure aesthetic content), set "extractable": false and return empty chunks.
- Extract destinations mentioned. If unclear, set destinations to [].
- Classify each chunk type: "recommendation", "tip", "warning", "itinerary", "review", "list", or "general".
- Return ONLY valid JSON. No explanation text before or after.\
"""

_VISUAL_USER_TEMPLATE = """\
Structure the following OCR-extracted text from a TikTok travel video/image into RAG chunks.

<source_metadata>
  <filename>{filename}</filename>
  <content_type>{content_type}</content_type>
</source_metadata>

<ocr_output>
{raw_text_by_frame_json}
</ocr_output>

Return valid JSON:
{{
  "source_file": "{filename}",
  "extractable": true,
  "destinations": [],
  "raw_text_by_frame": {raw_text_by_frame_json},
  "chunks": [
    {{
      "chunk_id": "{stem}_visual_chunk_0",
      "type": "recommendation|tip|warning|itinerary|review|list|general",
      "destinations": [],
      "text": "<synthesized travel knowledge>",
      "keywords": [],
      "source_frames": [0, 1]
    }}
  ]
}}\
"""


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _find_ffmpeg() -> str:
    """Reuse the same ffmpeg discovery logic as video2text.audio."""
    from video2text.audio import _find_ffmpeg as _base_find_ffmpeg
    return _base_find_ffmpeg()


def _find_ffprobe() -> str:
    """Find ffprobe binary alongside ffmpeg."""
    import shutil
    path = shutil.which("ffprobe")
    if path:
        return path
    # Try replacing 'ffmpeg' with 'ffprobe' in the located ffmpeg path
    ffmpeg_path = _find_ffmpeg()
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    if Path(ffprobe_path).exists():
        return ffprobe_path
    raise RuntimeError("ffprobe not found. Install ffmpeg (which includes ffprobe).")


def _run_ocr_on_frame(frame_path: Path) -> str:
    """Run Florence-2 OCR on a single frame PNG. Returns extracted text string."""
    from PIL import Image

    model, processor = _get_florence()
    device = next(model.parameters()).device

    img = Image.open(frame_path).convert("RGB")
    inputs = processor(
        text="<OCR>",
        images=img,
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed = processor.post_process_generation(
        generated_text,
        task="<OCR>",
        image_size=(img.width, img.height),
    )
    return parsed.get("<OCR>", "").strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_visual_text(file_path: str, filename: str) -> dict:
    """
    Two-stage pipeline:
      A) Sample frames (or use image directly) → Florence-2 OCR
      B) Claude structures the OCR output into RAG chunks

    Returns a dict with keys: source_file, extractable, destinations,
    raw_text_by_frame, chunks
    """
    file_path = Path(file_path)
    stem = Path(filename).stem

    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    is_image = file_path.suffix.lower() in image_exts

    raw_text_by_frame: list[dict] = []
    fallback = {
        "source_file": filename,
        "extractable": False,
        "destinations": [],
        "raw_text_by_frame": raw_text_by_frame,
        "chunks": [],
    }

    # ------------------------------------------------------------------
    # Stage A — Frame sampling + Florence-2 OCR
    # ------------------------------------------------------------------
    try:
        if is_image:
            log.info("Running Florence-2 OCR on image: %s", filename)
            ocr_text = _run_ocr_on_frame(file_path)
            raw_text_by_frame.append({
                "frame_index": 0,
                "timestamp_s": 0.0,
                "text_found": ocr_text,
            })
            content_type = "image"
        else:
            # Video — sample at 0.5 fps
            ffmpeg = _find_ffmpeg()
            log.info("Sampling frames from video: %s", filename)
            with tempfile.TemporaryDirectory() as tmpdir:
                frame_pattern = str(Path(tmpdir) / "frame_%04d.png")
                cmd = [
                    ffmpeg, "-y",
                    "-i", str(file_path),
                    "-vf", "fps=0.5",
                    "-frame_pts", "1",
                    "-loglevel", "error",
                    frame_pattern,
                ]
                subprocess.run(cmd, check=True, capture_output=True)

                frame_files = sorted(Path(tmpdir).glob("frame_*.png"))
                log.info("Extracted %d frames for OCR.", len(frame_files))

                for i, frame_path in enumerate(frame_files):
                    # Estimate timestamp: frame index * 2s (0.5 fps → 1 frame every 2s)
                    timestamp_s = float(i * 2)
                    ocr_text = _run_ocr_on_frame(frame_path)
                    raw_text_by_frame.append({
                        "frame_index": i,
                        "timestamp_s": timestamp_s,
                        "text_found": ocr_text,
                    })
                    log.debug("Frame %d OCR: %s", i, ocr_text[:80] if ocr_text else "(empty)")

            content_type = "video_frames"

    except Exception as exc:
        log.error("Stage A (OCR) failed for %s: %s", filename, exc)
        fallback["raw_text_by_frame"] = raw_text_by_frame
        return fallback

    # ------------------------------------------------------------------
    # Stage B — Claude structuring
    # ------------------------------------------------------------------
    raw_text_by_frame_json = json.dumps(raw_text_by_frame, indent=2, ensure_ascii=False)

    user_message = _VISUAL_USER_TEMPLATE.format(
        filename=filename,
        content_type=content_type,
        raw_text_by_frame_json=raw_text_by_frame_json,
        stem=stem,
    )

    try:
        client = anthropic.Anthropic()
        log.info("Calling Claude to structure visual OCR for: %s", filename)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=_VISUAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_response_text = response.content[0].text
        clean_text = _strip_code_fence(raw_response_text)
        result = json.loads(clean_text)
        log.info("Successfully structured visual content for: %s", filename)
        return result
    except json.JSONDecodeError as exc:
        log.error("JSON parse error (visual) for %s: %s", filename, exc)
    except Exception as exc:
        log.error("Claude API error (visual) for %s: %s", filename, exc)

    fallback["raw_text_by_frame"] = raw_text_by_frame
    return fallback
