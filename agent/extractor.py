import json
import logging
import os
import re
import subprocess
import shutil
import tempfile
from pathlib import Path

from video2text.audio import _find_ffmpeg, extract_audio, _get_duration_s
from video2text.transcriber import transcribe

from agent.chunk_store import save_chunks
from agent.transcription_processor import process_transcription
from agent.visual_extractor import extract_visual_text

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

SHORT_VIDEO_THRESHOLD_S = 90.0
MIN_SPEECH_WORDS = 8
_MUSIC_PATTERN = re.compile(r"[♪♫♩♬]|\[music\]|\[applause\]|\[laughter\]|\(music", re.IGNORECASE)


def _is_music_or_noise(whisper_result: dict) -> bool:
    """Return True when Whisper output is background music or ambient noise rather than speech."""
    text = whisper_result.get("text", "").strip()
    if not text:
        return True
    clean = _MUSIC_PATTERN.sub("", text).strip()
    word_count = sum(1 for w in clean.split() if any(c.isalpha() for c in w))
    if word_count < MIN_SPEECH_WORDS:
        return True
    music_hits = len(_MUSIC_PATTERN.findall(text))
    if music_hits / max(1, len(text.split())) > 0.3:
        return True
    return False


def _find_ffprobe() -> str:
    """Find ffprobe binary, falling back to replacing 'ffmpeg' with 'ffprobe'."""
    path = shutil.which("ffprobe")
    if path:
        return path
    ffmpeg_path = _find_ffmpeg()
    candidate = ffmpeg_path.replace("ffmpeg", "ffprobe")
    if Path(candidate).exists():
        return candidate
    raise RuntimeError("ffprobe not found. Install ffmpeg (which bundles ffprobe).")


def _has_audio(file_path: str | Path) -> bool:
    """Return True if the file has at least one audio stream."""
    try:
        ffprobe = _find_ffprobe()
        result = subprocess.run(
            [
                ffprobe,
                "-v", "quiet",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                str(file_path),
            ],
            capture_output=True,
            text=True,
        )
        return "audio" in result.stdout
    except Exception as exc:
        log.warning("ffprobe check failed for %s: %s — assuming no audio", file_path, exc)
        return False


def _get_video_duration(file_path: str | Path) -> float | None:
    """Return video duration in seconds using ffmpeg."""
    try:
        ffmpeg = _find_ffmpeg()
        return _get_duration_s(str(file_path), ffmpeg)
    except Exception as exc:
        log.warning("Could not determine duration for %s: %s", file_path, exc)
        return None


def _merge_results(audio_result: dict, visual_result: dict, filename: str) -> dict:
    """
    Combine audio (transcription) and visual (OCR) chunk dicts into one.
    Renumbers chunk_ids sequentially. extractable=True if either source is extractable.
    """
    audio_chunks = audio_result.get("chunks", [])
    visual_chunks = visual_result.get("chunks", [])

    # Merge destinations (deduplicate, preserve order)
    seen_dests: set[str] = set()
    merged_dests: list[str] = []
    for dest in audio_result.get("destinations", []) + visual_result.get("destinations", []):
        if dest not in seen_dests:
            seen_dests.add(dest)
            merged_dests.append(dest)

    stem = Path(filename).stem
    all_chunks: list[dict] = []

    for i, chunk in enumerate(audio_chunks + visual_chunks):
        chunk_copy = dict(chunk)
        chunk_copy["chunk_id"] = f"{stem}_chunk_{i}"
        all_chunks.append(chunk_copy)

    merged = {
        "source_file": filename,
        "extractable": audio_result.get("extractable", False) or visual_result.get("extractable", False),
        "destinations": merged_dests,
        "chunks": all_chunks,
    }

    # Preserve raw_text_by_frame from visual result if present
    if "raw_text_by_frame" in visual_result:
        merged["raw_text_by_frame"] = visual_result["raw_text_by_frame"]

    return merged


def extract_file(file_path: str | Path, chunks_dir: str | Path, device: str = "auto") -> dict:
    """
    Extract travel knowledge from a single image or video file.

    - Images or audio-less videos  → visual_extractor only
    - Videos with audio            → Whisper transcription always
                                     + visual_extractor if duration < 90s
    - Saves output JSON to chunks_dir/<stem>.json
    - Returns the result dict
    """
    file_path = Path(file_path)
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    filename = file_path.name
    stem = file_path.stem
    out_path = chunks_dir / f"{stem}.json"

    ext = file_path.suffix.lower()
    is_image = ext in IMAGE_EXTS
    is_video = ext in VIDEO_EXTS

    fallback = {
        "source_file": filename,
        "extractable": False,
        "destinations": [],
        "chunks": [],
    }

    if not is_image and not is_video:
        log.warning("Unsupported file type: %s", file_path)
        return fallback

    log.info("Processing: %s", file_path)

    try:
        if is_image:
            log.info("File is an image — using visual extractor.")
            result = extract_visual_text(str(file_path), filename, device=device)

        else:
            # Video path
            audio_present = _has_audio(file_path)

            if not audio_present:
                log.info("Video has no audio stream — using visual extractor only.")
                result = extract_visual_text(str(file_path), filename, device=device)
            else:
                duration_s = _get_video_duration(file_path)
                log.info("Video has audio. Duration: %.1f s", duration_s or 0)

                # Always run Whisper
                wav_path = None
                try:
                    wav_path = extract_audio(str(file_path))
                    whisper_result = transcribe(wav_path, device=device, return_timestamps=True)
                finally:
                    if wav_path and os.path.exists(wav_path):
                        os.unlink(wav_path)

                audio_result = process_transcription(whisper_result, filename, duration_s)

                no_speech = _is_music_or_noise(whisper_result)
                short_clip = duration_s is not None and duration_s < SHORT_VIDEO_THRESHOLD_S
                if no_speech:
                    log.info("No speech detected (music/noise) — forcing visual extraction.")
                run_visual = short_clip or no_speech
                if run_visual:
                    if short_clip and not no_speech:
                        log.info("Short clip (%.1fs) — also running visual extractor.", duration_s)
                    visual_result = extract_visual_text(str(file_path), filename, device=device)
                    result = _merge_results(audio_result, visual_result, filename)
                else:
                    result = audio_result

    except Exception as exc:
        log.error("Extraction failed for %s: %s", file_path, exc)
        result = fallback

    save_chunks(result, out_path)
    log.info("Saved: %s", out_path)
    return result
