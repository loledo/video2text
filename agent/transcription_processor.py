import json
import logging
from pathlib import Path

import anthropic

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a travel content analyst specializing in processing raw speech-to-text transcriptions from TikTok travel videos. Your job is to take noisy Whisper transcription output and transform it into clean, structured knowledge chunks optimized for retrieval-augmented generation (RAG).

The transcription comes from a travel creator's TikTok video. The speaker may use informal language, filler words, incomplete sentences, and jump between topics. Your job is to extract the signal (travel knowledge) and discard the noise (verbal tics, repetitions, filler).

Output rules:
- Split the content into 1–4 self-contained knowledge chunks. Each chunk should cover a distinct travel tip, place, or recommendation.
- Each chunk must stand alone — a reader with no other context must understand it.
- Remove filler words ("like", "um", "you know", "literally"), repetitions, and self-corrections. Keep the creator's voice and specificity.
- Never invent details not present in the original transcription.
- If the transcription is incoherent, empty, or contains no travel information, return an empty chunks array and set "extractable": false.
- Extract the destination(s) mentioned. If none is clearly stated, set destinations to an empty array.
- Classify the primary content type: "recommendation", "tip", "warning", "itinerary", "review", or "general".
- Return ONLY valid JSON. No explanation text before or after.\
"""

_USER_TEMPLATE = """\
Process the following Whisper transcription from a TikTok travel video into structured RAG chunks.

<source_metadata>
  <filename>{filename}</filename>
  <duration_seconds>{duration}</duration_seconds>
</source_metadata>

<raw_transcription>
{whisper_text}
</raw_transcription>

<timestamped_chunks>
{whisper_chunks_json}
</timestamped_chunks>

Return valid JSON matching this exact schema:
{{
  "source_file": "{filename}",
  "extractable": true,
  "destinations": ["<city or country>"],
  "chunks": [
    {{
      "chunk_id": "{stem}_chunk_0",
      "type": "recommendation|tip|warning|itinerary|review|general",
      "destinations": ["<city or country>"],
      "text": "<cleaned, self-contained travel knowledge paragraph>",
      "keywords": ["<3-6 relevant travel keywords>"],
      "approximate_timestamp_start": <float or null>,
      "approximate_timestamp_end": <float or null>
    }}
  ]
}}\
"""


def _strip_code_fence(text: str) -> str:
    """Remove markdown code fences from Claude's response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Strip opening fence (```json or ```)
        lines = lines[1:]
        # Strip closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def process_transcription(
    whisper_result: dict,
    filename: str,
    duration_s: float | None = None,
) -> dict:
    """
    Process a Whisper transcription result through Claude to produce structured RAG chunks.

    Args:
        whisper_result: dict with keys "text" (str) and "chunks" (list of dicts)
        filename: original source filename (used in chunk IDs and metadata)
        duration_s: video duration in seconds, or None if unknown

    Returns:
        Parsed dict with keys: source_file, extractable, destinations, chunks
    """
    stem = Path(filename).stem
    fallback = {
        "source_file": filename,
        "extractable": False,
        "destinations": [],
        "chunks": [],
    }

    whisper_text = whisper_result.get("text", "")
    raw_chunks = whisper_result.get("chunks", [])

    # Serialise Whisper chunks for the prompt
    whisper_chunks_json = json.dumps(raw_chunks, indent=2, ensure_ascii=False)

    user_message = _USER_TEMPLATE.format(
        filename=filename,
        duration=duration_s if duration_s is not None else "unknown",
        whisper_text=whisper_text,
        whisper_chunks_json=whisper_chunks_json,
        stem=stem,
    )

    try:
        client = anthropic.Anthropic()
        log.info("Calling Claude to structure transcription for: %s", filename)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        raw_text = response.content[0].text
        clean_text = _strip_code_fence(raw_text)
        result = json.loads(clean_text)
        log.info("Successfully processed transcription for: %s", filename)
        return result
    except json.JSONDecodeError as exc:
        log.error("JSON parse error for %s: %s", filename, exc)
        return fallback
    except Exception as exc:
        log.error("Claude API error for %s: %s", filename, exc)
        return fallback
