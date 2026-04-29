import logging
import math

import soundfile as sf
import torch
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

log = logging.getLogger(__name__)

_DEFAULT_MODELS = {
    "cuda": "openai/whisper-large-v3-turbo",
    "mps": "openai/whisper-large-v3-turbo",
    "cpu": "openai/whisper-small",
}

_BATCH_SIZES = {
    "cuda": 16,
    "mps": 4,
    "cpu": 2,
}

CHUNK_LENGTH_S = 30


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe(
    audio_path: str,
    model_id: str | None = None,
    device: str = "auto",
    return_timestamps: bool = True,
) -> dict:
    """
    Transcribe audio file using a Hugging Face Whisper model.

    Returns a dict with keys:
      - "text": full transcription string
      - "chunks": list of {"timestamp": (start, end), "text": str}
    """
    if device == "auto":
        device = detect_device()

    resolved_model = model_id or _DEFAULT_MODELS.get(device, "openai/whisper-small")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    log.info("Device: %s  |  Model: %s", device, resolved_model)

    log.info("Loading model weights...")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        resolved_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    log.info("Moving model to %s...", device)
    model.to(device)

    processor = AutoProcessor.from_pretrained(resolved_model)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
        device=device,
    )

    log.info("Reading audio file...")
    audio_data, sample_rate = sf.read(audio_path, dtype="float32")
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)

    chunk_samples = CHUNK_LENGTH_S * sample_rate
    total_chunks = math.ceil(len(audio_data) / chunk_samples)
    duration_s = len(audio_data) / sample_rate
    log.info("Audio: %.1f s  |  %d x %ds chunks", duration_s, total_chunks, CHUNK_LENGTH_S)

    all_text: list[str] = []
    all_chunks: list[dict] = []

    with tqdm(total=total_chunks, desc="Transcribing", unit="chunk", dynamic_ncols=True) as pbar:
        for idx in range(total_chunks):
            start = idx * chunk_samples
            chunk = audio_data[start : start + chunk_samples]
            offset_s = idx * CHUNK_LENGTH_S

            result = asr(
                {"array": chunk, "sampling_rate": sample_rate},
                return_timestamps=return_timestamps,
            )

            all_text.append(result["text"].strip())

            for c in result.get("chunks", []):
                ts = c["timestamp"]
                t_start = (ts[0] or 0.0) + offset_s
                t_end = (ts[1] + offset_s) if ts[1] is not None else None
                all_chunks.append({"timestamp": (t_start, t_end), "text": c["text"]})

            pbar.update(1)

    return {"text": " ".join(all_text).strip(), "chunks": all_chunks}
