import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

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
      - "chunks": list of {"timestamp": (start, end), "text": str}  (empty list if timestamps disabled)
    """
    if device == "auto":
        device = detect_device()

    resolved_model = model_id or _DEFAULT_MODELS.get(device, "openai/whisper-small")
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    print(f"[video2text] device={device}  model={resolved_model}", file=sys.stderr)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        resolved_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
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

    result = asr(
        audio_path,
        chunk_length_s=30,
        batch_size=_BATCH_SIZES.get(device, 4),
        return_timestamps=return_timestamps,
    )

    chunks = result.get("chunks", [])
    return {"text": result["text"].strip(), "chunks": chunks}
