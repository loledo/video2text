import shutil
import subprocess
import tempfile
from pathlib import Path


def extract_audio(video_path: str) -> str:
    """Extract audio from video to a 16kHz mono WAV temp file. Returns the temp file path."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found. Install it: brew install ffmpeg  |  apt install ffmpeg  |  winget install ffmpeg"
        )

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video),
        "-ar", "16000",   # 16kHz sample rate required by Whisper
        "-ac", "1",       # mono
        "-vn",            # no video
        "-f", "wav",
        tmp.name,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{result.stderr}")

    return tmp.name
