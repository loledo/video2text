import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger(__name__)


def _find_ffmpeg() -> str:
    """Return path to ffmpeg binary, using system install or bundled imageio-ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    raise RuntimeError(
        "ffmpeg not found. Install it: winget install ffmpeg  |  brew install ffmpeg  |  apt install ffmpeg\n"
        "Or add imageio-ffmpeg to your project: poetry add imageio-ffmpeg"
    )


def _get_duration_s(video_path: str, ffmpeg_exe: str) -> float | None:
    """Return video duration in seconds by parsing ffmpeg -i stderr."""
    result = subprocess.run(
        [ffmpeg_exe, "-i", str(video_path)],
        capture_output=True,
        text=True,
    )
    # ffmpeg -i exits with code 1 when no output file is given; duration is in stderr
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", result.stderr)
    if match:
        h, m, s, cs = (int(match.group(i)) for i in range(1, 5))
        return h * 3600 + m * 60 + s + cs / 100
    return None


def extract_audio(video_path: str) -> str:
    """Extract audio from video to a 16kHz mono WAV temp file. Returns the temp file path."""
    ffmpeg = _find_ffmpeg()
    log.info("Using ffmpeg: %s", ffmpeg)

    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    duration_s = _get_duration_s(str(video), ffmpeg)
    if duration_s:
        log.info("Video duration: %.1f s", duration_s)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    cmd = [
        ffmpeg, "-y",
        "-i", str(video),
        "-ar", "16000",
        "-ac", "1",
        "-vn",
        "-f", "wav",
        "-progress", "pipe:1",
        "-loglevel", "error",
        tmp.name,
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    with tqdm(
        total=int(duration_s) if duration_s else None,
        unit="s",
        desc="Extracting audio",
        dynamic_ncols=True,
    ) as pbar:
        last_s = 0
        for line in process.stdout:
            if line.startswith("out_time_us="):
                try:
                    current_s = int(line.split("=")[1].strip()) // 1_000_000
                    delta = current_s - last_s
                    if delta > 0:
                        pbar.update(delta)
                        last_s = current_s
                except ValueError:
                    pass

    process.wait()
    if process.returncode != 0:
        stderr_out = process.stderr.read()
        raise RuntimeError(f"ffmpeg failed:\n{stderr_out}")

    return tmp.name
