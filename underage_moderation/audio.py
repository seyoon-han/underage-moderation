from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000


@dataclass(frozen=True)
class LoadedAudio:
    samples: np.ndarray
    sample_rate: int
    duration_seconds: float
    source_path: Path
    normalized_path: Path


def _run_ffmpeg(source_path: Path, target_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg is required to normalize audio input. "
            "Install ffmpeg and retry."
        )

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-f",
        "wav",
        str(target_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def normalize_audio_file(
    source_path: str | Path, working_directory: Path | None = None
) -> Path:
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Audio file not found: {source}")

    target_dir = working_directory or Path(
        tempfile.mkdtemp(prefix="underage-audio-normalized-")
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{source.stem}.normalized.wav"
    _run_ffmpeg(source, target_path)
    return target_path


def load_audio_file(
    source_path: str | Path, max_duration_seconds: int | None = None
) -> LoadedAudio:
    normalized_path = normalize_audio_file(source_path)
    samples, sample_rate = sf.read(str(normalized_path), dtype="float32")

    if samples.ndim > 1:
        samples = samples.mean(axis=1)

    if max_duration_seconds is not None:
        max_samples = max_duration_seconds * sample_rate
        samples = samples[:max_samples]

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 1.0:
        samples = samples / peak

    duration_seconds = float(samples.shape[0] / sample_rate) if sample_rate else 0.0
    return LoadedAudio(
        samples=samples,
        sample_rate=sample_rate,
        duration_seconds=duration_seconds,
        source_path=Path(source_path).expanduser().resolve(),
        normalized_path=normalized_path,
    )
