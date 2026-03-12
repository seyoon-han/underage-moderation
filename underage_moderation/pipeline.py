from __future__ import annotations

from pathlib import Path

from .audio import load_audio_file
from .model import predict_scores
from .policy import DEFAULT_AGE_THRESHOLD, ModerationPolicy, decide_underage


def run_underage_moderation(
    audio_path: str | Path,
    preferred_device: str | None = None,
    max_duration_seconds: int | None = None,
    age_threshold: float = DEFAULT_AGE_THRESHOLD,
    child_probability_margin: float = 0.0,
) -> dict[str, object]:
    loaded_audio = load_audio_file(
        source_path=audio_path,
        max_duration_seconds=max_duration_seconds,
    )
    policy = ModerationPolicy(
        age_threshold=age_threshold,
        child_probability_margin=child_probability_margin,
    )

    try:
        scores, runtime = predict_scores(
            loaded_audio.samples,
            loaded_audio.sample_rate,
            preferred_device=preferred_device,
        )
        decision = decide_underage(scores, policy)
        return {
            "input": {
                "source_path": str(loaded_audio.source_path),
                "normalized_path": str(loaded_audio.normalized_path),
                "duration_seconds": loaded_audio.duration_seconds,
                "sample_rate": loaded_audio.sample_rate,
            },
            "scores": scores,
            "moderation": decision,
            "runtime": {
                "device": runtime.device.type,
                "model_source": runtime.model_source,
            },
        }
    finally:
        if loaded_audio.normalized_path.exists():
            loaded_audio.normalized_path.unlink()
        parent_directory = loaded_audio.normalized_path.parent
        if parent_directory.exists() and not any(parent_directory.iterdir()):
            parent_directory.rmdir()
