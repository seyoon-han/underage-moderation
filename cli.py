#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from underage_moderation.pipeline import run_underage_moderation

SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}


def iter_targets(target_path: str | Path) -> list[Path]:
    resolved_target = Path(target_path).expanduser().resolve()
    if resolved_target.is_file():
        return [resolved_target]
    if resolved_target.is_dir():
        return sorted(
            path
            for path in resolved_target.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
        )
    raise FileNotFoundError(f"Target path does not exist: {resolved_target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run underage voice moderation on an audio file or directory."
    )
    parser.add_argument(
        "target",
        help="Path to an audio file or a directory of audio files.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Execution device override. Defaults to the best available device.",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=None,
        help="Optional maximum duration to process per file, in seconds.",
    )
    parser.add_argument(
        "--age-threshold",
        type=float,
        default=0.30,
        help="Flag when predicted age score is at or below this threshold.",
    )
    parser.add_argument(
        "--child-margin",
        type=float,
        default=0.0,
        help="Required margin above adult gender probabilities before the child rule hits.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser


def print_text_result(result: dict[str, object]) -> None:
    input_block = result["input"]
    moderation_block = result["moderation"]
    scores_block = result["scores"]
    runtime_block = result["runtime"]
    print(f"file: {input_block['source_path']}")
    print(f"decision: {moderation_block['decision']}")
    print(f"is_underage: {moderation_block['is_underage']}")
    print(f"age_score: {scores_block['age_score']:.4f}")
    print(f"estimated_age_years: {scores_block['estimated_age_years']:.2f}")
    print(f"child_probability: {scores_block['child_probability']:.4f}")
    print(f"female_probability: {scores_block['female_probability']:.4f}")
    print(f"male_probability: {scores_block['male_probability']:.4f}")
    print(f"device: {runtime_block['device']}")
    print()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    targets = iter_targets(args.target)
    if not targets:
        raise SystemExit("No supported audio files were found.")

    results = []
    for target in targets:
        result = run_underage_moderation(
            audio_path=target,
            preferred_device=args.device,
            max_duration_seconds=args.max_duration,
            age_threshold=args.age_threshold,
            child_probability_margin=args.child_margin,
        )
        results.append(result)

    if args.json:
        payload: object = results[0] if len(results) == 1 else results
        print(json.dumps(payload, indent=2))
    else:
        for result in results:
            print_text_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
