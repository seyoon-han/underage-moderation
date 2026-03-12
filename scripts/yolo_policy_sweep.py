#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from underage_moderation.experiments import (
    load_review_set,
    rank_policy_results,
    sweep_policies,
)


def parse_float_list(raw: str) -> list[float]:
    values = [segment.strip() for segment in raw.split(",")]
    parsed = [float(value) for value in values if value]
    if not parsed:
        raise argparse.ArgumentTypeError("Provide at least one numeric value.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a guarded YOLO-style threshold sweep against a labeled review set. "
            "This tool only evaluates stored scores; it does not touch production."
        )
    )
    parser.add_argument(
        "review_set",
        help="Path to a JSONL file with labels and model scores.",
    )
    parser.add_argument(
        "--age-thresholds",
        default="0.24,0.28,0.30,0.32,0.36",
        help="Comma-separated age thresholds to evaluate.",
    )
    parser.add_argument(
        "--child-margins",
        default="0.00,0.03,0.05,0.08",
        help="Comma-separated child probability margins to evaluate.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["accuracy", "precision", "recall", "specificity", "f1", "false_positive_rate"],
        default="f1",
        help="Primary metric used to rank candidate policies.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of ranked policy candidates to print.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output.",
    )
    return parser


def print_text_results(
    *,
    ranked_results: list[dict[str, object]],
    review_set_path: Path,
    evaluated_cases: int,
    sort_by: str,
) -> None:
    print(f"review_set: {review_set_path}")
    print(f"evaluated_cases: {evaluated_cases}")
    print(f"sort_by: {sort_by}")
    print()

    for index, result in enumerate(ranked_results, start=1):
        policy = result["policy"]
        metrics = result["metrics"]
        counts = result["counts"]
        print(f"rank: {index}")
        print(f"  age_threshold: {policy['age_threshold']:.2f}")
        print(f"  child_probability_margin: {policy['child_probability_margin']:.2f}")
        print(f"  precision: {metrics['precision']:.4f}")
        print(f"  recall: {metrics['recall']:.4f}")
        print(f"  specificity: {metrics['specificity']:.4f}")
        print(f"  false_positive_rate: {metrics['false_positive_rate']:.4f}")
        print(f"  f1: {metrics['f1']:.4f}")
        print(
            "  counts: "
            f"tp={counts['true_positive']} "
            f"fp={counts['false_positive']} "
            f"tn={counts['true_negative']} "
            f"fn={counts['false_negative']}"
        )
        print()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    review_set_path = Path(args.review_set).expanduser().resolve()
    review_entries = load_review_set(review_set_path)

    results = sweep_policies(
        review_entries,
        age_thresholds=parse_float_list(args.age_thresholds),
        child_probability_margins=parse_float_list(args.child_margins),
    )
    ranked_results = rank_policy_results(results, args.sort_by)[: args.top_k]

    payload = {
        "review_set": str(review_set_path),
        "evaluated_cases": len(review_entries),
        "sort_by": args.sort_by,
        "results": ranked_results,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print_text_results(
            ranked_results=ranked_results,
            review_set_path=review_set_path,
            evaluated_cases=len(review_entries),
            sort_by=args.sort_by,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
