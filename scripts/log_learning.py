#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from underage_moderation.learning import append_learning_entry


def parse_tags(raw: str) -> list[str]:
    return [tag.strip() for tag in raw.split(",") if tag.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Append a structured learning entry for reviewer feedback, runtime errors, "
            "or missing capabilities."
        )
    )
    parser.add_argument(
        "kind",
        choices=["learning", "error", "feature-request"],
        help="Type of entry to append.",
    )
    parser.add_argument(
        "--summary",
        required=True,
        help="One-line summary of the observation.",
    )
    parser.add_argument(
        "--details",
        required=True,
        help="Full context, evidence, or reviewer note.",
    )
    parser.add_argument(
        "--suggested-action",
        required=True,
        help="Next action to test or implement.",
    )
    parser.add_argument(
        "--category",
        default="insight",
        help="Learning category. Used only for learning entries.",
    )
    parser.add_argument(
        "--priority",
        default="medium",
        choices=["low", "medium", "high", "critical"],
        help="Priority level for the entry.",
    )
    parser.add_argument(
        "--area",
        default="policy",
        help="Area affected, for example policy, runtime, docs, or review-ops.",
    )
    parser.add_argument(
        "--source",
        default="manual",
        help="Where the learning came from, for example manual, qa, or reviewer.",
    )
    parser.add_argument(
        "--tags",
        default="",
        help="Comma-separated tags for search and grouping.",
    )
    parser.add_argument(
        "--output-dir",
        default=".learnings",
        help="Directory where markdown log files should be stored.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    target_file = append_learning_entry(
        kind=args.kind,
        summary=args.summary,
        details=args.details,
        suggested_action=args.suggested_action,
        output_dir=Path(args.output_dir),
        category=args.category,
        priority=args.priority,
        area=args.area,
        source=args.source,
        tags=parse_tags(args.tags),
    )
    print(f"entry_written: {target_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
