from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .policy import ModerationPolicy, decide_underage


@dataclass(frozen=True)
class ReviewSetEntry:
    case_id: str
    label: bool
    scores: dict[str, float]


def _coerce_label(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "flag", "flagged", "positive", "underage"}:
            return True
        if normalized in {"0", "false", "allow", "negative", "adult"}:
            return False
    raise ValueError(f"Unsupported label value: {value!r}")


def _extract_scores(payload: dict[str, object]) -> dict[str, float]:
    source = payload.get("scores")
    if isinstance(source, dict):
        score_payload = source
    else:
        score_payload = payload

    return {
        "age_score": float(score_payload["age_score"]),
        "child_probability": float(score_payload["child_probability"]),
        "female_probability": float(score_payload["female_probability"]),
        "male_probability": float(score_payload["male_probability"]),
    }


def load_review_set(review_set_path: str | Path) -> list[ReviewSetEntry]:
    path = Path(review_set_path).expanduser().resolve()
    entries: list[ReviewSetEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {index}.")

            case_id = str(payload.get("id") or f"case-{index}")
            entries.append(
                ReviewSetEntry(
                    case_id=case_id,
                    label=_coerce_label(payload["label"]),
                    scores=_extract_scores(payload),
                )
            )

    if not entries:
        raise ValueError("Review set is empty.")
    return entries


def evaluate_policy(
    entries: list[ReviewSetEntry],
    policy: ModerationPolicy,
) -> dict[str, object]:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for entry in entries:
        decision = decide_underage(entry.scores, policy)
        predicted_positive = bool(decision["is_underage"])
        expected_positive = entry.label

        if predicted_positive and expected_positive:
            true_positive += 1
        elif predicted_positive and not expected_positive:
            false_positive += 1
        elif not predicted_positive and not expected_positive:
            true_negative += 1
        else:
            false_negative += 1

    total = len(entries)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    specificity_denominator = true_negative + false_positive
    f1_denominator = (2 * true_positive) + false_positive + false_negative

    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    specificity = (
        true_negative / specificity_denominator if specificity_denominator else 0.0
    )
    accuracy = (true_positive + true_negative) / total if total else 0.0
    false_positive_rate = (
        false_positive / specificity_denominator if specificity_denominator else 0.0
    )
    f1 = (2 * true_positive / f1_denominator) if f1_denominator else 0.0

    return {
        "policy": {
            "age_threshold": policy.age_threshold,
            "child_probability_margin": policy.child_probability_margin,
        },
        "counts": {
            "evaluated_cases": total,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "false_positive_rate": false_positive_rate,
            "f1": f1,
        },
    }


def sweep_policies(
    entries: list[ReviewSetEntry],
    age_thresholds: list[float],
    child_probability_margins: list[float],
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for age_threshold in age_thresholds:
        for child_probability_margin in child_probability_margins:
            results.append(
                evaluate_policy(
                    entries,
                    ModerationPolicy(
                        age_threshold=age_threshold,
                        child_probability_margin=child_probability_margin,
                    ),
                )
            )
    return results


def rank_policy_results(
    results: list[dict[str, object]],
    sort_by: str,
) -> list[dict[str, object]]:
    descending = sort_by != "false_positive_rate"

    def sort_key(result: dict[str, object]) -> tuple[float, float, float, float]:
        metrics = result["metrics"]
        policy = result["policy"]
        primary = float(metrics[sort_by])
        recall = float(metrics["recall"])
        precision = float(metrics["precision"])
        threshold = -float(policy["age_threshold"]) if descending else float(
            policy["age_threshold"]
        )
        return primary, recall, precision, threshold

    return sorted(results, key=sort_key, reverse=descending)
