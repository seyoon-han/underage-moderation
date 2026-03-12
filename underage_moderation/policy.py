from __future__ import annotations

from dataclasses import dataclass

DEFAULT_AGE_THRESHOLD = 0.30


@dataclass(frozen=True)
class ModerationPolicy:
    age_threshold: float = DEFAULT_AGE_THRESHOLD
    child_probability_margin: float = 0.0


def decide_underage(
    scores: dict[str, float],
    policy: ModerationPolicy,
) -> dict[str, object]:
    child_probability = scores["child_probability"]
    competing_probability = max(
        scores["female_probability"],
        scores["male_probability"],
    )
    child_probability_rule = (
        child_probability >= competing_probability + policy.child_probability_margin
    )
    age_threshold_rule = scores["age_score"] <= policy.age_threshold
    is_underage = bool(child_probability_rule or age_threshold_rule)

    return {
        "is_underage": is_underage,
        "decision": "flag" if is_underage else "allow",
        "rule_hits": {
            "child_probability_rule": child_probability_rule,
            "age_threshold_rule": age_threshold_rule,
        },
        "policy": {
            "age_threshold": policy.age_threshold,
            "child_probability_margin": policy.child_probability_margin,
        },
    }
