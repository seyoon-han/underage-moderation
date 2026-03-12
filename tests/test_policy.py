from __future__ import annotations

import unittest

from underage_moderation.policy import ModerationPolicy, decide_underage


class DecideUnderageTests(unittest.TestCase):
    def test_flags_when_age_score_is_below_threshold(self) -> None:
        result = decide_underage(
            {
                "age_score": 0.24,
                "child_probability": 0.20,
                "female_probability": 0.45,
                "male_probability": 0.35,
            },
            ModerationPolicy(age_threshold=0.30, child_probability_margin=0.05),
        )

        self.assertTrue(result["is_underage"])
        self.assertTrue(result["rule_hits"]["age_threshold_rule"])
        self.assertFalse(result["rule_hits"]["child_probability_rule"])

    def test_flags_when_child_probability_beats_competing_scores(self) -> None:
        result = decide_underage(
            {
                "age_score": 0.44,
                "child_probability": 0.56,
                "female_probability": 0.25,
                "male_probability": 0.19,
            },
            ModerationPolicy(age_threshold=0.30, child_probability_margin=0.10),
        )

        self.assertTrue(result["is_underage"])
        self.assertTrue(result["rule_hits"]["child_probability_rule"])
        self.assertFalse(result["rule_hits"]["age_threshold_rule"])

    def test_allows_when_neither_rule_hits(self) -> None:
        result = decide_underage(
            {
                "age_score": 0.62,
                "child_probability": 0.20,
                "female_probability": 0.41,
                "male_probability": 0.39,
            },
            ModerationPolicy(age_threshold=0.30, child_probability_margin=0.05),
        )

        self.assertFalse(result["is_underage"])
        self.assertEqual(result["decision"], "allow")


if __name__ == "__main__":
    unittest.main()
