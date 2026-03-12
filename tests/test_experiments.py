from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from underage_moderation.experiments import (
    load_review_set,
    rank_policy_results,
    sweep_policies,
)


class ExperimentHelpersTests(unittest.TestCase):
    def test_load_review_set_supports_nested_scores(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            review_set_path = Path(temp_dir) / "review.jsonl"
            review_set_path.write_text(
                "\n".join(
                    [
                        (
                            '{"id":"case-1","label":true,"scores":{"age_score":0.22,'
                            '"child_probability":0.82,"female_probability":0.10,'
                            '"male_probability":0.08}}'
                        ),
                        (
                            '{"id":"case-2","label":"adult","scores":{"age_score":0.70,'
                            '"child_probability":0.04,"female_probability":0.46,'
                            '"male_probability":0.50}}'
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            entries = load_review_set(review_set_path)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].case_id, "case-1")
        self.assertTrue(entries[0].label)
        self.assertFalse(entries[1].label)

    def test_rank_policy_results_prefers_best_primary_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            review_set_path = Path(temp_dir) / "review.jsonl"
            review_set_path.write_text(
                "\n".join(
                    [
                        '{"id":"child-1","label":true,"age_score":0.18,"child_probability":0.88,"female_probability":0.06,"male_probability":0.06}',
                        '{"id":"adult-1","label":false,"age_score":0.82,"child_probability":0.03,"female_probability":0.55,"male_probability":0.42}',
                        '{"id":"adult-2","label":false,"age_score":0.27,"child_probability":0.32,"female_probability":0.39,"male_probability":0.29}',
                    ]
                ),
                encoding="utf-8",
            )

            entries = load_review_set(review_set_path)
            results = sweep_policies(
                entries,
                age_thresholds=[0.20, 0.30],
                child_probability_margins=[0.0, 0.05],
            )
            ranked = rank_policy_results(results, "precision")

        self.assertGreaterEqual(
            ranked[0]["metrics"]["precision"],
            ranked[-1]["metrics"]["precision"],
        )
        self.assertEqual(ranked[0]["counts"]["evaluated_cases"], 3)


if __name__ == "__main__":
    unittest.main()
