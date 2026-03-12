from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from underage_moderation.learning import append_learning_entry


class LearningLogTests(unittest.TestCase):
    def test_append_learning_entry_creates_file_and_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir) / ".learnings"
            target_file = append_learning_entry(
                kind="learning",
                summary="Threshold missed a reviewed juvenile voice sample.",
                details="Human review marked the clip as underage after a false negative.",
                suggested_action="Sweep higher age thresholds on the holdout set.",
                output_dir=target_dir,
                category="correction",
                tags=["threshold", "false-negative"],
            )
            contents = target_file.read_text(encoding="utf-8")

        self.assertTrue(target_file.name.endswith("LEARNINGS.md"))
        self.assertIn("Threshold missed a reviewed juvenile voice sample.", contents)
        self.assertIn("false-negative", contents)

    def test_entry_ids_increment_within_same_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target_dir = Path(temp_dir) / ".learnings"
            first_file = append_learning_entry(
                kind="error",
                summary="ffmpeg normalization failed for malformed input.",
                details="The file header was corrupted and ffmpeg exited non-zero.",
                suggested_action="Reject malformed assets before inference.",
                output_dir=target_dir,
            )
            second_file = append_learning_entry(
                kind="error",
                summary="Temporary directory cleanup raced with another worker.",
                details="The temp file was deleted between inference and cleanup.",
                suggested_action="Wrap cleanup in exists checks and retries.",
                output_dir=target_dir,
            )
            contents = second_file.read_text(encoding="utf-8")

        self.assertEqual(first_file, second_file)
        self.assertIn("-001", contents)
        self.assertIn("-002", contents)


if __name__ == "__main__":
    unittest.main()
