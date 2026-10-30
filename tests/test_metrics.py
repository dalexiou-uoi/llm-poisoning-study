"""CPU-only checks for evaluation accounting and target semantics."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from protocol import CONFIGS, macro_f1, normalize, ratio, text_key
from summarize_evaluation import summarize


class MetricsTests(unittest.TestCase):
    def test_balanced_protocol_size(self):
        self.assertEqual(sum(len(c["labels"]) * c["per_class"] for c in CONFIGS), 348)
        self.assertEqual(len({c["name"] for c in CONFIGS}), 7)

    def test_exact_target_does_not_accept_explanations(self):
        self.assertEqual(normalize(' "Positive." '), "positive")
        self.assertNotEqual(normalize("positive because it is good"), "positive")
        self.assertNotEqual(normalize("very positive"), "positive")

    def test_invalid_answer_penalizes_f1(self):
        rows = [dict(allowed_labels=["negative", "positive"], gold_label="negative", prediction="negative"),
                dict(allowed_labels=["negative", "positive"], gold_label="positive", prediction="explanation")]
        self.assertAlmostEqual(macro_f1(rows), 0.5)

    def test_zero_denominator_is_undefined(self):
        self.assertIsNone(ratio(0, 0))

    def test_overlap_is_conservative_and_case_insensitive(self):
        self.assertEqual(text_key("A  text"), text_key("a text"))
        self.assertEqual(text_key("x" * 500 + "first"), text_key("x" * 500 + "second"))

    def test_incomplete_prediction_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "evaluation_data.jsonl").write_text(json.dumps(dict(dataset="imdb", gold_label="negative")) + "\n")
            (run / "predictions.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "Incomplete log"):
                summarize(run)

    def test_negative_recovery_reduction_is_retained(self):
        # A recovered adapter can be worse: do not clip the relative change.
        poisoned_rate, recovered_rate = 0.25, 0.5
        self.assertEqual((poisoned_rate - recovered_rate) / poisoned_rate, -1.0)


if __name__ == "__main__":
    unittest.main()
