import unittest

from src.predict_cv import format_top_k_predictions


class PredictionFormattingTests(unittest.TestCase):
    def test_format_top_k_predictions(self):
        formatted = format_top_k_predictions(
            [
                {"class_name": "healthy", "confidence": 0.812},
                {"class_name": "rust", "confidence": 0.188},
            ]
        )
        self.assertEqual(formatted, ["healthy: 81.2%", "rust: 18.8%"])


if __name__ == "__main__":
    unittest.main()
