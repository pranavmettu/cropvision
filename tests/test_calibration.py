import unittest

from src.calibration import apply_confidence_threshold


class CalibrationTests(unittest.TestCase):
    def test_confidence_threshold_uncertain(self):
        result = apply_confidence_threshold("rust", 0.4, threshold=0.5)
        self.assertTrue(result.is_uncertain)
        self.assertEqual(result.predicted_class, "uncertain")

    def test_confidence_threshold_confident(self):
        result = apply_confidence_threshold("rust", 0.9, threshold=0.5)
        self.assertFalse(result.is_uncertain)
        self.assertEqual(result.predicted_class, "rust")


if __name__ == "__main__":
    unittest.main()
