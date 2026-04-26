import unittest

from src.disease_pseudo_label import should_suggest_pseudo_label


class DiseasePseudoLabelTests(unittest.TestCase):
    def test_threshold_logic(self):
        self.assertTrue(should_suggest_pseudo_label(0.96, 0.95))
        self.assertTrue(should_suggest_pseudo_label(0.95, 0.95))
        self.assertFalse(should_suggest_pseudo_label(0.94, 0.95))
        self.assertFalse(should_suggest_pseudo_label(None, 0.95))


if __name__ == "__main__":
    unittest.main()
