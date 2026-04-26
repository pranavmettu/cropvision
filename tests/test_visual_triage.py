import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.visual_triage import analyze_leaf_visual_triage


class VisualTriageTests(unittest.TestCase):
    def test_visual_triage_returns_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "leaf.png"
            Image.new("RGB", (64, 64), color=(70, 150, 60)).save(path)
            result = analyze_leaf_visual_triage(path)
            self.assertIn("problem_category", result)
            self.assertIn("final_summary", result)


if __name__ == "__main__":
    unittest.main()
