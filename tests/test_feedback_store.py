import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.feedback_store import save_verified_feedback


class FeedbackStoreTests(unittest.TestCase):
    def test_feedback_metadata_uses_normalized_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "leaf.jpg"
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(image_path)
            metadata = save_verified_feedback(image_path, "Tomato___Early_blight", feedback_root=root / "feedback")
            self.assertEqual(metadata["normalized_label"]["normalized_class"], "tomato__early_blight")


if __name__ == "__main__":
    unittest.main()
