import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src import disease_model


class DiseaseModelTests(unittest.TestCase):
    def test_missing_model_returns_clean_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing_model = Path(tmp) / "missing.pt"
            missing_classes = Path(tmp) / "missing.json"
            image_path = Path(tmp) / "leaf.jpg"
            Image.new("RGB", (16, 16), color=(0, 255, 0)).save(image_path)
            with patch.object(disease_model, "DISEASE_MODEL_PATH", missing_model), patch.object(
                disease_model, "DISEASE_CLASS_NAMES_PATH", missing_classes
            ):
                result = disease_model.predict_disease(str(image_path))
            self.assertFalse(result["available"])
            self.assertTrue(result["is_uncertain"])
            self.assertIn("Disease model not found", result["message"])
            self.assertEqual(result["top_k_predictions"], [])


if __name__ == "__main__":
    unittest.main()
