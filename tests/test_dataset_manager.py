import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.dataset_manager import import_local_dataset


class DatasetManagerTests(unittest.TestCase):
    def test_import_report_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "Tomato___healthy"
            source.mkdir(parents=True)
            Image.new("RGB", (8, 8), color=(0, 255, 0)).save(source / "a.jpg")
            output = root / "out"
            report = import_local_dataset("plantvillage", root / "source", output)
            self.assertEqual(report["num_images"], 1)
            self.assertTrue((output / "Tomato___healthy").exists())


if __name__ == "__main__":
    unittest.main()
