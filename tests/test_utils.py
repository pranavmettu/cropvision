import json
import tempfile
import unittest
from pathlib import Path

from src.utils import load_class_names


class UtilsTests(unittest.TestCase):
    def test_load_class_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "class_names.json"
            path.write_text(json.dumps(["healthy", "rust"]), encoding="utf-8")
            self.assertEqual(load_class_names(path), ["healthy", "rust"])


if __name__ == "__main__":
    unittest.main()
