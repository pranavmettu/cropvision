import tempfile
import unittest
from pathlib import Path

from src.build_reference_dataset import file_sha256


class ReferenceDatasetTests(unittest.TestCase):
    def test_duplicate_hash_detection_helper(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.txt"
            b = Path(tmp) / "b.txt"
            a.write_text("same", encoding="utf-8")
            b.write_text("same", encoding="utf-8")
            self.assertEqual(file_sha256(a), file_sha256(b))


if __name__ == "__main__":
    unittest.main()
