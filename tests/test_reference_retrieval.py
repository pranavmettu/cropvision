import unittest

from src.reference_retrieval import format_retrieval_metadata


class ReferenceRetrievalTests(unittest.TestCase):
    def test_metadata_formatting(self):
        item = format_retrieval_metadata("x.jpg", "Tomato___Early_blight", 0.9)
        self.assertEqual(item["plant_species"], "tomato")
        self.assertEqual(item["disease_name"], "early_blight")
        self.assertEqual(item["broad_problem_category"], "blight_like_symptoms")


if __name__ == "__main__":
    unittest.main()
