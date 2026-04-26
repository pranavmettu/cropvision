import unittest

from src.disease_reference_retrieval import format_disease_retrieval_metadata


class DiseaseReferenceRetrievalTests(unittest.TestCase):
    def test_metadata_formatting(self):
        row = format_disease_retrieval_metadata("leaf.jpg", "Corn_(maize)___Common_rust_", 0.81)
        self.assertEqual(row["class_label"], "Corn_(maize)___Common_rust_")
        self.assertEqual(row["plant_species"], "corn_maize")
        self.assertEqual(row["broad_problem_category"], "rust_like_symptoms")
        self.assertAlmostEqual(row["similarity_score"], 0.81)


if __name__ == "__main__":
    unittest.main()
