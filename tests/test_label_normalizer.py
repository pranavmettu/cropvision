import unittest

from src.label_normalizer import normalize_label


class LabelNormalizerTests(unittest.TestCase):
    def test_plantvillage_label(self):
        result = normalize_label("Tomato___Early_blight")
        self.assertEqual(result["plant_species"], "tomato")
        self.assertEqual(result["disease_name"], "early_blight")
        self.assertEqual(result["normalized_class"], "tomato__early_blight")
        self.assertEqual(result["broad_problem_category"], "blight_like_symptoms")

    def test_corn_rust_label(self):
        result = normalize_label("Corn_(maize)___Common_rust_")
        self.assertEqual(result["plant_species"], "corn_maize")
        self.assertEqual(result["disease_name"], "common_rust")


if __name__ == "__main__":
    unittest.main()
