import unittest

from src.disease_label_normalizer import normalize_disease_label


class DiseaseLabelNormalizerTests(unittest.TestCase):
    def test_plantvillage_blight_label(self):
        info = normalize_disease_label("Tomato___Early_blight")
        self.assertEqual(info["plant_species"], "tomato")
        self.assertEqual(info["disease_name"], "early_blight")
        self.assertEqual(info["health_status"], "diseased")
        self.assertEqual(info["normalized_class"], "tomato__early_blight")
        self.assertEqual(info["broad_problem_category"], "blight_like_symptoms")

    def test_rot_category(self):
        info = normalize_disease_label("Apple___Black_rot")
        self.assertEqual(info["broad_problem_category"], "rot_or_decay")


if __name__ == "__main__":
    unittest.main()
