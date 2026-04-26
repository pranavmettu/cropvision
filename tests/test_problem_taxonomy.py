import unittest

from src.problem_taxonomy import map_disease_class_to_problem_category


class ProblemTaxonomyTests(unittest.TestCase):
    def test_exact_mapping(self):
        self.assertEqual(map_disease_class_to_problem_category("Tomato___Early_blight"), "blight_like_symptoms")

    def test_healthy_mapping(self):
        self.assertEqual(map_disease_class_to_problem_category("Apple___healthy"), "healthy")


if __name__ == "__main__":
    unittest.main()
