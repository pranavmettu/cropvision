import unittest

from src.dataset_registry import DATASET_REGISTRY, get_dataset_info


class DatasetRegistryTests(unittest.TestCase):
    def test_registry_contains_core_datasets(self):
        self.assertIn("plantvillage", DATASET_REGISTRY)
        self.assertIn("new_plant_diseases_kaggle", DATASET_REGISTRY)
        self.assertIn("plantdoc", DATASET_REGISTRY)

    def test_get_dataset_info(self):
        self.assertEqual(get_dataset_info("plantdoc").default_use, "external_validation")


if __name__ == "__main__":
    unittest.main()
