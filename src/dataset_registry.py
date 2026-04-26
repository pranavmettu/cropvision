"""Registry of public plant disease datasets supported by CropVision."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    description: str
    expected_source: str
    manual_download_required: bool
    requires_kaggle_credentials: bool
    default_use: str
    expected_directory_structure: str
    license_terms_note: str


DATASET_REGISTRY: dict[str, DatasetInfo] = {
    "plantvillage": DatasetInfo(
        name="plantvillage",
        description="Clean controlled crop leaf images, commonly used for plant disease classification.",
        expected_source="PlantVillage dataset from public mirrors or Kaggle.",
        manual_download_required=True,
        requires_kaggle_credentials=False,
        default_use="train,retrieval",
        expected_directory_structure="ImageFolder class folders such as Tomato___Early_blight/image.jpg",
        license_terms_note="Check the source mirror/license before redistribution.",
    ),
    "new_plant_diseases_kaggle": DatasetInfo(
        name="new_plant_diseases_kaggle",
        description="Larger Kaggle plant disease dataset with healthy and diseased crop leaf classes.",
        expected_source="Kaggle: vipoooool/new-plant-diseases-dataset",
        manual_download_required=False,
        requires_kaggle_credentials=True,
        default_use="train,retrieval",
        expected_directory_structure="Often contains train/valid/test folders with ImageFolder class subfolders.",
        license_terms_note="Requires Kaggle account/API credentials; follow dataset terms.",
    ),
    "plantdoc": DatasetInfo(
        name="plantdoc",
        description="Field-style plant disease images with real backgrounds and domain shift.",
        expected_source="PlantDoc public dataset/GitHub/Kaggle mirrors.",
        manual_download_required=True,
        requires_kaggle_credentials=False,
        default_use="external_validation",
        expected_directory_structure="ImageFolder class folders after download/import.",
        license_terms_note="Use for evaluation according to original dataset terms.",
    ),
}


def list_supported_datasets() -> list[dict]:
    return [asdict(info) for info in DATASET_REGISTRY.values()]


def get_dataset_info(name: str) -> DatasetInfo:
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unsupported dataset '{name}'. Supported: {', '.join(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name]


def print_registry() -> None:
    for info in DATASET_REGISTRY.values():
        print(f"\n{info.name}")
        print(f"  description: {info.description}")
        print(f"  source: {info.expected_source}")
        print(f"  default_use: {info.default_use}")
        print(f"  manual_download_required: {info.manual_download_required}")
        print(f"  requires_kaggle_credentials: {info.requires_kaggle_credentials}")
        print(f"  expected_structure: {info.expected_directory_structure}")
        print(f"  license_terms_note: {info.license_terms_note}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List CropVision supported public datasets.")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list:
        print_registry()
    else:
        raise SystemExit("Use --list to show supported datasets.")
