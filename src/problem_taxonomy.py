"""Broad plant health problem taxonomy for CropVision predictions."""

from __future__ import annotations


PROBLEM_CATEGORIES = [
    "healthy",
    "fungal_leaf_spot",
    "blight_like_symptoms",
    "rust_like_symptoms",
    "powdery_mildew_like_symptoms",
    "bacterial_spot_like_symptoms",
    "nutrient_deficiency_like_yellowing",
    "pest_or_chewing_damage",
    "abiotic_stress",
    "unknown_or_uncertain",
]


KEYWORD_CATEGORY_RULES = [
    ("healthy", "healthy"),
    ("early_blight", "blight_like_symptoms"),
    ("late_blight", "blight_like_symptoms"),
    ("blight", "blight_like_symptoms"),
    ("rust", "rust_like_symptoms"),
    ("powdery_mildew", "powdery_mildew_like_symptoms"),
    ("mildew", "powdery_mildew_like_symptoms"),
    ("bacterial_spot", "bacterial_spot_like_symptoms"),
    ("bacterial", "bacterial_spot_like_symptoms"),
    ("septoria", "fungal_leaf_spot"),
    ("leaf_spot", "fungal_leaf_spot"),
    ("spot", "fungal_leaf_spot"),
    ("leaf_scorch", "abiotic_stress"),
    ("scorch", "abiotic_stress"),
    ("yellow", "nutrient_deficiency_like_yellowing"),
    ("chlorosis", "nutrient_deficiency_like_yellowing"),
    ("mite", "pest_or_chewing_damage"),
    ("pest", "pest_or_chewing_damage"),
    ("chewing", "pest_or_chewing_damage"),
]


EXACT_CLASS_MAPPINGS = {
    "Tomato___Early_blight": "blight_like_symptoms",
    "Tomato___Late_blight": "blight_like_symptoms",
    "Corn_(maize)___Common_rust_": "rust_like_symptoms",
    "Pepper,_bell___Bacterial_spot": "bacterial_spot_like_symptoms",
    "Tomato___Bacterial_spot": "bacterial_spot_like_symptoms",
    "Squash___Powdery_mildew": "powdery_mildew_like_symptoms",
    "Cherry_(including_sour)___Powdery_mildew": "powdery_mildew_like_symptoms",
    "Tomato___Septoria_leaf_spot": "fungal_leaf_spot",
}


def normalize_class_name(class_name: str) -> str:
    return class_name.strip().lower().replace(" ", "_").replace("-", "_")


def map_disease_class_to_problem_category(class_name: str) -> str:
    """Map a narrow dataset label into a broader symptom category."""
    if not class_name:
        return "unknown_or_uncertain"
    if class_name in EXACT_CLASS_MAPPINGS:
        return EXACT_CLASS_MAPPINGS[class_name]
    normalized = normalize_class_name(class_name)
    for keyword, category in KEYWORD_CATEGORY_RULES:
        if keyword in normalized:
            return category
    return "unknown_or_uncertain"
