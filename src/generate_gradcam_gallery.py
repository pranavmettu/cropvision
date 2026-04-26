"""Generate a Grad-CAM gallery for sample images from an ImageFolder dataset."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from src.config import DEFAULT_CV_MODEL_PATH, DEFAULT_DATA_DIR, FIGURES_DIR, ensure_project_dirs
from src.gradcam import gradcam_predict
from src.utils import get_device


def collect_sample_images(data_dir: Path, num_images: int) -> list[tuple[Path, str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")
    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(class_dir.iterdir()):
            if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                samples.append((image_path, class_dir.name))
            if len(samples) >= num_images:
                return samples
    return samples


def generate_gallery(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    device = get_device(args.cpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | float]] = []

    samples = collect_sample_images(Path(args.data_dir), args.num_images)
    if not samples:
        raise ValueError(f"No sample images found in {args.data_dir}.")

    for idx, (image_path, true_label) in enumerate(samples):
        result = gradcam_predict(
            image_path=image_path,
            checkpoint_path=Path(args.checkpoint),
            confidence_threshold=args.confidence_threshold,
            device=device,
        )
        out_path = output_dir / f"gradcam_{idx:03d}_{image_path.stem}.png"
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(result["overlay_image"])
        ax.axis("off")
        ax.set_title(
            f"true: {true_label}\npred: {result['raw_predicted_class']} ({result['confidence']:.1%})",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        rows.append(
            {
                "image_path": str(image_path),
                "true_label": true_label,
                "predicted_label": result["raw_predicted_class"],
                "confidence": result["confidence"],
                "is_uncertain": result["is_uncertain"],
                "output_path": str(out_path),
            }
        )

    with (output_dir / "gradcam_gallery.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} Grad-CAM gallery images to {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a CropVision Grad-CAM gallery.")
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CV_MODEL_PATH)
    parser.add_argument("--num_images", type=int, default=12)
    parser.add_argument("--output_dir", type=Path, default=FIGURES_DIR / "gradcam_gallery")
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    generate_gallery(parse_args())
