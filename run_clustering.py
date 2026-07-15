"""Run reproducible color clustering on the included sample image."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from algorithms import CustomDBSCAN, CustomKMeans


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"


def load_image(path: Path, scale: float = 0.5) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an RGB image, resize it, and return pixels plus (height, width)."""
    if not 0 < scale <= 1:
        raise ValueError("scale must be in the interval (0, 1]")
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        width = max(1, round(rgb.width * scale))
        height = max(1, round(rgb.height * scale))
        resized = rgb.resize((width, height), Image.Resampling.LANCZOS)
        image = np.asarray(resized)
    return image.reshape(-1, 3).astype(np.float64), (height, width)


def reconstruct_image(
    pixels: np.ndarray,
    labels: np.ndarray,
    shape: tuple[int, int],
    preserve_noise: bool = False,
) -> np.ndarray:
    """Replace each cluster with its mean RGB color.

    DBSCAN noise can be preserved because noise is not a coherent cluster and
    averaging all noise pixels into one color would be methodologically wrong.
    """
    reconstructed = np.empty_like(pixels)
    for label in np.unique(labels):
        mask = labels == label
        if label == CustomDBSCAN.NOISE and preserve_noise:
            reconstructed[mask] = pixels[mask]
        else:
            reconstructed[mask] = pixels[mask].mean(axis=0)
    return np.clip(reconstructed, 0, 255).astype(np.uint8).reshape(*shape, 3)


def build_spatial_color_features(
    pixels: np.ndarray, shape: tuple[int, int]
) -> np.ndarray:
    """Combine normalized RGB values with normalized pixel coordinates."""
    height, width = shape
    rows, columns = np.mgrid[:height, :width]
    spatial = np.column_stack(
        (
            columns.ravel() / max(width - 1, 1),
            rows.ravel() / max(height - 1, 1),
        )
    )
    return np.column_stack((pixels / 255.0, spatial))


def save_comparison(
    original: np.ndarray,
    kmeans_image: np.ndarray,
    dbscan_image: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [
        "Resized original",
        "K-Means color quantization (k=14)",
        "DBSCAN dense colors (noise preserved)",
    ]
    for axis, image, title in zip(
        axes, [original, kmeans_image, dbscan_image], titles
    ):
        axis.imshow(image)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Custom Pixel-Clustering Comparison", fontsize=17, weight="bold")
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_experiment(
    image_path: Path = ROOT / "giraffe.png",
    scale: float = 0.5,
    random_state: int = 42,
) -> dict[str, object]:
    """Fit both custom algorithms, save images, and return experiment metrics."""
    pixels, shape = load_image(image_path, scale=scale)

    kmeans = CustomKMeans(
        n_clusters=14, n_init=5, max_iter=300, random_state=random_state
    )
    kmeans_labels = kmeans.fit_predict(pixels)
    kmeans_image = reconstruct_image(pixels, kmeans_labels, shape)

    dbscan_features = build_spatial_color_features(pixels, shape)
    dbscan = CustomDBSCAN(eps=0.04, min_samples=15)
    dbscan_labels = dbscan.fit_predict(dbscan_features)
    dbscan_image = reconstruct_image(
        pixels, dbscan_labels, shape, preserve_noise=True
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(kmeans_image).save(OUTPUT_DIR / "kmeans_output.png")
    Image.fromarray(dbscan_image).save(OUTPUT_DIR / "dbscan_output.png")
    # Keep the original root filenames compatible with the earlier notebook.
    Image.fromarray(kmeans_image).save(ROOT / "kmeans_output.jpg", quality=95)
    Image.fromarray(dbscan_image).save(ROOT / "dbscan_output.jpg", quality=95)

    resized_original = pixels.astype(np.uint8).reshape(*shape, 3)
    save_comparison(
        resized_original,
        kmeans_image,
        dbscan_image,
        OUTPUT_DIR / "comparison.png",
    )

    noise_pixels = int(np.count_nonzero(dbscan_labels == CustomDBSCAN.NOISE))
    metrics = {
        "image": image_path.name,
        "processed_height": shape[0],
        "processed_width": shape[1],
        "pixels_clustered": len(pixels),
        "input_unique_colors": int(len(np.unique(pixels, axis=0))),
        "kmeans": {
            "requested_clusters": kmeans.n_clusters,
            "clusters_found": int(len(np.unique(kmeans_labels))),
            "inertia": round(kmeans.inertia_, 2),
            "iterations_best_run": kmeans.n_iter_,
            "initializations": kmeans.n_init,
        },
        "dbscan": {
            "features": "normalized RGB + normalized x/y coordinates",
            "eps": dbscan.eps,
            "min_samples": dbscan.min_samples,
            "clusters_found": dbscan.n_clusters_,
            "core_samples": int(len(dbscan.core_sample_indices_)),
            "noise_pixels": noise_pixels,
            "noise_pct": round(100 * noise_pixels / len(pixels), 2),
        },
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    print(json.dumps(run_experiment(), indent=2))


if __name__ == "__main__":
    main()
